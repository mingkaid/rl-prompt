# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attentional Seq2seq.
"""

import os
import torch
import wandb
import click
import importlib
import omegaconf
import numpy as np
import texar.torch as tx
from copy import deepcopy
from itertools import chain
from functools import partial
from typing import Any, List, Tuple, Dict, Callable, Optional, cast
import random

from modules.models import Transformer, GPT2ConditionedMLP
# from modules.metrics import compute_GEM_metrics_from_outputs
from sql.types import BatchType
from sql.utils import ForwardMode
from sql.utils import colorful_warning
from sql.utils import preprocess_target_texts
from sql.utils import add_prefix_to_dict_keys_inplace
from sql.modules import TXSoftQModel
from sql.misc_utils import unionize_dicts
from sql.misc_utils import nested_detach_and_clone
from configs.models import config_model_optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS_TARGET_TEXTS = False

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _modify_model_config(config: omegaconf.DictConfig) -> None:
    config_model_optimizer.opt["optimizer"]["kwargs"]["lr"] = config.learning_rate
    if config.gradient_clipping is True:
        config_model_optimizer.opt["gradient_clip"] = {
            "type": "torch.nn.utils.clip_grad_norm_",
            "kwargs": {
                "max_norm": 5.0
            }
        }


# What's the type hint for Module?
def prepare_data(config_data: Any,
                 device=device) -> Tuple[tx.data.PairedTextData,
                                            tx.data.PairedTextData,
                                            tx.data.PairedTextData,
                                            tx.data.TrainTestDataIterator]:

    train_data = tx.data.PairedTextData(
        hparams=config_data.train, device=device)
    val_data = tx.data.PairedTextData(
        hparams=config_data.val, device=device)
    test_data = tx.data.PairedTextData(
        hparams=config_data.test, device=device)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)

    return (
        train_data,
        val_data,
        test_data,
        data_iterator)


def prepare_model(
        config: omegaconf.DictConfig,
        train_data: tx.data.PairedTextData,
        max_source_length: int,
        max_decoding_length: int,
        use_behavior_model: bool = False,
        device=device,
) -> TXSoftQModel:
    
    valid_models = ["transformer_small", "gpt2_conditioned_mlp"]
    if config.architecture not in valid_models:
        raise ValueError

    if config.architecture in ["transformer_small"]:
        ModelClass: Callable = partial(
            Transformer,
            config_name=config.architecture)
    if config.architecture in ["gpt2_conditioned_mlp"]:
        ModelClass: Callable = partial(
            GPT2ConditionedMLP,
            config_name=config.architecture,
            device=0,
            # TST arguments
            tst_dataset=config.tst_dataset,
            tst_data_seed=config.tst_data_seed,
            policy_lm='distilgpt2',
            input_specific=True,
            logit_bias=0,
            dataset=config.tst_dataset,
            dataset_seed=config.tst_data_seed,
            dataset_basepath='/data/mingkai/prompt-generation/dirty-code/rl-prompt',
            n_repeats=4,
            fluent_prompt=False)

    behavior_model = None
    if use_behavior_model is True:
        behavior_checkpoint_path = "/export/share/Experiments/20210330/MLE-model.e2e.pth"
        behavior_model = ModelClass(train_data=train_data)
        behavior_model.load_state_dict(
            torch.load(behavior_checkpoint_path))
        # Yes, we use training mode (for better exploration, maybe?)
        behavior_model.train()
        colorful_warning(f"Loaded behavior model from "
                         f"{behavior_checkpoint_path}",
                         bg="blue")

    model = TXSoftQModel(
        model_constructor=(
            lambda: ModelClass(
                train_data=train_data,
                max_source_length=max_source_length,
                max_decoding_length=max_decoding_length)
        ),
        behavior_model_constructor=(
            lambda: behavior_model),
        sql_loss_impl=config.sql_loss_impl,
        target_update_method=config.target_sync_method,
        target_learning_rate=config.target_learning_rate,
        reward_shaping=config.reward_shaping,
        reward_old_min=config.reward_old_min,
        reward_old_max=config.reward_old_max,
        reward_shaping_min=config.reward_shaping_min,
        reward_shaping_max=config.reward_shaping_max,
        sql_loss_coefficients=config.sql_loss_coefficients,
        sql_loss_margin_constant=config.sql_loss_margin_constant,
        sql_loss_margin_coefficient=config.sql_loss_margin_coefficient,
        top_k=config.top_k,
        top_p=config.top_p,
        beam_width=config.beam_width,
        reward_name=config.reward_name,
        # Hacks not implemented in parent class
        hack_truncate_length_constant=config.hack_truncate_length_constant,
        # TST arguments
        tst_dataset=config.tst_dataset,
        tst_data_seed=config.tst_data_seed,
        # Classification arguments
        LM_type=config.LM_type,
        experiment=config.experiment,
        experiment_seed=config.experiment_seed,
        kshot=config.kshot,
        task_name=config.task,
        # Deprecated Arguments
        use_target_network=config.use_target_network,
        target_sql_loss_impl=config.target_sql_loss_impl
    )
    model.to(device)

    # if config.base_checkpoint_path is not None:
    #     model.load_checkpoint(config.base_checkpoint_path)

    if config.checkpoint_path is not None:
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        print(click.style(f"Loaded model from {config.checkpoint_path}", fg="green"))

    return model


def prepare_train_ops(
        config: omegaconf.DictConfig,
        model: TXSoftQModel
) -> Tuple[Callable, Optional[Callable]]:

    if model._actor_model is None:
        train_op = tx.core.get_train_op(
            params=model._model.parameters(),
            hparams=config_model_optimizer.opt)
    else:
        train_op = tx.core.get_train_op(
            params=chain(
                model._model.parameters(),
                model._actor_model.parameters()),
            hparams=config_model_optimizer.opt)

    target_train_op = None
    if config.target_sync_method == "learn":
        target_opt_config = deepcopy(config_model_optimizer.opt)
        target_opt_config["optimizer"]["kwargs"]["lr"] = (
            config.target_learning_rate)
        target_train_op = tx.core.get_train_op(
            params=model._target_model.parameters(),
            hparams=target_opt_config)

    return train_op, target_train_op


def main(config: omegaconf.DictConfig) -> None:
    print(click.style(omegaconf.OmegaConf.to_yaml(config), fg="red"))
    # Looks like a hack
    set_random_seed(config.random_seed)
    # 2 worked somewhat
    wandb.init(project="soft-Q-learning", config=eval(str(config)))
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Modify the model config
    _modify_model_config(config)
    # Choosing task dynamically
    config_data: Any = importlib.import_module(
        f"configs.data.{config.task_name}")
    # Prepare data, model, and ops
    train_data, val_data, test_data, data_iterator = prepare_data(
        config_data=config_data)
    model = prepare_model(
        config=config,
        train_data=train_data,
        # When `config_data` does not specify `max_source_length`
        # we will use `max_decoding_length` instead.
        max_source_length=getattr(
            config_data,
            "max_source_length",
            config_data.max_decoding_length),
        max_decoding_length=config_data.max_decoding_length)
    train_op, target_train_op = prepare_train_ops(
        config=config,
        model=model)
    wandb.watch(model, log=None)

    def _train_epoch(training_mode: str) -> List[Dict]:
        print('Start Train')
        data_iterator.switch_to_train_data()
        model.train()

        epoch_logs = []
        for step, batch in enumerate(data_iterator):

            # Process the target texts so that they share
            # similar vocabularies like `<UNK>`.
            if PREPROCESS_TARGET_TEXTS is True:
                if not isinstance(batch, tx.data.Batch):
                    raise TypeError
                batch._batch["target_text"] = preprocess_target_texts(
                    tokens_or_list_of_tokens=batch["target_text"],
                    vocab=model._model.target_vocab,
                    remove_special_tokens=False)

            # Do not sync when we learn the target model
            if config.target_sync_method == "learn":
                if target_train_op is None:
                    raise ValueError
                target_train_op()

            # If we use polyak-averaging
            # just do update every step
            if config.target_sync_method == "polyak":
                model.sync_target_model()

            if (config.target_sync_method == "copy" and
                    step % config.target_sync_steps == 0):
                model.sync_target_model()

            if training_mode == "sql-mixed":
                candidate_modes = [
                    ForwardMode.SQL_OFF_GT,
                    ForwardMode.SQL_ON]

                if config.mix_strategy == "alternate":
                    modes = [candidate_modes[step % len(candidate_modes)]]

                if config.mix_strategy == "mix":
                    modes = candidate_modes

            elif training_mode == "sql-mixed-behavior":
                candidate_modes = [
                    ForwardMode.SQL_OFF_GT,
                    ForwardMode.SQL_OFF_BEHAVIOR]

                if config.mix_strategy == "alternate":
                    modes = [candidate_modes[step % len(candidate_modes)]]

                if config.mix_strategy == "mix":
                    modes = candidate_modes

            elif training_mode == "sql-mixed-everything":
                candidate_modes = [
                    ForwardMode.SQL_OFF_GT,
                    ForwardMode.SQL_ON,
                    ForwardMode.SQL_OFF_BEHAVIOR]

                if config.mix_strategy == "alternate":
                    modes = [candidate_modes[step % len(candidate_modes)]]

                if config.mix_strategy == "mix":
                    modes = candidate_modes

            elif training_mode == "pg-mixed":
                candidate_modes = [
                    ForwardMode.MLE,
                    ForwardMode.PG]

                if config.mix_strategy == "alternate":
                    modes = [candidate_modes[step % len(candidate_modes)]]

                if config.mix_strategy == "mix":
                    modes = candidate_modes

            else:
                training_mode_map = {
                    "mle": ForwardMode.MLE,
                    "pg": ForwardMode.PG,
                    "sql-onpolicy": ForwardMode.SQL_ON,
                    "sql-offpolicy": ForwardMode.SQL_OFF_GT,
                }

                modes = [training_mode_map[training_mode]]# , ForwardMode.SQL_ON]

            loss_list = []
            additional_info_list = []
            for mode in modes:
                # print(batch)
                print(mode)
                _loss, _additional_info = model(
                    mode=mode,
                    batch=batch)

                loss_list.append(_loss)
                additional_info_list.append(_additional_info)

            # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/2
            loss = torch.mean(torch.stack(loss_list))
            additional_info = unionize_dicts(additional_info_list)

            loss.backward()
            train_op()

            batch_log = nested_detach_and_clone(
                additional_info, to_cpu=True)
            epoch_logs.append(batch_log)
            wandb.log(batch_log)

            if (config.num_batches_per_epoch is not None and
                    config.num_batches_per_epoch == step):
                break
                
        print('Finish Train')

        return epoch_logs
    
    def _warmup_epoch(training_mode: str) -> List[Dict]:
        print('Start Warmup')
        data_iterator.switch_to_train_data()
        model.train()

        epoch_logs = []
        for step, batch in enumerate(data_iterator):

            # Process the target texts so that they share
            # similar vocabularies like `<UNK>`.
            if PREPROCESS_TARGET_TEXTS is True:
                if not isinstance(batch, tx.data.Batch):
                    raise TypeError
                batch._batch["target_text"] = preprocess_target_texts(
                    tokens_or_list_of_tokens=batch["target_text"],
                    vocab=model._model.target_vocab,
                    remove_special_tokens=False)

            # Do not sync when we learn the target model
            if config.target_sync_method == "learn":
                if target_train_op is None:
                    raise ValueError
                target_train_op()

            # If we use polyak-averaging
            # just do update every step
            if config.target_sync_method == "polyak":
                model.sync_target_model()

            if (config.target_sync_method == "copy" and
                    step % config.target_sync_steps == 0):
                model.sync_target_model()

            if training_mode == "sql-mixed":
                candidate_modes = [
                    ForwardMode.SQL_OFF_GT,
                    ForwardMode.SQL_ON]

                if config.mix_strategy == "alternate":
                    modes = [candidate_modes[step % len(candidate_modes)]]

                if config.mix_strategy == "mix":
                    modes = candidate_modes

            elif training_mode == "sql-mixed-behavior":
                candidate_modes = [
                    ForwardMode.SQL_OFF_GT,
                    ForwardMode.SQL_OFF_BEHAVIOR]

                if config.mix_strategy == "alternate":
                    modes = [candidate_modes[step % len(candidate_modes)]]

                if config.mix_strategy == "mix":
                    modes = candidate_modes

            elif training_mode == "sql-mixed-everything":
                candidate_modes = [
                    ForwardMode.SQL_OFF_GT,
                    ForwardMode.SQL_ON,
                    ForwardMode.SQL_OFF_BEHAVIOR]

                if config.mix_strategy == "alternate":
                    modes = [candidate_modes[step % len(candidate_modes)]]

                if config.mix_strategy == "mix":
                    modes = candidate_modes

            elif training_mode == "pg-mixed":
                candidate_modes = [
                    ForwardMode.MLE,
                    ForwardMode.PG]

                if config.mix_strategy == "alternate":
                    modes = [candidate_modes[step % len(candidate_modes)]]

                if config.mix_strategy == "mix":
                    modes = candidate_modes

            else:
                training_mode_map = {
                    "mle": ForwardMode.MLE,
                    "pg": ForwardMode.PG,
                    "sql-onpolicy": ForwardMode.SQL_ON,
                    "sql-offpolicy": ForwardMode.SQL_OFF_GT,
                }

                modes = [training_mode_map[training_mode]]# , ForwardMode.SQL_ON]

            loss_list = []
            additional_info_list = []
            for mode in modes:
                # print(batch)
                print(mode)
                _loss, _additional_info = model(
                    mode=mode,
                    batch=batch)

                loss_list.append(_loss)
                additional_info_list.append(_additional_info)

            # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/2
            loss = torch.mean(torch.stack(loss_list))
            additional_info = unionize_dicts(additional_info_list)

            # loss.backward()
            # train_op()

            batch_log = nested_detach_and_clone(
                additional_info, to_cpu=True)
            epoch_logs.append(batch_log)
            wandb.log(batch_log)

            if (config.num_warmup_batches_per_epoch is not None and
                    config.num_warmup_batches_per_epoch == step):
                break
                
        print('Finish Warmup')

        return epoch_logs

    @torch.no_grad()
    def _eval_epoch(mode: str, save_base_path: Optional[str] = None) -> Dict[str, np.number]:
        print('Running Eval')
        if mode == "val":
            # print('Running validation')
            data_iterator.switch_to_val_data()
            unique_pairs_file = getattr(
                config_data, "val_unique_pairs_file", None)
        else:
            # print('Running test')
            data_iterator.switch_to_test_data()
            unique_pairs_file = getattr(
                config_data, "test_unique_pairs_file", None)

        if unique_pairs_file is not None:
            if PREPROCESS_TARGET_TEXTS is True:
                raise NotImplementedError

            # The loaded data is a `defaultdict`,
            # but we do not want `defaultdict` behavior here.
            source_target_pairs = dict(torch.load(unique_pairs_file))

            def _get_list_of_targets_from_batch(batch: BatchType) -> List[List[str]]:
                source_texts = tx.utils.strip_special_tokens(
                    batch["source_text"], is_token_list=True)
                source_texts = [" ".join(text) for text in source_texts]
                return [source_target_pairs[text] for text in source_texts]

        else:
            def _get_list_of_targets_from_batch(batch: BatchType) -> List[List[str]]:
                # Here `target_texts` are list of string tokens of one reference
                target_texts_ori = [text[1:] for text in batch["target_text"]]
                if PREPROCESS_TARGET_TEXTS is True:
                    target_texts_ori = cast(
                        List[List[str]],
                        preprocess_target_texts(
                            tokens_or_list_of_tokens=target_texts_ori,
                            vocab=model._model.target_vocab,
                            remove_special_tokens=False))
                target_texts = tx.utils.strip_special_tokens(
                    target_texts_ori, is_token_list=True)
                return [[" ".join(text)] for text in target_texts]

        model.eval()

        srcs = []
        refs = []
        hypos = []
        for batch in data_iterator:
            infer_outputs, _ = model(
                mode=ForwardMode.INFER,
                batch=batch)
            output_ids = infer_outputs["sample_id"][:, :, 0].cpu()

            # Here `target_texts` are list of string of all references
            source_texts = [
                " ".join(text) for text in
                tx.utils.strip_special_tokens(
                    batch["source_text"],
                    is_token_list=True)]
            target_texts = _get_list_of_targets_from_batch(batch)
            output_texts = tx.data.vocabulary.map_ids_to_strs(
                ids=output_ids, vocab=val_data.target_vocab)

            for src, ref, hypo in zip(source_texts, target_texts, output_texts):
                srcs.append(src)
                refs.append(ref)
                hypos.append(hypo)

        if config.reward_name in ["bleu", "bleu+bleurt"]:
            score_log = {}
            score = tx.evals.corpus_bleu_moses(
                list_of_references=refs,
                hypotheses=hypos)

        if config.reward_name in ["rouge", "bleurt", "sentiment", "gpt2-topic", "gpt2-bleu", 
                                  "gpt2-bleu-sentiment", "gpt2-bleu-no-input", "gpt2-sentiment-no-input",
                                  "gpt2-sentiment-bleu-no-input", "gpt2-sentiment-bertscore-no-input",
                                  'gpt2-trigger','gpt2-classifier',
                                  "entailment", "entailment2", "entailment3", "toxicity"]:
            if unique_pairs_file is not None:
                colorful_warning("Only taking the first reference. "
                                 "This might lead to incorrect results.")

            score, score_log = model._reward_function(
                sources=srcs,
                # here `refs` are list of lists
                targets=[ref[0] for ref in refs],
                predictions=hypos,
                to_tensor=True,
                mode="infer")

            score = score.mean().item()

        # Scores for the GEM benchmark
#         gem_scores_dict = compute_GEM_metrics_from_outputs(
#             sources=srcs,
#             list_of_targets=refs,
#             predictions=hypos,
#             base_file_name_to_dump=save_base_path)

        add_prefix_to_dict_keys_inplace(
            score_log,
            prefix=f"{mode}/rewards/")
#         add_prefix_to_dict_keys_inplace(
#             gem_scores_dict,
#             prefix=f"{mode}/GEM/")
        
        print('Finish Eval')

        return unionize_dicts([
            score_log,
            # gem_scores_dict,
            {
                f"{mode}/score": score,
                f"{mode}/target_length": np.mean([
                    len(text.split())
                    for texts in refs
                    for text in texts]),
                f"{mode}/output_length": np.mean([
                    len(text.split())
                    for text in hypos])
            }
        ])

    # Create directory to save validation outputs
    valid_save_dir = os.path.join(config.save_dir, "validation")
    if not os.path.exists(valid_save_dir):
        os.makedirs(valid_save_dir)
        # Save outputs in this directory to `wandb`
        # wandb.save(os.path.join(valid_save_dir, "*.json"))

    for i in range(config.num_epochs):
        # Warming up scheduling
        if (
            (config.warmup_num_epochs is not None) and
            (i < config.warmup_num_epochs)
        ):
            _warmup_epoch(config.warmup_training_mode)
        else:
            _train_epoch(config.training_mode)

        save_base_path = os.path.join(
            valid_save_dir,
            f"epoch-{i}")
        eval_log = _eval_epoch(
            mode="val",
            save_base_path=save_base_path)
        wandb.log(eval_log)

        if i % config.save_frequency == 0:
            torch.save({
                "epoch": i,
                "model_state_dict": model.state_dict(),
                # "train_logs": train_logs,
            }, os.path.join(config.save_dir, f"./outputs.{i}.pth"))


@torch.no_grad()
def get_epoch_logs(
        model: TXSoftQModel,
        data_iterator: tx.data.TrainTestDataIterator,
        mode: str,
        method: str,
        beam_width: int = 1,
        num_batches: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
) -> List[Dict[str, Any]]:

    if mode not in ["train", "val"]:
        raise ValueError
    if method not in ["sampling", "beam-search"]:
        raise ValueError

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(click.style(f"Loaded model from {checkpoint_path}", fg="green"))

    if mode == "train":
        data_iterator.switch_to_train_data()
        model.train()
    if mode == "val":
        data_iterator.switch_to_val_data()
        model.eval()

    epoch_logs = []
    for step, batch in enumerate(data_iterator):

        if method == "sampling":
            (logits,
             _,
             _,
             outputs_id,
             _,
             labels,
             sequence_lengths) = model._decode_sampling(
                batch=batch,
                use_target=False)

        if method == "beam-search":
            infer_outputs = model._model(
                batch=batch,
                mode=ForwardMode.INFER,
                beam_width=beam_width)
            logits = infer_outputs.get("logits")
            outputs_id = infer_outputs["sample_id"][:, :, 0].cpu()

        raw_rewards, shaped_rewards, rewards_log = model._compute_rewards(
            batch=batch,
            outputs_id=outputs_id,
            labels=None,  # actually unused
            sequence_lengths=None)  # actually unused

        epoch_logs.append({
            "batch": batch,
            "logits": logits,
            "outputs_id": outputs_id.cpu().numpy(),
            "raw_rewards": raw_rewards,
            "rewards_log": rewards_log,
        })

        if num_batches is not None and step >= num_batches:
            break

    return epoch_logs
