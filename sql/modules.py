import torch
import texar.torch as tx
from typing import Tuple, Dict, Union, Optional, Callable, Any, cast

from sql.utils import ForwardMode
from sql import utils as sql_utils
# from sql import replay_buffer
from sql.modules_base import SoftQModelBase
from sql.types import (
    BatchType,
    HF_BatchType,
    FloatTensor,
    LongTensor)
from modules.models import Transformer, GPT2ConditionedMLP
TexarModules = Union[Transformer, GPT2ConditionedMLP]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TXSoftQModel(SoftQModelBase):

    def __init__(
            self,
            model_constructor: Callable[[], TexarModules],
            behavior_model_constructor: Optional[Callable[[], Optional[TexarModules]]],
            sql_loss_impl: str,
            target_update_method: Optional[str],
            target_learning_rate: float,
            reward_shaping: bool,
            reward_old_min: float,
            reward_old_max: float,
            reward_shaping_min: float,
            reward_shaping_max: float,
            beam_width: int,
            reward_name: str,
            sql_loss_coefficients: Optional[float] = None,
            sql_loss_margin_constant: Optional[float] = None,
            sql_loss_margin_coefficient: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            # Hacks not implemented in parent class
            hack_truncate_length_constant: Optional[int] = None,
            # prompt reward parameters
            prompt_task_lm: Optional[str] = None,
            prompt_dataset: Optional[str] = None,
            prompt_dataset_seed: Optional[int] = None,
            prompt_dataset_basepath: Optional[str] = None,
            # text style transfer arguments
            tst_clf_basepath: Optional[str] = None,
            tst_n_repeats: Optional[int] = None,
            tst_num_samples: Optional[int] = None, # Num of samples from which to take the output
            tst_num_bootstraps: Optional[int] = None, # Num of bootstraps to reduce reward randomness
            # classification arguments
            clf_kshot: Optional[int] = None,
            clf_num_classes: Optional[int] = None,
            # # TST arguments
            # tst_dataset: Optional[str] = None,
            # tst_data_seed: Optional[int] = None,
            # # Classification arguments
            # LM_type: str = 'gpt2',
            # experiment: str = 'Test',
            # experiment_seed: int = 0,
            # kshot: int = -1,
            # task_name: str = 'SST-2',
            # Deprecated Arguments
            use_target_network: bool = True,
            target_sql_loss_impl: Optional[str] = None,
    ) -> None:
        """
        Deprectaed Features:
        1. Reply Buffer
        2. Learn the target model, using different loss functions
        """
        target_model_constructor = None
        actor_model_constructor = None
        if target_update_method in ["copy", "polyak"]:
            target_model_constructor = model_constructor

        if sql_loss_impl in ["sac"]:
            actor_model_constructor = model_constructor

        if use_target_network is False:
            raise ValueError("Deprecated")
        if target_sql_loss_impl is not None:
            raise NotImplementedError

        super().__init__(
            model_constructor=model_constructor,
            target_model_constructor=target_model_constructor,
            actor_model_constructor=actor_model_constructor,
            behavior_model_constructor=behavior_model_constructor,
            sql_loss_impl=sql_loss_impl,
            target_update_method=target_update_method,
            target_learning_rate=target_learning_rate,
            reward_shaping=reward_shaping,
            reward_shaping_old_min=reward_old_min,
            reward_shaping_old_max=reward_old_max,
            reward_shaping_new_min=reward_shaping_min,
            reward_shaping_new_max=reward_shaping_max,
            sql_loss_coefficients=sql_loss_coefficients,
            sql_loss_margin_constant=sql_loss_margin_constant,
            sql_loss_margin_coefficient=sql_loss_margin_coefficient,
            top_k=top_k,
            top_p=top_p,
            beam_width=beam_width,
            reward_name=reward_name,
            # prompt reward parameters
            prompt_task_lm=prompt_task_lm,
            prompt_dataset=prompt_dataset,
            prompt_dataset_seed=prompt_dataset_seed,
            prompt_dataset_basepath=prompt_dataset_basepath,
            # text style transfer arguments
            tst_clf_basepath=tst_clf_basepath,
            tst_n_repeats=tst_n_repeats,
            tst_num_samples=tst_num_samples, # Num of samples from which to take the output
            tst_num_bootstraps=tst_num_bootstraps, # Num of bootstraps to reduce reward randomness
            # classification arguments
            clf_kshot=clf_kshot,
            clf_num_classes=clf_num_classes,
            # tst_dataset=tst_dataset,
            # tst_data_seed=tst_data_seed,
            # LM_type=LM_type,
            # experiment=experiment,
            # experiment_seed=experiment_seed,
            # kshot=kshot,
            # task_name=task_name
            )

        if not (isinstance(self._model, Transformer) or isinstance(self._model, GPT2ConditionedMLP)):
            raise TypeError
        if not (isinstance(self._model_, Transformer) or isinstance(self._model, GPT2ConditionedMLP)):
            raise TypeError
        if self._actor_model is not None and not isinstance(self._actor_model, Transformer):
            raise TypeError
        if self._behavior_model is not None and not isinstance(self._behavior_model, Transformer):
            raise TypeError

        # Mypy stuff
        self._model: TexarModules
        self._model_: TexarModules
        self._actor_model: Optional[TexarModules]
        self._behavior_model: Optional[TexarModules]

        # Hacks
        if hack_truncate_length_constant is not None:
            sql_utils.colorful_warning(
                f"Using hack_truncate_length_constant={hack_truncate_length_constant}",
                bg="blue")
        self._hack_truncate_length_constant = hack_truncate_length_constant

    def _decode_teacher_forcing(
            self,
            batch: Union[BatchType, HF_BatchType],
            use_target: bool,
    ) -> Tuple[FloatTensor,
               Optional[FloatTensor],
               Optional[FloatTensor],
               LongTensor,
               Optional[LongTensor],
               LongTensor,
               LongTensor]:

        batch = cast(BatchType, batch)
        outputs, _ = self._model(
            batch=batch,
            mode=ForwardMode.MLE)

        logits_ = None
        if use_target is True:
            outputs_, _ = self._model_(
                batch=batch,
                mode=ForwardMode.SQL_OFF_GT)
            logits_ = outputs_.logits.contiguous()

        logits_pi = None
        sampled_outputs_id = None
        if self.is_actor_critic is True:
            if self._actor_model is None:
                raise ValueError

            outputs_pi, _ = self._actor_model(
                batch=batch,
                mode=ForwardMode.SQL_OFF_GT)
            logits_pi = outputs_pi.logits.contiguous()

            # Add top-P/K here?
            sampled_outputs_id = (
                torch.distributions
                .Categorical(logits=logits_pi)
                .sample()
                .contiguous())

        return (
            outputs.logits.contiguous(),
            logits_,
            logits_pi,
            batch["target_text_ids"][:, 1:].contiguous(),
            sampled_outputs_id,
            batch["target_text_ids"][:, 1:].contiguous(),
            batch["target_length"].contiguous() - 1,
        )

    def _decode_behavior_forcing(
            self,
            batch: Union[BatchType, HF_BatchType],
            use_target: bool,
    ) -> Tuple[FloatTensor,
               Optional[FloatTensor],
               Optional[FloatTensor],
               LongTensor,
               Optional[LongTensor],
               LongTensor,
               LongTensor]:

        batch = cast(BatchType, batch)
        if self._behavior_model is None:
            raise ValueError

        # We do not need gradients from the behavior model
        with torch.no_grad():
            behavior_outputs, behavior_sample_lengths = self._behavior_model(
                batch=batch,
                # `SQL_ON` and `PG` would be the same here
                mode=ForwardMode.SQL_ON,
                top_k=self._top_k,
                top_p=self._top_p)

        # Note:
        # Here the `behavior_batch[target_text]` will be populated
        # with `batch[target_text]` but `behavior_batch[target_text_*]`
        # will be populated with `outputs_id`. This is fine as when
        # calculating rewards, `target_text` will be used instead
        # of `target_text_ids`.
        behavior_batch = sql_utils.make_batch_from_outputs(
            batch=batch,
            outputs_id=behavior_outputs.sample_id,
            sequence_lengths=behavior_sample_lengths,
            target_vocab=self._model.target_vocab,
            include_target_text="ground-truth")

        return self._decode_teacher_forcing(
            batch=behavior_batch,
            use_target=use_target)

    def _decode_sampling(
            self,
            batch: Union[BatchType, HF_BatchType],
            use_target: bool,
    ) -> Tuple[FloatTensor,
               Optional[FloatTensor],
               Optional[FloatTensor],
               LongTensor,
               Optional[LongTensor],
               LongTensor,
               LongTensor]:

        batch = cast(BatchType, batch)
        if not self.is_actor_critic:
            outputs, sample_lengths = self._model(
                batch=batch,
                mode=ForwardMode.SQL_ON,
                top_k=self._top_k,
                top_p=self._top_p)

            batch_ = sql_utils.make_batch_from_outputs(
                batch=batch,
                outputs_id=outputs.sample_id,
                sequence_lengths=sample_lengths,
                target_vocab=self._model.target_vocab)

            logits_pi = None
            sampled_outputs_id = None

        else:
            if self._actor_model is None:
                raise ValueError

            outputs_pi, sample_lengths_pi = self._actor_model(
                batch=batch,
                mode=ForwardMode.SQL_ON,
                top_k=self._top_k,
                top_p=self._top_p)

            batch_ = sql_utils.make_batch_from_outputs(
                batch=batch,
                outputs_id=outputs_pi.sample_id,
                sequence_lengths=sample_lengths_pi,
                target_vocab=self._model.target_vocab)

            outputs, sample_lengths = self._model(
                batch=batch_,
                mode=ForwardMode.SQL_OFF_GT)

            logits_pi = outputs_pi.logits.contiguous()
            sampled_outputs_id = outputs_pi.sample_id.contiguous()
            if not (sample_lengths == sample_lengths_pi).all().item():
                raise ValueError

        # Use off-policy because the target
        # has to follow the steps taken by the model
        outputs_, sample_lengths_ = self._model_(
            batch=batch_,
            mode=ForwardMode.SQL_OFF_GT)

        # Sanity check
        if sample_lengths_ is not None:
            # In `transformer`, this will be None, so skip the check.
            if not (sample_lengths == sample_lengths_).all().item():
                raise ValueError

        if self._hack_truncate_length_constant is not None:
            # Truncate length beyond a specified constant
            # beyond ground truth length.
            length_to_truncate = (
                batch["target_length"] - 1 +
                self._hack_truncate_length_constant)
            sample_lengths = torch.minimum(
                sample_lengths,
                length_to_truncate)

        return (
            outputs.logits.contiguous(),
            outputs_.logits.contiguous(),
            logits_pi,
            outputs.sample_id.contiguous(),
            sampled_outputs_id,
            batch["target_text_ids"][:, 1:].contiguous(),
            sample_lengths.contiguous(),
        )

    def _compute_rewards(
            self,
            batch: Union[BatchType, HF_BatchType],
            outputs_id: LongTensor,
            labels: LongTensor,
            sequence_lengths: LongTensor,
    ) -> Tuple[FloatTensor, FloatTensor, Dict[str, Any]]:
        # Decode the outputs
        source_texts = tx.utils.strip_special_tokens(
            batch["source_text"],
            is_token_list=True)
        target_texts = tx.utils.strip_special_tokens(
            [text[1:] for text in batch["target_text"]],
            is_token_list=True)
        output_texts = tx.data.vocabulary.map_ids_to_strs(
            ids=outputs_id.cpu(),
            vocab=self._model.target_vocab)

        if self._hack_truncate_length_constant is not None:
            # Special handling of when the `outputs_id` are
            # not truncated but the `sequence_lengths` are
            # truncated. This would cause some mismatch in
            # reward computation, so manually fix it here.
            output_texts = sql_utils.map_ids_to_strs_truncated(
                outputs_id=outputs_id.cpu(),
                sequence_lengths=sequence_lengths,
                vocab=self._model.target_vocab)

        rewards_tensor, rewards_log = self._reward_function(
            sources=[" ".join(tokens) for tokens in source_texts],
            targets=[" ".join(tokens) for tokens in target_texts],
            predictions=output_texts,
            to_tensor=True,
            mode="train")

        rewards_tensor = rewards_tensor.to(device)
        shaped_rewards_tensor = self._reward_shaping_func(rewards_tensor)
        return rewards_tensor, shaped_rewards_tensor, rewards_log

    def _forward_decoding(
            self,
            batch: Union[BatchType, HF_BatchType],
    ) -> Tuple[Dict, Dict]:
        if not self.is_actor_critic:
            outputs = self._model(
                batch=batch,
                mode=ForwardMode.INFER,
                beam_width=self._beam_width)
        else:
            if self._actor_model is None:
                raise ValueError

            outputs = self._actor_model(
                batch=batch,
                mode=ForwardMode.INFER,
                beam_width=self._beam_width)

        return outputs, {}

    def forward(
            self,
            mode: ForwardMode,
            batch: BatchType,
    ) -> Tuple[Union[FloatTensor, Dict], Dict[str, Any]]:

        if mode == ForwardMode.INFER:
            return self._forward_decoding(batch=batch)

        _, loss, loss_log = self._forward(
            mode=mode,
            batch=batch)

        return loss, loss_log


    # def forward_SQL(self, batch: BatchType, mode: ForwardMode) -> Tuple[FloatTensor, Dict[str, Any]]:

    #     if self._target_update_method == "learn":
    #         # Detach target logits when computing model loss, and
    #         # detach model logits when computing target loss
    #         sql_loss, sql_loss_log = sql_losses.soft_q_loss_with_sparse_rewards(
    #             implementation=self._sql_loss_impl,
    #             actions=actions,
    #             logits=logits,
    #             logits_=logits_.detach(),
    #             rewards=shaped_rewards,
    #             sequence_length=sample_lengths,
    #             coefficient=self._sql_loss_coefficients)

    #         for i in range(len(target_outputs.logits_collections)):
    #             target_loss, target_loss_log = sql_losses.soft_q_loss_with_sparse_rewards(
    #                 implementation=self._target_sql_loss_impl,
    #                 actions=actions,
    #                 logits=target_outputs.logits_collections[i],
    #                 logits_=logits.detach(),
    #                 rewards=shaped_rewards,
    #                 sequence_length=sample_lengths,
    #                 coefficient=self._sql_loss_coefficients)

    #             sql_loss = sql_loss + target_loss
    #             sql_utils.add_prefix_to_dict_keys_inplace(
    #                 target_loss_log,
    #                 prefix=f"target-{i}/")
    #             sql_loss_log = unionize_dicts([
    #                 sql_loss_log, target_loss_log])
