import json
import torch
import click
import warnings
import numpy as np
# import torch.nn.functional as F
import texar.torch as tx
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum
from graphviz import Digraph
from torchmetrics.functional import bleu_score

from sql.types import FloatTensor, LongTensor, BatchType
from typing import Union, Tuple, Dict, Any, Optional, Callable, List, cast


class ForwardMode(Enum):
    MLE = "MLE"
    PG = "PG"
    SQL_ON = "SQL_ON"
    SQL_OFF_GT = "SQL_OFF_GT"
    SQL_OFF_RB = "SQL_OFF_RB"
    SQL_OFF_BEHAVIOR = "SQL_OFF_BEHAVIOR"
    INFER = "INFER"


def get_reward_shaping_func(
        old_min: float,
        old_max: float,
        new_min: float,
        new_max: float
) -> Callable[[FloatTensor], FloatTensor]:
    def _shaping_func(reward: FloatTensor) -> FloatTensor:
        percentile = (reward - old_min) / (old_max - old_min)
        return percentile * (new_max - new_min) + new_min

    return _shaping_func


def compute_sentence_bleu_batch(
        output_texts: List[List[str]],
        target_texts: List[List[str]],
        method: Optional[str] = None,
) -> List[float]:
    if method is None:
        method = "moses"

    if method not in ["moses", "lightning", "accuracy"]:
        raise ValueError

    if len(output_texts) != len(target_texts):
        raise ValueError

    if not all([
            isinstance(output_texts, list),
            isinstance(target_texts, list),
            isinstance(output_texts[0], list),
            isinstance(target_texts[0], list)]):
        raise TypeError

    rewards = []
    for hypo, ref in zip(output_texts, target_texts):

        if method == "moses":
            reward = tx.evals.sentence_bleu_moses(
                references=[ref],
                hypothesis=hypo)

        if method == "lightning":
            reward = bleu_score(
                # official code for `bleu_score`
                # probably has wrong type annotations
                reference_corpus=[[ref]],
                translate_corpus=[hypo])
            reward = (reward * 100).item()

        if method == "accuracy":
            reward = compute_accuracy_scores([ref], [hypo])
            reward = reward * 100

        rewards.append(reward)

    return rewards


def visualize_trajectory(
        logits: FloatTensor,
        sample_id: LongTensor,
        ground_truth_id: LongTensor,
        id_to_token_map: Dict[int, str],
        size_string: Optional[str] = None,
) -> Digraph:

    if size_string is None:
        size_string = "7.5,7.5"

    tokens = list(id_to_token_map.values())
    cmap = plt.get_cmap("magma")
    norm = mpl.colors.Normalize(
        vmin=logits.min(),
        vmax=logits.max())

    def _get_rgb_string(x: FloatTensor) -> str:
        rgb_tuple = cmap(norm(x.item()))
        rgb_string_tuple = [str(x) for x in rgb_tuple]
        return " ".join(rgb_string_tuple)

    dot = Digraph(
        engine="neato",
        graph_attr={
            "size": size_string,
        }
    )

    # for step in range(0, max(sample_id.shape[-1],
    #                          ground_truth_id.shape[-1])):
    for step in range(0, sample_id.shape[-1], 0):
        for index, token in enumerate(tokens):
            dot.node(
                name=f"{step}-{token}",
                label=f"{token}",
                style="filled,bold",
                shape="box",
                pos=f"{step * 1.1},{index + 1}!",
                color=_get_rgb_string(logits[step, index]),
            )
            if index > 0:
                dot.edge(
                    f"{step}-{tokens[index - 1]}",
                    f"{step}-{tokens[index]}",
                    dir="none")

    for step in range(1, sample_id.shape[-1]):
        last_token_index = cast(int, sample_id[step - 1].item())
        this_token_index = cast(int, sample_id[step].item())
        last_token = id_to_token_map[last_token_index]
        this_token = id_to_token_map[this_token_index]
        dot.edge(
            f"{step-1}-{last_token}",
            f"{step}-{this_token}",
            penwidth="5.0",
            color="black")

    for step in range(1, ground_truth_id.shape[-1]):
        last_token_index = cast(int, ground_truth_id[step - 1].item())
        this_token_index = cast(int, ground_truth_id[step].item())
        last_token = id_to_token_map[last_token_index]
        this_token = id_to_token_map[this_token_index]
        dot.edge(
            f"{step-1}-{last_token}",
            f"{step}-{this_token}",
            penwidth="5.0",
            color="blue")

    return dot


def add_prefix_to_dict_keys_inplace(
        d: Dict[str, Any],
        prefix: str,
        keys_to_exclude: Optional[List[str]] = None,
) -> None:

    # https://stackoverflow.com/questions/4406501/change-the-name-of-a-key-in-dictionary
    keys = list(d.keys())
    for key in keys:
        if keys_to_exclude is not None and key in keys_to_exclude:
            continue

        new_key = f"{prefix}{key}"
        d[new_key] = d.pop(key)


def make_batch_from_outputs(
        batch: BatchType,
        outputs_id: LongTensor,
        sequence_lengths: LongTensor,
        target_vocab: tx.data.Vocab,
        include_target_text: Optional[str] = None,
) -> BatchType:
    # Perhaps not necessary, but use this just in case
    with torch.no_grad():
        # Add the start token from the ground truth targets
        model_outputs = torch.cat([
            batch["target_text_ids"][:, [0]],
            outputs_id], dim=-1)

        # We are most likely not going to need it
        model_output_tokens_extended = None
        if include_target_text == "ground-truth":
            model_output_tokens_extended = batch["target_text"]

        if include_target_text == "outputs":
            model_output_tokens = tx.data.vocabulary.map_ids_to_strs(
                ids=outputs_id.cpu(),
                vocab=target_vocab,
                join=False,
                strip_pad=None,
                strip_bos=None,
                strip_eos=None,
            ).tolist()

            model_output_tokens_extended = [
                ([target_vocab.bos_token] + tokens)
                for tokens in model_output_tokens]

        # The target network should follow the trajectory
        # of the model, thus this means the input of target
        # should be the input of the model (which might be
        # different from the ground truth inputs during decoding).
        pseudo_batch: BatchType = {
            "source_text": batch["source_text"],
            "source_length": batch["source_length"],
            "source_text_ids": batch["source_text_ids"],
            "target_text": model_output_tokens_extended,
            # `+1` because of the `BOS` token, and `-1`
            # will be used when this value is used.
            "target_length": sequence_lengths + 1,
            "target_text_ids": model_outputs,
        }

    return pseudo_batch


def compute_accuracy_scores(
        As: List[List[str]],
        Bs: List[List[str]]
) -> np.number:
    if len(As) != len(Bs):
        raise ValueError

    accuracies = []
    for a, b in zip(As, Bs):
        if a == b:
            accuracies.append(1.0)
        else:
            accuracies.append(0.0)

    return np.mean(accuracies)


def list_methods_of_an_object(obj: Any) -> List[str]:
    return [
        method_name for method_name in dir(obj)
        if callable(getattr(obj, method_name))]


def preprocess_target_texts(
        tokens_or_list_of_tokens: Union[List[str], List[List[str]]],
        vocab: tx.data.Vocab,
        remove_special_tokens: bool,
) -> Union[List[str], List[List[str]]]:

    processed_tokens_or_list_of_tokens = (
        vocab.map_ids_to_tokens_py(
            vocab.map_tokens_to_ids_py(
                tokens_or_list_of_tokens)))

    if remove_special_tokens is True:
        processed_tokens_or_list_of_tokens = (
            tx.utils.strip_special_tokens(
                processed_tokens_or_list_of_tokens,
                is_token_list=True))

    # (1) If `processed_tokens_or_list_of_tokens` is `np.ndarray[np.str_]`,
    # this will return `List[str]`, as `np.str_.tolist()` returns
    # the same item but in `str` type.
    # (2) If `processed_tokens_or_list_of_tokens` is `np.ndarray[np.ndarray[np.str_]]`,
    # this would return `List[List[str]]`.
    return [x.tolist() for x in processed_tokens_or_list_of_tokens]


def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs))


def map_ids_to_strs_truncated(
        outputs_id: LongTensor,
        sequence_lengths: LongTensor,
        vocab: tx.data.Vocab,
) -> np.ndarray:
    # `map_ids_to_strs` but truncate the sequence
    # before stripping special tokens.
    output_tokens = vocab.map_ids_to_tokens_py(outputs_id)
    output_tokens = [
        tokens[:l]
        for tokens, l in
        zip(output_tokens.tolist(),
            sequence_lengths.tolist())]

    output_texts = tx.data.vocabulary.str_join(output_tokens)
    output_texts = tx.data.vocabulary.strip_special_tokens(output_texts)

    # To match the output type (also Numpy Array)
    return np.array(output_texts)


def dump_dict_to_json_file(file_name: str, data: Dict) -> None:
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
