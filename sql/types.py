import torch
import numpy as np
from typing import Union, List, Dict, Any, TypedDict, NamedTuple


# PyTorch Types
BoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]
FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]

# Numpy Types
# https://numpy.org/doc/stable/reference/arrays.scalars.html

# Huggingface Types
HF_BatchType = Dict[str, Union[torch.Tensor, Any]]


BatchType = TypedDict(
    "BatchType",
    {
        "source_text": List[List[str]],
        "source_text_ids": LongTensor,
        "source_length": LongTensor,
        "target_text": List[List[str]],
        "target_text_ids": LongTensor,
        "target_length": LongTensor,
    }
)

ExperienceType = TypedDict(
    "ExperienceType",
    {
        "target_batch": BatchType,
        "raw_rewards": FloatTensor,
        "shaped_rewards": FloatTensor,
        "sample_lengths": LongTensor,
    }
)


class AttentionRNNDecoderOutput2(NamedTuple):
    logits: FloatTensor
    logits_collections: List[FloatTensor]


class AttentionRNNDecoderOutput3(NamedTuple):
    logits: FloatTensor


# GEM Types
# https://github.com/GEM-benchmark/GEM-metrics/tree/main/test_data
GemDatasetSourceEntryType = TypedDict(
    "GemDatasetSourceEntryType",
    {"source": str}
)

GemDatasetSourceType = TypedDict(
    "GemDatasetSourceType",
    {
        "language": str,
        "values": List[GemDatasetSourceEntryType],
    }
)

GemDatasetTargetEntryType = TypedDict(
    "GemDatasetTargetEntryType",
    {"target": List[str]}
)

GemDatasetTargetType = TypedDict(
    "GemDatasetTargetType",
    {
        "language": str,
        "values": List[GemDatasetTargetEntryType],
    }
)

GemDatasetPredictionEntryType = TypedDict(
    "GemDatasetPredictionEntryType",
    {"generated": str}
)

GemDatasetPredictionType = TypedDict(
    "GemDatasetPredictionType",
    {
        "language": str,
        # https://github.com/GEM-benchmark/GEM-metrics/blob/main/gem_metrics/texts.py#L91
        # "task": str,
        "values": List[GemDatasetPredictionEntryType],
    }
)
