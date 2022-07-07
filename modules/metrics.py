import json
from typing import List, Dict, Optional, cast
from gem_metrics import (
    References,
    Sources,
    Predictions,
    compute,
    metric_list_to_metric_dict)

from sql.utils import dump_dict_to_json_file
from sql.types import (
    GemDatasetSourceEntryType,
    GemDatasetSourceType,
    GemDatasetTargetEntryType,
    GemDatasetTargetType,
    GemDatasetPredictionEntryType,
    GemDatasetPredictionType)


def compute_GEM_metrics_from_outputs(
        sources: List[str],
        list_of_targets: List[List[str]],
        predictions: List[str],
        base_file_name_to_dump: Optional[str] = None,
        metric_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    if metric_list is None:
        metric_list = [
            # "bertscore",
            "bleu",
            # "bleurt",
            "meteor",
            "rouge",
        ]

    sources_json = GemDatasetSourceType(
        language="en", values=[
            GemDatasetSourceEntryType(source=source)
            for source in sources])
    targets_json = GemDatasetTargetType(
        language="en", values=[
            GemDatasetTargetEntryType(target=targets)
            for targets in list_of_targets])
    predictions_json = GemDatasetPredictionType(
        language="en", values=[
            GemDatasetPredictionEntryType(generated=prediction)
            for prediction in predictions])

    if base_file_name_to_dump is not None:
        dump_dict_to_json_file(
            f"{base_file_name_to_dump}.sources.json",
            cast(Dict, sources_json))
        dump_dict_to_json_file(
            f"{base_file_name_to_dump}.targets.json",
            cast(Dict, targets_json))
        dump_dict_to_json_file(
            f"{base_file_name_to_dump}.predictions.json",
            cast(Dict, predictions_json))

    return compute_GEM_metrics(
        sources_json=sources_json,
        targets_json=targets_json,
        predictions_json=predictions_json,
        metric_list=metric_list)


def compute_GEM_metrics_from_files(
        sources_file_name: str,
        targets_file_name: str,
        predictions_file_name: str,
        metric_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    if metric_list is None:
        metric_list = [
            "bertscore",
            "bleu",
            "bleurt",
            "meteor",
            "rouge",
        ]

    with open(sources_file_name) as f:
        sources_json = json.load(f)
    with open(targets_file_name) as f:
        targets_json = json.load(f)
    with open(predictions_file_name) as f:
        predictions_json = json.load(f)

    return compute_GEM_metrics(
        sources_json=sources_json,
        targets_json=targets_json,
        predictions_json=predictions_json,
        metric_list=metric_list)


def compute_GEM_metrics(
    sources_json: GemDatasetSourceType,
    targets_json: GemDatasetTargetType,
    predictions_json: GemDatasetPredictionType,
    metric_list: List[str],
) -> Dict[str, float]:

    scores_dict = compute(
        outs=Predictions(predictions_json),
        refs=References(targets_json),
        srcs=Sources(sources_json),
        metrics_dict=metric_list_to_metric_dict(metric_list))

    # Cleanup ROUGE-related scores
    for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        if key in scores_dict.keys():
            scores_dict[key] = scores_dict[key]["mid"]["fmeasure"]

    # Cleanup BERTScore
    for key in ["bertscore"]:
        if key in scores_dict.keys():
            scores_dict[key] = scores_dict[key]["f1"]

    for key in ["predictions_file", "references_file"]:
        scores_dict.pop(key)

    return scores_dict
