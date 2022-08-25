from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List

from fsc_reward import PromptedClassificationReward


class PromptedClassificationDataset(Dataset):
    def __init__(
        self, 
        source_texts: List[str], 
        class_labels: List[str]
    ):
        assert len(source_texts) == len(class_labels)
        self.source_texts = source_texts
        self.class_labels = class_labels

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'class_labels': self.class_labels[idx]}
        return item


def make_few_shot_classification_dataset(
        config: "DictConfig") -> Tuple[PromptedClassificationDataset]: 
    data_dict = {}
    for split in ['train', 'dev', 'test']: 
        source_texts, class_labels, num_classes, verbalizers, template = \
            load_few_shot_classification_dataset(config.dataset, 
                                                 config.dataset_seed, 
                                                 split, config.base_path, 
                                                 config.num_shots)
        fsc_dataset = PromptedClassificationDataset(source_texts, 
                                                    class_labels)
        data_dict[split] = fsc_dataset

    return (data_dict['train'], data_dict['dev'], data_dict['test'],
            num_classes, verbalizers, template)


def load_few_shot_classification_dataset(
    dataset: str,
    dataset_seed: Optional[int],
    split: str,
    base_path: str,
    num_shots: int
) -> Tuple[List[str]]:
    assert dataset in ['agnews', 'cr', 'mr', 'sst-2', 
                       'sst-5', 'yelp-2', 'yelp-5']
    assert split in ['train', 'dev', 'test']
    assert num_shots in [16]

    seed_dict = {0:'16-100', 1:'16-13', 2:'16-21', 3:'16-42', 4:'16-87'}
    seed_path = seed_dict[dataset_seed]
    filepath = f'{num_shots}-shot/{dataset}/{seed_path}/{split}.tsv'
    full_filepath = os.path.join(base_path, filepath)
    df = pd.read_csv(full_filepath, sep='\t')
    if 'text' in df:
        source_texts = df.text.tolist()
    else: 
        source_texts = df.sentence.tolist()
    class_labels = df.label.tolist()

    verbalizers = get_dataset_verbalizers(dataset)
    num_classes = len(verbalizers)

    template = None
    if dataset == 'agnews': 
        template = "<mask> {prompt} {sentence_1}"

    return (source_texts, class_labels, 
            num_classes, verbalizers, template)


def get_dataset_verbalizers(dataset: str) -> List[str]: 
    if dataset in ['sst-2', 'yelp-2', 'mr', 'cr']:
        verbalizers = ['\u0120terrible', '\u0120great'] # num_classes
    elif dataset == 'agnews': 
        verbalizers = ['World', 'Sports', 'Business', 'Tech'] # num_classes
    elif dataset in ['sst-5', 'yelp-5']:
        verbalizers = ['\u0120terrible', '\u0120bad', '\u0120okay', 
                       '\u0120good', '\u0120great'] # num_classes
    elif dataset == 'subj':
        verbalizers = ['\u0120subjective', '\u0120objective']
    elif dataset == 'trec':
        verbalizers = ['\u0120Description', '\u0120Entity',
                    '\u0120Expression', '\u0120Human',
                    '\u0120Location', '\u0120Number']
    elif dataset == 'yahoo':
        verbalizers = ['culture', 'science',
                    'health', 'education',
                    'computer', 'sports',
                    'business', 'music',
                    'family', 'politics']
    elif dataset == 'dbpedia':
        verbalizers = ['\u0120Company', '\u0120Education',
                    '\u0120Artist', '\u0120Sports',
                    '\u0120Office', '\u0120Transportation',
                    '\u0120Building', '\u0120Natural',
                    '\u0120Village', '\u0120Animal',
                    '\u0120Plant', '\u0120Album',
                    '\u0120Film', '\u0120Written']
    return verbalizers


@dataclass
class FewShotClassificationDatasetConfig:
    dataset: str = "???"
    dataset_seed: Optional[int] = None 
    base_path: str = './data'
    num_shots: int = 16


def make_prompted_classification_reward(
    num_classes: int,
    verbalizers: List[str],
    template: Optional[str],  
    config: "DictConfig") -> PromptedClassificationReward:
    return PromptedClassificationReward(config.task_lm, config.is_mask_lm, 
                                        config.compute_zscore, 
                                        config.incorrect_coeff, 
                                        config.correct_coeff,
                                        num_classes, verbalizers, template)


@dataclass
class PromptedClassificationRewardConfig:
    task_lm: str = 'distilroberta-base'
    is_mask_lm: Optional[bool] = None
    compute_zscore: bool = True
    incorrect_coeff: float = 180.0
    correct_coeff: float = 200.0
