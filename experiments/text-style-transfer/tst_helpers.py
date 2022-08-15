from dataclasses import dataclass
import numpy as np
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List

from tst_reward import PromptedTextStyleTransferReward


class TextStyleTransferDataset(Dataset):
    def __init__(self, source_texts, target_labels):
        assert len(source_texts) == len(target_labels)
        self.source_texts = source_texts
        self.target_labels = target_labels

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'target_labels': self.target_labels[idx]}
        return item


def make_text_style_transfer_datasets(
        config: "DictConfig") -> Tuple[TextStyleTransferDataset]: 
    assert config.direction in ['0_to_1', '1_to_0']
    label = int(config.direction[0])
    data_dict = {}
    for split in ['train', 'dev', 'test']: 
        if config.dataset == "yelp" and split == 'dev': 
            max_size = 16
        else: 
            max_size = config.max_size

        source_texts, target_labels = load_text_style_transfer_dataset(
            config.dataset, label, split,
            config.dataset_seed, config.base_path, max_size,
            config.max_length, config.max_length_tokenizer)
        tst_dataset = TextStyleTransferDataset(source_texts, target_labels)
        data_dict[split] = tst_dataset

    return data_dict['train'], data_dict['dev'], data_dict['test']


def load_text_style_transfer_dataset(
    dataset: str,
    label: int,
    split: str,
    dataset_seed: Optional[int],
    base_path: str,
    max_size: Optional[int],
    max_length: Optional[int],
    max_length_tokenizer: Optional[str]
) -> Tuple[List[str]]:
    assert dataset in ['yelp', 'shakespeare']
    assert label in [0, 1]
    assert split in ['train', 'dev', 'test']

    if dataset == 'yelp':
        filepath = f'{dataset}/preprocessed/sentiment.{split}.{label}.preprocess'
        full_filepath = os.path.join(base_path, filepath)
        with open(full_filepath) as f:
            sentences = [line.strip() for line in f]

    elif dataset == 'shakespeare':
        seed_dict = {0: f'100-100', 1: f'100-13', 2: f'100-21'}
        filepath = f'{dataset}/100-shot/{seed_dict[seed]}/{split}.tsv'
        df = pd.read_csv(filepath, sep='\t')
        sentences = df.query(f'label == {label}').text.tolist()

    # Option to keep only certain number of examples
    if max_size is not None:
        sentences = sentences[:max_size]

    # Keep training sentences that are shorter than max_length
    # to keep running time reasonable
    if max_length is not None:
        tokenizer = AutoTokenizer.load_from_pretrained(max_length_tokenizer)
        new_sentences = [sent for sent in sentences
                         if len(tokenizer(sent)['input_ids']) < max_length]

        # Resample sentences back to the original number of examples
        rng = np.random.default_rng(dataset_seed)
        sentences = (new_sentences
                     + list(rng.choice(new_sentences,
                                       size=(len(sentences)
                                             - len(new_sentences)),
                                       replace=False)))

    source_texts = sentences
    if label == 0: target_labels = ['LABEL_1' for _ in source_texts]
    elif label == 1: target_labels = ['LABEL_0' for _ in source_texts]

    return source_texts, target_labels


@dataclass
class TextStyleTransferDatasetConfig:
    dataset: str = "???"
    dataset_seed: Optional[int] = None
    direction: str = "???" 
    base_path: str = './data'
    max_size: Optional[int] = None
    max_length: Optional[int] = None
    max_length_tokenizer: Optional[str] = None


def make_prompted_text_style_transfer_reward(
        config: "DictConfig") -> PromptedTextStyleTransferReward:
    return PromptedTextStyleTransferReward(
        config.task_lm, config.style_classifier_path, 
        config.num_repeats, config.num_samples, config.num_bootstraps,
        config.compute_zscore, config.lower_outputs)


@dataclass
class PromptedTextStyleTransferRewardConfig:
    task_lm: str = 'distilgpt2'
    style_classifier_path: str = '???'
    num_repeats: int = 4
    num_samples: int = 32
    num_bootstraps: int = 4
    compute_zscore: bool = True  # Whether to compute z-score of rewards
    lower_outputs: bool = False  # Whether to convert all outputs to lower case
