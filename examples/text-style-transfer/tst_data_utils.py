import os
import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List


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
    assert split in ['train', 'dev', 'test', 'ref']

    if dataset == 'yelp':
        filepath = f'{dataset}/clean/sentiment.{split}.{label}.clean'
        full_filepath = os.path.join(base_path, filepath)
        with open(full_filepath) as f:
            sentences = [line.strip() for line in f]

    elif dataset == 'shakespeare':
        seed_dict = {0: f'100-100', 1: f'100-13', 2: f'100-21'}
        filepath = f'{dataset}/100-shot/{seed_dict[dataset_seed]}/{split}.tsv'
        full_filepath = os.path.join(base_path, filepath)
        df = pd.read_csv(full_filepath, sep='\t')
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


def preprocess_input(text):
    text = re.sub('\s{2,}', ' ', text)
    text = re.sub('(.*?)( )([\.,!?\'])', r'\1\3', text)
    text = re.sub('([a-z])( )(n\'t)', r'\1\3', text)
    text = re.sub('\$ \_', r'$_', text)
    text = re.sub('(\( )(.*?)( \))', r'(\2)', text)
    text = re.sub('(``)( )*(.*?)', r"``\3", text)
    text = re.sub('(.*?)( )*(\'\')', r"\1''", text)
    return text