import os
import pandas as pd
import torch
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial
from transformers import RobertaTokenizer
from datasets import load_dataset, Dataset, DatasetDict
import pdb

dataset_dir = "../../data/16-shot/{data_name}/16-{seed}"

def read_dataset(data_dir):
    def create_dataset(lines):
        examples = {}
        examples['sentence'] = [line[0] for line in lines]
        examples['label']    = [line[1] for line in lines]
        dataset = Dataset.from_dict(examples)
        return dataset
    
    train = create_dataset(pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t').values.tolist())
    valid = create_dataset(pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t').values.tolist())
    test  = create_dataset(pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t').values.tolist())
    return DatasetDict({
        'train': train,
        'test': test,
        'validation': valid
    })

def convert_to_features(example_batch, tokenizer, n_prompt_tokens):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    # target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
    target_encodings = tokenizer.convert_tokens_to_ids(example_batch['target_text'])
    target_encodings = [[t] for t in target_encodings]
    
    mask_pos, prompt_mask = [], []
    for input_ids in input_encodings['input_ids']:
        mask_pos.append(input_ids.index(tokenizer.mask_token_id))
        # First Prompt Token ĠX
        prompt_pos = input_ids.index(1577)
        pos = torch.zeros(len(input_ids))
        pos[prompt_pos:prompt_pos+n_prompt_tokens] = 1
        prompt_mask.append(pos.tolist())
    
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'mask_pos': mask_pos,
        'prompt_mask': prompt_mask,
        # 'labels': target_encodings['input_ids'],
        'labels': target_encodings,
    }

    return encodings


def truncatehead(text, tokenizer, length=510):
    tokens = tokenizer.encode(text, truncation=False, padding=False, add_special_tokens=False)
    # Truncate Head
    if len(tokens) > length:
        tokens = tokens[-(length):]
        text = tokenizer.decode(tokens)
    return text

def truncatetail(text, tokenizer, length=510):
    tokens = tokenizer.encode(text, truncation=False, padding=False, add_special_tokens=False)
    # Truncate Head
    if len(tokens) > length:
        tokens = tokens[:(length)]
        text = tokenizer.decode(tokens)
    return text
            
    

class Sentiment2Loader(Loader):
    def __init__(self, data_name=None, tokenizer=None, n_prompt_tokens=50, seed=13):
        super().__init__()
        
        self.data_name = data_name
        self.seed = seed
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Ġterrible",
            1: "Ġgreat",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            
            # [TODO]
            # example['sentence'] = truncatehead(example['sentence'], lengh=510-6-self.n_prompt_tokens)
            # example['sentence'] = truncatetail(example['sentence'])
            # example['input_text'] = '%s . %s . It was %s .' % (prompt, example['sentence'], self.tokenizer.mask_token)
            # example['input_text'] = '%s %s It was %s .' % (example['sentence'], prompt, self.tokenizer.mask_token)
            
            if self.data_name == 'yelp-2':
                example['input_text'] = '%s %s It was %s .' % (example['text'], prompt, self.tokenizer.mask_token)
            else:
                example['input_text'] = '%s %s It was %s .' % (example['sentence'], prompt, self.tokenizer.mask_token)
                                
            example['input_text'] = truncatehead(example['input_text'], self.tokenizer)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s . It was %s .' % (example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        
        # [TODO]
        print(f'==> Read Data /{self.data_name}/16-{self.seed}')
        dataset_folder = os.path.abspath(dataset_dir.format(data_name=self.data_name, seed=self.seed))
        dataset = read_dataset(dataset_folder)[split]
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer, n_prompt_tokens=self.n_prompt_tokens), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "prompt_mask": ins["prompt_mask"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos", "prompt_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    
class Sentiment5Loader(Loader):
    def __init__(self, data_name=None, tokenizer=None, n_prompt_tokens=50, seed=13):
        super().__init__()
        
        self.data_name = data_name
        self.seed = seed
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Ġterrible",
            1: "Ġbad",
            2: "Ġokay",
            3: "Ġgood",
            4: "Ġgreat",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            
            # [TODO]
            # example['sentence'] = truncatehead(example['sentence'], lengh=510-6-self.n_prompt_tokens)
            # example['sentence'] = truncatetail(example['sentence'])
            # example['input_text'] = '%s %s It was %s .' % (example['sentence'], prompt, self.tokenizer.mask_token)
            # example['input_text'] = '%s %s %s .' % (example['sentence'], prompt, self.tokenizer.mask_token)
            
            example['input_text'] = '%s %s It was %s .' % (example['sentence'], prompt, self.tokenizer.mask_token)
            example['input_text'] = truncatehead(example['input_text'], self.tokenizer)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s It was %s .' % (example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        
        # [TODO]
        print(f'==> Read Data /{self.data_name}/16-{self.seed}')
        dataset_folder = os.path.abspath(dataset_dir.format(data_name=self.data_name, seed=self.seed))
        dataset = read_dataset(dataset_folder)[split]
        
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer, n_prompt_tokens=self.n_prompt_tokens), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "prompt_mask": ins["prompt_mask"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos", "prompt_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle

        
class AGNewsLoader(Loader):
    def __init__(self, data_name, tokenizer=None, n_prompt_tokens=50, seed=13):
        super().__init__()
        self.data_name = data_name
        self.seed = seed 
        
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Tech"
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            # example['input_text'] = '%s News: %s %s' % (self.tokenizer.mask_token, prompt,  example['text'])
            example['input_text'] = '%s News: %s %s' % (self.tokenizer.mask_token, prompt,  example['text'])
            example['input_text'] = truncatetail(example['input_text'], self.tokenizer)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s News: %s' % (self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        
        print(f'==> Read Data /{self.data_name}/16-{self.seed}')
        dataset_folder = os.path.abspath(dataset_dir.format(data_name=self.data_name, seed=self.seed))
        dataset = read_dataset(dataset_folder)[split]
        
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer, n_prompt_tokens=self.n_prompt_tokens), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "prompt_mask": ins["prompt_mask"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos", "prompt_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class TRECLoader(Loader):
    def __init__(self, data_name, tokenizer=None, n_prompt_tokens=50, seed=13):
        super().__init__()
        self.data_name = data_name
        self.seed = seed 
        
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "ĠDescription",
            1: "ĠEntity",
            2: "ĠExpression",
            3: "ĠHuman",
            4: "ĠLocation",
            5: "ĠNumber"
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            # example['input_text'] = '%s News: %s %s' % (self.tokenizer.mask_token, prompt,  example['text'])
            example['input_text'] = '%s %s %s' % (self.tokenizer.mask_token, prompt,  example['sentence'])
            # example['input_text'] = '%s: %s %s' % (self.tokenizer.mask_token, prompt,  example['sentence'])
            example['input_text'] = truncatetail(example['input_text'], self.tokenizer)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s: %s' % (self.tokenizer.mask_token, example['sentence'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        print(f'==> Read Data /{self.data_name}/16-{self.seed}')
        dataset_folder = os.path.abspath(dataset_dir.format(data_name=self.data_name, seed=self.seed))
        dataset = read_dataset(dataset_folder)[split]
        
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        
        
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer, n_prompt_tokens=self.n_prompt_tokens), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "prompt_mask": ins["prompt_mask"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos", "prompt_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    
class SubjLoader(Loader):
    def __init__(self, data_name, tokenizer=None, n_prompt_tokens=50, seed=13):
        super().__init__()
        self.data_name = data_name
        self.seed = seed 
        
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Ġsubjective",
            1: "Ġobjective",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            # example['input_text'] = '%s News: %s %s' % (self.tokenizer.mask_token, prompt,  example['text'])
            example['input_text'] = '%s %s %s' % (example['sentence'], prompt, self.tokenizer.mask_token)
            # example['input_text'] = '%s %s This is %s .' % (example['sentence'], prompt, self.tokenizer.mask_token)
            example['input_text'] = truncatehead(example['input_text'], self.tokenizer)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s This is %s .' % (example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        print(f'==> Read Data /{self.data_name}/16-{self.seed}')
        dataset_folder = os.path.abspath(dataset_dir.format(data_name=self.data_name, seed=self.seed))
        dataset = read_dataset(dataset_folder)[split]
        
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer, n_prompt_tokens=self.n_prompt_tokens), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "prompt_mask": ins["prompt_mask"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos", "prompt_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    

class YahooLoader(Loader):
    def __init__(self, data_name, tokenizer=None, n_prompt_tokens=50, seed=13):
        super().__init__()
        self.data_name = data_name
        self.seed = seed 
        
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {0:'culture',1:'science',2:'health',3:'education',4:'computer',5:'sports',6:'business',7:'music',8:'family',9:'politics'}
        
    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            # example['input_text'] = '%s News: %s %s' % (self.tokenizer.mask_token, prompt,  example['text'])
            example['input_text'] = ' %s %s %s' % (prompt, self.tokenizer.mask_token, example['sentence'])
            # example['input_text'] = ' %s Topic %s: %s' % (prompt, self.tokenizer.mask_token, example['sentence'])
            example['input_text'] = truncatetail(example['input_text'], self.tokenizer)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = 'Topic %s: %s' % (self.tokenizer.mask_token, example['sentence'])
            # example['input_text'] = '%s  %s .' % (example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        print(f'==> Read Data /{self.data_name}/16-{self.seed}')
        dataset_folder = os.path.abspath(dataset_dir.format(data_name=self.data_name, seed=self.seed))
        dataset = read_dataset(dataset_folder)[split]
        
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer, n_prompt_tokens=self.n_prompt_tokens), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "prompt_mask": ins["prompt_mask"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos", "prompt_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    
class DBPediaLoader(Loader):
    def __init__(self, data_name, tokenizer=None, n_prompt_tokens=50, seed=13):
        super().__init__()
        self.data_name = data_name
        self.seed = seed 
        
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {0:'ĠCompany',1:'ĠEducation',2:'ĠArtist',3:'ĠSports',4:'ĠOffice',5:'ĠTransportation',6:'ĠBuilding',7:'ĠNatural',
                           8:'ĠVillage',9:'ĠAnimal',10:'ĠPlant',11:'ĠAlbum',12:'ĠFilm',13:'ĠWritten'}

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            # example['input_text'] = '%s News: %s %s' % (self.tokenizer.mask_token, prompt,  example['text'])
            example['input_text'] = ' %s %s %s' % (prompt, self.tokenizer.mask_token, example['sentence'])
            # example['input_text'] = ' %s [Category: %s] %s' % (prompt, self.tokenizer.mask_token, example['content'])
            example['input_text'] = truncatetail(example['input_text'], self.tokenizer)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '[Category: %s] %s' % (self.tokenizer.mask_token, example['sentence'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        print(f'==> Read Data /{self.data_name}/16-{self.seed}')
        dataset_folder = os.path.abspath(dataset_dir.format(data_name=self.data_name, seed=self.seed))
        dataset = read_dataset(dataset_folder)[split]
        
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
            
        
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer, n_prompt_tokens=self.n_prompt_tokens), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "prompt_mask": ins["prompt_mask"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos", "prompt_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    
    
class MRPCLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('glue', 'mrpc', split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class RTELoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "No",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('glue', 'rte', split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle



class SNLILoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "Maybe",
            2: "No",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['premise'], self.tokenizer.mask_token ,example['hypothesis'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['premise'], self.tokenizer.mask_token, example['hypothesis'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('snli', split=split)
        dataset = dataset.filter(lambda example: example['label'] in [0, 1, 2])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    