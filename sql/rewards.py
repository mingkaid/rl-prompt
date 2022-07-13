from math import pi
import os
import click
import torch
import numpy as np
import pandas as pd
import sacrebleu as scb
from collections import Counter
from collections import defaultdict
from typing import List, Tuple, Union, Dict, Optional, Callable, Any, cast
from bert_score import BERTScorer
from transformers import (
    pipeline,
    TextClassificationPipeline,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    RobertaForSequenceClassification,
    AutoModelForMaskedLM)

from modules import gpt2 as gpt2_modules
from sql.types import FloatTensor
from sql import utils as sql_utils
from sql import misc_utils

import math
from math import cos, pi
import random
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_Xs_Ys_sizes(
        Xs: List,
        Ys: List,
        check_type_is_list: bool = False,
        check_first_element_is_string: bool = True,
) -> None:
    if len(Xs) != len(Ys):
        raise ValueError(
            f"Xs.length = {len(Xs)}, "
            f"Ys.length = {len(Ys)}")

    if check_type_is_list is True:
        if not isinstance(Xs, list) or not isinstance(Ys, list):
            raise ValueError(
                f"Xs.type = {type(Xs)}, "
                f"Ys.type = {type(Ys)}")

    if check_first_element_is_string is True:
        if not isinstance(Xs[0], str) or not isinstance(Ys[0], str):
            raise ValueError(
                f"Xs[0].type = {type(Xs[0])}, "
                f"Ys[0].type = {type(Ys[0])}")

    
class PromptedTextStyleTransferReward(object):
    END_PUNCT = '"'
    def __init__(
            self,
            # Prompt Reward Parameters
            prompt_task_lm: str = 'distilgpt2',
            prompt_dataset: Optional[str] = None,
            prompt_dataset_seed: Optional[int] = None,
            prompt_dataset_basepath: str = '.',
            # TST-specific parameters
            tst_clf_basepath: Optional[str] = None,
            tst_n_repeats: int = 4,
            tst_num_samples: int = 32, # Num of samples from which to take the output
            tst_num_bootstraps: int = 4, # Num of bootstraps to reduce reward randomness
            **kwargs
    ) -> None:

        # https://huggingface.co/gpt2
        generator_model = prompt_task_lm
        print('Task LM:', generator_model)
        generator_device = 0 # TODO
        reward_device = 0 # TODO
        tokenizer = AutoTokenizer.from_pretrained(generator_model, pad_token='<|endoftext|>')
        self._generator = pipeline(
            "text-generation",
            model=generator_model,
            tokenizer=tokenizer,
            device=generator_device)
        self.num_samples = tst_num_samples
        self.num_bootstraps = tst_num_bootstraps
        
        self.dataset = prompt_dataset
        self.seed = prompt_dataset_seed
        if prompt_dataset_basepath == '.': # TODO
            self.basepath = os.path.abspath(os.path.join('.', os.pardir, os.pardir, os.pardir))
        else:
            self.basepath = prompt_dataset_basepath
        print(self.basepath)

        if tst_clf_basepath is not None: 
            self.tst_clf_basepath = tst_clf_basepath
        else: 
            self.tst_clf_basepath = os.path.join(self.basepath, 'experiments/tst_classifiers/')
        dataset_clf_tokenizers = {'yelp': 'bert-base-uncased',
                                  'shakespeare': 'bert-base-uncased'}
        dataset_clf_paths = {'yelp': self.tst_clf_basepath + "/yelp-bert-base-uncased-train",
                             'shakespeare': self.tst_clf_basepath + f"/shakespeare-bert-base-uncased-train-100-{self.seed}"}
        
        self._classifier = pipeline(
            "sentiment-analysis",
            model=dataset_clf_paths[self.dataset],
            tokenizer=dataset_clf_tokenizers[self.dataset],
            device=reward_device)

        self._tst_templates = ['{prompt} "{sentence_1}" "']
        self._counter = 0
        self.tokens_explored = set()
        self.n_repeats = tst_n_repeats
        self.lower = ('lower' in self.dataset) or (self.dataset == 'yelp')
        # self.ctc_scorer = StyleTransferScorer(align='E-roberta')
        self._bert_scorer = BERTScorer('roberta-large', device=reward_device, rescale_with_baseline=True, lang='en')
        
        self._tst_inputs = self._load_tst_inputs()
        self._tst_inputs_idx = {('train', 'LABEL_0'): 0, 
                                ('train', 'LABEL_1'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0}
        self.temperature = 1.0
        print(f'Model Input Max Length = {self._generator.model.config.max_length}')
        
    def _load_tst_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        tst_inputs = {}
        # tokenizer = self._generator.tokenizer
        if self.dataset == 'yelp':
            filepath_train_0 = os.path.join(self.basepath, "prompt_tasks/text-style-transfer/yelp/preprocessed/sentiment.train.0.preprocess")
            filepath_train_1 = os.path.join(self.basepath, "prompt_tasks/text-style-transfer/yelp/preprocessed/sentiment.train.1.preprocess")
            filepath_dev_0 = os.path.join(self.basepath, "prompt_tasks/text-style-transfer/yelp/preprocessed/sentiment.dev.0.preprocess")
            filepath_dev_1 = os.path.join(self.basepath, "prompt_tasks/text-style-transfer/yelp/preprocessed/sentiment.dev.1.preprocess")
            filepath_test_ref_0 = os.path.join(self.basepath, "prompt_tasks/text-style-transfer/yelp/preprocessed/sentiment.test_ref.0.preprocess")
            filepath_test_ref_1 = os.path.join(self.basepath, "prompt_tasks/text-style-transfer/yelp/preprocessed/sentiment.test_ref.1.preprocess")
            
            with open(filepath_train_0) as f: 
                sentences_train_0 = [line.strip() for line in f]
            with open(filepath_train_1) as f: 
                sentences_train_1 = [line.strip() for line in f]

            with open(filepath_dev_0) as f: 
                sentences_dev_0 = [line.strip() for line in f]
            with open(filepath_dev_1) as f: 
                sentences_dev_1 = [line.strip() for line in f]

            with open(filepath_test_ref_0) as f: 
                sentences_test_ref_0 = [line.strip() for line in f]
            with open(filepath_test_ref_1) as f: 
                sentences_test_ref_1 = [line.strip() for line in f]
            test_size = 16
                
        elif self.dataset in ['shakespeare']:
            seed_dic = {0:f'100-100', 1:f'100-13', 2:f'100-21'}
            
            filepath_train = os.path.join(self.basepath, 
                                          'prompt_tasks/text-style-transfer/',
                                          f'{self.dataset}/100-shot/{seed_dic[self.seed]}/train.tsv')
            filepath_dev = os.path.join(self.basepath, 
                                          'prompt_tasks/text-style-transfer/',
                                          f'{self.dataset}/100-shot/{seed_dic[self.seed]}/dev.tsv')
            filepath_test = os.path.join(self.basepath, 
                                          'prompt_tasks/text-style-transfer/',
                                          f'{self.dataset}/100-shot/{seed_dic[self.seed]}/test.tsv')
            
            df_train = pd.read_csv(filepath_train, sep='\t')
            df_dev = pd.read_csv(filepath_dev, sep='\t')
            df_test = pd.read_csv(filepath_test, sep='\t')
            
            sentences_train_0 = df_train.query('label == 0').text.tolist()
            print(sentences_train_0[:5])
            sentences_train_1 = df_train.query('label == 1').text.tolist()
            print(sentences_train_1[:5])
            sentences_dev_0 = df_dev.query('label == 0').text.tolist()
            sentences_dev_1 = df_dev.query('label == 1').text.tolist()
            sentences_test_0 = df_test.query('label == 0').text.tolist()
            sentences_test_1 = df_test.query('label == 1').text.tolist()
            test_size = 100
            
            # Keep training sentences that are shorter than 25 tokens to keep running time reasonable
            if self.dataset in ['shakespeare']:
                max_length = 25
                new_train_0 = [sent for sent in sentences_train_0 \
                               if len(self._generator.tokenizer(sent)['input_ids']) < max_length]
                new_train_1 = [sent for sent in sentences_train_1 \
                               if len(self._generator.tokenizer(sent)['input_ids']) < max_length]
                new_dev_0 = [sent for sent in sentences_dev_0 \
                             if len(self._generator.tokenizer(sent)['input_ids']) < max_length]
                new_dev_1 = [sent for sent in sentences_dev_1 \
                             if len(self._generator.tokenizer(sent)['input_ids']) < max_length]
                
                # Resample sentences back to 100 examples
                rng = np.random.default_rng(2022)
                sentences_train_0 = new_train_0 + list(rng.choice(new_train_0, 
                                                                  size=(len(sentences_train_0) - len(new_train_0)), 
                                                                  replace=False))
                sentences_train_1 = new_train_1 + list(rng.choice(new_train_1, 
                                                                  size=(len(sentences_train_1) - len(new_train_1)), 
                                                                  replace=False))
                sentences_dev_0 = new_dev_0 + list(rng.choice(new_dev_0, 
                                                                  size=(len(sentences_dev_0) - len(new_dev_0)), 
                                                                  replace=False))
                sentences_dev_1 = new_dev_1 + list(rng.choice(new_dev_1, 
                                                                  size=(len(sentences_dev_1) - len(new_dev_1)), 
                                                                  replace=False))
                
            
        idx = 0 # Start index
        size = 100
        
        size = len(sentences_train_1)
        tst_inputs[('train', 'LABEL_0')] = sentences_train_1[idx:(idx+size)]
        print(idx, size, tst_inputs[('train', 'LABEL_0')][:5])
        tst_inputs[('train', 'LABEL_0')] = list(itertools.chain(*[[s for _ in range(self.n_repeats)] \
                                                                   for s in tst_inputs[('train', 'LABEL_0')]]))
        
        size = len(sentences_train_0)
        tst_inputs[('train', 'LABEL_1')] = sentences_train_0[idx:(idx+size)]
        print(idx, size, tst_inputs[('train', 'LABEL_1')][:5])
        tst_inputs[('train', 'LABEL_1')] = list(itertools.chain(*[[s for _ in range(self.n_repeats)] \
                                                                   for s in tst_inputs[('train', 'LABEL_1')]]))
        
        tst_inputs[('infer', 'LABEL_0')] = sentences_dev_1[idx:(idx+test_size)]
        tst_inputs[('infer', 'LABEL_1')] = sentences_dev_0[idx:(idx+test_size)]
        print(idx, test_size, tst_inputs[('infer', 'LABEL_0')][:5])
        
        return tst_inputs

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        # return tokens
        return [self._generator.tokenizer
                .convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, source_strings: List[str], prompt_strings: List[str]) -> List[str]:
        template = self._tst_templates[0]
        
        return [
            template.format(sentence_1=s_1, prompt=p) for s_1, p
            in zip(source_strings, prompt_strings)]

    def _get_inputs(self, mode: str, target_labels: List[str]): 
        import random
        inputs = []
        indices = []
        
        for i, label in enumerate(target_labels): 
            idx = self._tst_inputs_idx[(mode, label)]
            data = self._tst_inputs[(mode, label)]
            
            if mode == 'train': 
                inputs.append(data[idx])
                indices.append(int(idx // self.n_repeats))
            else:
                inputs.append(data[idx])
                indices.append(int(idx))
            
            idx += 1
            idx %= len(data)
            
            # Reshuffle training data after each iteration
            if idx == 0 and mode == 'train': 
                data_idx = np.arange(len(data) // self.n_repeats) * self.n_repeats
                random.Random(0).shuffle(data_idx)
                self._tst_inputs[(mode, label)] = \
                    (list(itertools.chain(*[[data[i] for _ in range(self.n_repeats)]
                                            for i in data_idx])))
            
            self._tst_inputs_idx[(mode, label)] = idx
        
        return indices, inputs


    def postprocess_output(self, text, end_punct='"', start_punct=None): 
        try: 
            end = text.index(end_punct)
        except ValueError: 
            end = len(text)
        text = text[:end].strip()
        
        if start_punct is not None: 
            start = text.find(start_punct)
            while start >= 0: 
                text = text[start+1:].strip()
                start = text.find(start_punct)
    
        try: 
            end = text.index('.')
        except ValueError: 
            end = len(text)

        try: 
            end = min(end, text.index('!'))
        except ValueError: 
            end = end

        try: 
            end = min(end, text.index('?'))
        except ValueError: 
            end = end
            
        text = text[:end+1].strip()
        if self.lower: text = text.lower()

        return text


    def forward(self, 
                target_labels: List[str], 
                prompts: List[str],
                to_tensor: bool, 
                mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError
        assert all([label in ['LABEL_0', 'LABEL_1'] for label in target_labels])

        self.tokens_explored = self.tokens_explored.union(*[set(p.split()) for p in prompts])
        source_indices, source_strings = self._get_inputs(mode, target_labels)
        prompt_strings = self._convert_tokens_to_string(prompts)
        formatted_prompts = self._format_prompts(source_strings, prompt_strings)

        if mode == 'train': 
            self._counter += 1

        from torch.utils.data import Dataset
        class MyDataset(Dataset):
            def __init__(self, x):
                self.samples = x

            def __getitem__(self,index):
                return self.samples[index]

            def __len__(self):
                return len(self.samples)

        n_reward = self.num_samples
        k_reward = self.num_bootstraps
        N = n_reward * k_reward
        X = MyDataset(formatted_prompts)
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            X,
            # max_length=60,
            # max_new_tokens=input_length,
            pad_token_id=50256,
            top_k=10,
            temperature=self.temperature,
            num_return_sequences=N,
            # Only return generated text, without the prompt
            return_full_text=False)
            

        rewards: List[FloatTensor] = []
        input_rewards: Dict(str, List(float)) = defaultdict(list)
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index, out in enumerate(generator_outputs):
            generated_texts = []
            for output in out: 
                text = output["generated_text"]
                generated_texts.append(self.postprocess_output(text))
            
            reference_texts = [source_strings[batch_index] for g in generated_texts]
            check_Xs_Ys_sizes(generated_texts, reference_texts)
            
            compute_bertscore = True
            if compute_bertscore:
                bertscore_f1 = self._bert_scorer.score(generated_texts, 
                                                        reference_texts)[2]
                bertscore_rewards = [max(b, 0) for b in (bertscore_f1 * 100).tolist()]
                bertscore = torch.tensor(bertscore_rewards).float().mean()
                quantities_to_log['bertscore'].append(bertscore)
            
            compute_recon = True
            if compute_recon:
                recon_rewards = bertscore_rewards
                recon = torch.tensor(recon_rewards).float().mean()
                quantities_to_log["recon"].append(recon)
            
            compute_sentiment = True
            if compute_sentiment:
                X_output = MyDataset(generated_texts)
                label = target_labels[batch_index]

                probs = []
                correct = []
                for c in self._classifier(X_output, batch_size=32, truncation=True): 
                    prob = ((c['label'] == label) * c['score'] + \
                            (c['label'] != label) * (1 - c['score']))
                    probs.append(prob)
                    correct.append(c['label'] == label)

                acc = torch.tensor(correct).float().mean()
                quantities_to_log['acc'].append(acc)
                style = torch.tensor(probs).float().mean()
                quantities_to_log['style'].append(style)
            
            compute_sum_reward = True
            if compute_sum_reward:
                recon_weight = 1
                style_weight = 1
                sum_rewards = [(recon_weight * r + style_weight * 100 * p) / (recon_weight + style_weight) \
                                for r, c, p in zip(recon_rewards, correct, probs)]
                
                # Monte Carlo k_reward times and average
                mc_avg = True
                if mc_avg:
                    l = len(sum_rewards)
                    k = k_reward
                    segmented_sum_rewards = [sum_rewards[i*l//k:(i+1)*l//k] for i in range(k)]

                    mean_sum_reward = torch.tensor(segmented_sum_rewards).float().mean()
                    values, indices = torch.tensor(segmented_sum_rewards).float().max(axis=1)
                    max_sum_reward = values.mean()

                    list_values = [segmented_sum_rewards[i][index] for i, index in enumerate(indices)]
                    input_rewards[reference_texts[0]] += list_values
                    max_reward_value = max(list_values)
                    
                top_index = sum_rewards.index(max_reward_value)
                reward = max_sum_reward
                
                quantities_to_log["sum_reward"].append(max_sum_reward)
                mean_reward = torch.tensor(sum_rewards).float().mean()
                quantities_to_log["mean_reward"].append(mean_reward)
                top_recon = torch.tensor(recon_rewards[top_index]).float()
                quantities_to_log["top_recon"].append(top_recon)
                # top_acc = torch.tensor(correct[top_index]).float()
                # quantities_to_log["top_acc"].append(top_acc)
                top_style = torch.tensor(probs[top_index]).float()
                quantities_to_log["top_style"].append(top_style)
                
                print(self._counter, '|',
                        prompts[batch_index], '|', 
                        formatted_prompts[batch_index], '|', 
                        generated_texts[top_index], '|', 
                        'BERTScore:', round(bertscore_rewards[top_index], 2), '|',
                        'Style:', round(probs[top_index], 2), '|',
                        'Sum Reward:', round(sum_rewards[top_index], 2), '|',
                        'Reward:', round(reward.item(), 2))
                        
            else:
                reward = bertscore

            quantities_to_log["num_tokens_explored"].append(torch.tensor(len(self.tokens_explored)).float())
            
            rewards.append(reward)

        rewards_tensor = torch.stack(rewards)
    
        batch_zscore = True
        if batch_zscore:
            input_max_reward_means = {k: np.mean(v) for k, v in input_rewards.items()}
            input_max_reward_stds = {k: np.std(v) for k, v in input_rewards.items()}
            idx_means = torch.tensor([input_max_reward_means[s] for s in source_strings]).float()
            idx_stds = torch.tensor([input_max_reward_stds[s] for s in source_strings]).float()
            print(idx_means)
            print(idx_stds)
            
            rewards_tensor = (rewards_tensor - idx_means) / (idx_stds + 1e-4)

        quantities_to_log["reward_var"].append(torch.var(torch.tensor(rewards)))
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            target_labels=sources,
            prompts=predictions,
            to_tensor=to_tensor,
            mode=mode)
    

class PromptedClassificationReward(object):

    def __init__(
            self,
            # Prompt Reward Parameters
            prompt_task_lm: str = 'roberta-large',
            prompt_dataset: Optional[str] = None,
            prompt_dataset_seed: Optional[int] = None,
            prompt_dataset_basepath: str = '.',
            # Classification-specific parameters
            clf_kshot: int = 16,
            **kwargs
    ) -> None:

        self.device = device
        self.dataset = prompt_dataset
        self.dataset_seed = prompt_dataset_seed
        if prompt_dataset_basepath == '.': # TODO
            self.dataset_basepath = os.path.abspath(os.path.join('.', os.pardir, os.pardir, os.pardir))
        else:
            self.dataset_basepath = prompt_dataset_basepath
        self.kshot = clf_kshot
        self.task_lm = prompt_task_lm
        print('Task LM:', self.task_lm)
        if 'gpt' in self.task_lm: # left-to-right LM
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm, pad_token='<|endoftext|>')
            self._generator = GPT2LMHeadModel.from_pretrained(self.task_lm).to(self.device)
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id
        elif 'bert' in self.task_lm: # Masked LM
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
            self._generator = AutoModelForMaskedLM.from_pretrained(self.task_lm).to(self.device)
        
        self._task_name = self.dataset
        if self._task_name in ['sst-2', 'yelp-2', 'mr', 'cr']:
            self.num_classes = 2
        elif self._task_name == 'agnews':
            self.num_classes = 4
        elif self._task_name in ['sst-5', 'yelp-5']:
            self.num_classes = 5

        self._templates = self.load_templates() # prompt templates
        self._inputs = self._load_inputs()
            
        self._inputs_idx = {(idx, 'LABEL_' + str(i)): 0 \
                            for i in range(self.num_classes) \
                            for idx in ['train', 'infer']}

        self._counter = 0
        if self._task_name in ['sst-2', 'yelp-2', 'mr', 'cr']:
            self.verbalizers = ['\u0120terrible', '\u0120great'] # num_classes
        elif self._task_name == 'agnews': 
            self.verbalizers = ['World', 'Sports', 'Business', 'Tech'] # num_classes
        elif self._task_name in ['sst-5', 'yelp-5']:
            self.verbalizers = ['\u0120terrible', '\u0120bad', '\u0120okay', '\u0120good', '\u0120great'] # num_classes
        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(verbalizer) for verbalizer in self.verbalizers]

    def load_templates(self) -> List[str]:
        if 'bert' not in self.task_lm:
            # Template for left-to-right LMs like GPT-2
            temp_template = "{sentence_1} {prompt}" 
        else:
            if self._task_name in ['sst-2', 'yelp-2', 'mr', 'cr', 'sst-5', 'yelp-5']:
                temp_template = "{sentence_1} {prompt} <mask> ."            
            elif self._task_name in ['agnews']: 
                temp_template = "<mask> {prompt} {sentence_1}"

        return [temp_template]

    def _load_few_shot_examples(self):
        base_path = os.path.join(self.dataset_basepath, f'prompt_tasks/few-shot-classification/{str(self.kshot)}-shot/')
        seed_dic = {0:'16-100', 1:'16-13', 2:'16-21', 3:'16-42', 4:'16-87'} # TODO
        
        dataset_path = os.path.join(base_path, self.dataset, seed_dic[self.dataset_seed])
        train_tsv = os.path.join(dataset_path, 'train.tsv')
        dev_tsv = os.path.join(dataset_path, 'dev.tsv')
        # our loaded data cache
        train_dicts = [{} for _ in range(self.num_classes)] # num_classes
        dev_dicts = [{} for _ in range(self.num_classes)]

        # loading train examples
        with open(train_tsv, 'r') as f:
            train = f.readlines()
        for i, line in enumerate(train):
            if i == 0: # skip first line
                continue
            line = line.strip('\n')
            label = int(line.strip('\n').split('\t')[1])
            text = line.split('\t')[0] # single sent.
            train_dicts[label][text] = label 

        # loading dev examples
        with open(dev_tsv, 'r') as f:
            dev = f.readlines()
        for i, line in enumerate(dev):
            if i == 0:
                continue
            line = line.strip('\n')
            label = int(line.strip('\n').split('\t')[1])
            text = line.split('\t')[0] 
            dev_dicts[label][text] = label 
                    
        return train_dicts, dev_dicts

    def _load_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        inputs = {}
        assert self.dataset_seed in [0, 1, 2, 3, 4]
        train_dicts, dev_dicts = self._load_few_shot_examples() # examples dict (sentence - string, label - int)

        for class_idx in range(self.num_classes):
            inputs[('train', 'LABEL_'+str(class_idx))] = list(train_dicts[class_idx].keys())
            inputs[('infer', 'LABEL_'+str(class_idx))] = list(dev_dicts[class_idx].keys())
        inputs['train'] = [(text, class_idx) for text in list(train_dicts[class_idx].keys()) for class_idx in range(self.num_classes)]
        inputs['infer'] = [(text, class_idx) for text in list(dev_dicts[class_idx].keys()) for class_idx in range(self.num_classes)]
        random.Random(0).shuffle(inputs['train'])
        random.Random(0).shuffle(inputs['infer'])
        return inputs # full train/val data

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._tokenizer.convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, source_strings: List[str], prompt_strings: List[str]) -> List[str]:
        template = self._templates[0]
        return [
            template.format(sentence_1=s, prompt=p) for s_1, p
            in zip(source_strings, prompt_strings) for s in s_1
        ], 0 # TODO

    def _get_few_shot_inputs(self, mode: str, target_labels: List[str], kshot: int): 
        # per batch (based on _load_inputs)
        inputs = []
        for i, label in enumerate(target_labels): # input sent. per batch
            idx = self._inputs_idx[(mode, label)] # placeholder
            inputs.append([self._inputs[(mode, 'LABEL_' + str(j))][idx] for j in range(self.num_classes)])
                
            idx += 1
            idx %= len(self._inputs[(mode, 'LABEL_0')])
            if idx == 0:
                for k in range(self.num_classes):
                    random.Random(0).shuffle(self._inputs[(mode, 'LABEL_' + str(k))])
            self._inputs_idx[(mode, label)] = idx 
        return inputs

    def _get_probs(self, texts, prompt_length=0, locations = 0):
        # for MLM, add mask token
        encoded_input  = self._tokenizer(texts, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=True)
        batch_size = len(texts)
        self._task_name = self.dataset
        if self._task_name in ['sst-2', 'yelp-2', 'mr', 'cr', 'sst-5', 'yelp-5']:
            seq_len = torch.ne(encoded_input.input_ids, self._tokenizer.pad_token_id).sum(-1) - 1
            if 'bert' in self.task_lm:
                seq_len = seq_len - 2
        elif self._task_name == 'agnews':
            seq_len = torch.tensor([1 for _ in range(batch_size)])
        with torch.no_grad(): # pay attention to special token \u0120
            if 'bert' in self.task_lm: # MLM, <s>, <\s>
                logits = self._generator(
                    input_ids=encoded_input.input_ids.to(device), 
                    attention_mask=encoded_input.attention_mask.to(device)
                    ).logits 
            elif 'gpt' in self.task_lm: # CLM, next token
                logits = self._generator(
                    input_ids=encoded_input.input_ids.to(device), 
                    ).logits
            else:
                logits = self._generator(
                    input_ids=encoded_input.input_ids.to(device), 
                    attention_mask=encoded_input.attention_mask.to(device),
                    decoder_input_ids=None
                    ).logits 
            logits = logits[range(batch_size), seq_len] 

        return logits 
    
    def forward(self, target_labels: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError
        source_strings = self._get_few_shot_inputs(mode, target_labels, self.kshot)
        # adding the prompt
        prompt_strings = self._convert_tokens_to_string(prompts)
        prompt_length = len(prompts[0].split())
        formatted_prompts, locations = self._format_prompts(source_strings, prompt_strings)
        probs = self._get_probs(formatted_prompts, prompt_length, locations)
        
        rewards: List[FloatTensor] = []
        input_rewards: Dict(str, List(float)) = defaultdict(list)
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        # few-shot combined reward index
        if mode == 'train': 
            batch_reward_index = [[i * self.num_classes + j for i in range(len(prompts))] for j in range(0, self.num_classes)]
    
        for batch_index in range(len(prompts)): # 1-shot: z-score always work, not needs to change code bone
            if mode == 'train':
                current_pt = [prompt_strings[batch_index] for _ in range(len(source_strings))]
                formatted_current_prompt_sentences, _ = self._format_prompts(source_strings, current_pt)
                current_probs = self._get_probs(formatted_current_prompt_sentences, prompt_length)
                # few-shot class-specific probs.
                # 1. few-shot class-specific probs vector.
                class_probs = [torch.softmax(current_probs[batch_reward_index[i]][:, self.verbalizer_ids], -1) \
                                        for i in range(self.num_classes)]
                # 2. few-shot class-specific probs.
                class_prob = [class_probs[i][:, i] for i in range(self.num_classes)]
                # current predicted highest prob. 
                pred_best_prob = [torch.max(class_probs[i], dim=-1).values for i in range(self.num_classes)]
                # current predicted label.
                pred_label = [torch.argmax(class_probs[i], dim=-1) for i in range(self.num_classes)]
                # whether predicted correctly
                flag_index = [torch.where(pred_label[i] == i, True, False) for i in range(self.num_classes)]
                # accuracy over few-shot examples, e.g. 16-shot
                acc = sum([torch.sum(flag_index[i]) for i in range(self.num_classes)])
                acc = acc/(len(flag_index[0]) * self.num_classes)
                # second predicted prob.     
                pred_sec_prob = [torch.topk(class_probs[i], k=2, dim=-1).values[:,1] for i in range(self.num_classes)]       
                # reward 
                rewards_ = [torch.where(class_prob[i] - pred_best_prob[i] >= 0, 2 * (pred_best_prob[i] - pred_sec_prob[i]), 1.8*(class_prob[i] - pred_best_prob[i])) for i in range(self.num_classes)] 
                # visualization, the constant might only be used for visualization
                if self._task_name == 'agnews':
                    average_reward = torch.mean(rewards_[0] + rewards_[1] + rewards_[2] + rewards_[3]) * 25
                    quantities_to_log["avg_world_reward"].append(torch.mean(rewards_[0]).item())
                    quantities_to_log["avg_sports_reward"].append(torch.mean(rewards_[1]).item())
                    quantities_to_log["avg_business_reward"].append(torch.mean(rewards_[2]).item())
                    quantities_to_log["avg_tech_reward"].append(torch.mean(rewards_[3]).item())
                elif self._task_name in ['sst-5', 'yelp-5']:
                    average_reward = torch.mean(rewards_[0] + rewards_[1] + rewards_[2] + rewards_[3] + rewards_[4]) * 20
                    quantities_to_log["avg_1_reward"].append(torch.mean(rewards_[0]).item())
                    quantities_to_log["avg_2_reward"].append(torch.mean(rewards_[1]).item())
                    quantities_to_log["avg_3_reward"].append(torch.mean(rewards_[2]).item())
                    quantities_to_log["avg_4_reward"].append(torch.mean(rewards_[3]).item())
                    quantities_to_log["avg_5_reward"].append(torch.mean(rewards_[4]).item())
                elif self._task_name in ['sst-2', 'yelp-2', 'mr', 'cr']:
                    average_reward = torch.mean(rewards_[0] + rewards_[1]) * 50
                    quantities_to_log["avg_neg_reward"].append(torch.mean(rewards_[0]).item())
                    quantities_to_log["avg_pos_reward"].append(torch.mean(rewards_[1]).item())

                quantities_to_log["avg_reward"].append(average_reward.item())

            elif mode == 'infer':
                infer_probs = [torch.softmax(probs[batch_index * self.num_classes + i, self.verbalizer_ids], -1) for i in range(self.num_classes)]
                # prob
                infer_prob = [infer_probs[i] for i in range(self.num_classes)]
                # predicted label
                pred_labels = [torch.argmax(infer_probs[i], dim=-1) for i in range(self.num_classes)]
                # TODO

            # z-score normalization (1st stage)
            self._counter += 1
            if mode == 'train':
                input_rewards['z'] += [average_reward.item()]
                # visualize one of them
                if self._task_name in ['sst-2', 'yelp-2', 'mr', 'cr']:
                    print(prompts[batch_index], '\n', 
                    formatted_prompts[batch_index * 2], '\n', 
                    formatted_prompts[batch_index * 2 + 1], '\n', 
                    'Accuracy:', acc.item(), '|',
                    'Reward:', round(average_reward.item(), 2)) 
                elif self._task_name == 'agnews':
                    print(prompts[batch_index], '\n',
                    formatted_prompts[batch_index * 4], '\n',
                    formatted_prompts[batch_index * 4 + 1], '\n',
                    formatted_prompts[batch_index * 4 + 2], '\n',
                    formatted_prompts[batch_index * 4 + 3], '\n',
                    'Accuracy:', acc.item(), '|'
                    'Reward:', round(average_reward.item(), 2))
                elif self._task_name in ['sst-5', 'yelp-5']:
                    print(prompts[batch_index], '\n',
                    formatted_prompts[batch_index * 5], '\n',
                    formatted_prompts[batch_index * 5 + 1], '\n',
                    formatted_prompts[batch_index * 5 + 2], '\n',
                    formatted_prompts[batch_index * 5 + 3], '\n',
                    formatted_prompts[batch_index * 5 + 4], '\n',
                    'Accuracy:', acc.item(), '|',
                    'Reward:', round(average_reward.item(), 2))
                rewards.append(average_reward)

            if mode == 'infer': # val score
                if self._task_name in ['sst-2', 'yelp-2', 'mr', 'cr']: # TODO
                    pos_acc = torch.tensor(1) if pred_labels[1] == 1 else torch.tensor(0)
                    neg_acc = torch.tensor(1) if pred_labels[0] == 0 else torch.tensor(0)
                    reward = torch.tensor((pos_acc+neg_acc)/2).float()
                elif self._task_name == 'agnews':
                    world_acc = torch.tensor(1) if pred_labels[0] == 0 else torch.tensor(0)
                    sports_acc = torch.tensor(1) if pred_labels[1] == 1 else torch.tensor(0)
                    business_acc = torch.tensor(1) if pred_labels[2] == 2 else torch.tensor(0)
                    tech_acc = torch.tensor(1) if pred_labels[3] == 3 else torch.tensor(0)         
                    reward = torch.tensor((world_acc+sports_acc+business_acc+tech_acc)/4).float() # To be visualized     
                elif self._task_name in ['sst-5', 'yelp-5']:
                    acc_1 = torch.tensor(1) if pred_labels[0] == 0 else torch.tensor(0)
                    acc_2 = torch.tensor(1) if pred_labels[1] == 1 else torch.tensor(0)
                    acc_3 = torch.tensor(1) if pred_labels[2] == 2 else torch.tensor(0)
                    acc_4 = torch.tensor(1) if pred_labels[3] == 3 else torch.tensor(0)
                    acc_5 = torch.tensor(1) if pred_labels[4] == 4 else torch.tensor(0)
                    reward = torch.tensor((acc_1+acc_2+acc_3+acc_4+acc_5)/5).float() # To be visualized
                
                rewards.append(reward)
                quantities_to_log["reward"].append(reward.item()) 
        rewards_tensor = torch.stack(rewards)
        
        # z-score normalization (2nd stage)
        if mode=='train':
            batch_zscore = True
            if batch_zscore:
                input_reward_means = {k:np.mean(v) for k,v in input_rewards.items()}
                input_reward_stds = {k:np.std(v) for k,v in input_rewards.items()}
                # not source strings
                idx_means = torch.tensor(input_reward_means['z']).float()
                idx_stds = torch.tensor(input_reward_stds['z']).float()
                rewards_tensor = (rewards_tensor - idx_means)/(idx_stds + 1e-4)

            for i in range(rewards_tensor.size(0)):
                quantities_to_log['resized_reward'].append(rewards_tensor[i].item())
        
        rewards_log = dict(
            (reward_key, torch.mean(torch.tensor(reward_vals)))
            for reward_key, reward_vals in quantities_to_log.items())

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:

        return self.forward(
            target_labels=sources,
            prompts=predictions,
            to_tensor=to_tensor,
            mode=mode)

    
reward_name_to_cls_map = {
    "prompted-text-style-transfer": PromptedTextStyleTransferReward,
    'prompted-classification': PromptedClassificationReward,
}

