from math import pi
import os
import click
import torch
import numpy as np
import pandas as pd
import sacrebleu as scb
from collections import Counter
from collections import defaultdict
from joblib import Parallel, delayed
# from fairseq.data.data_utils import collate_tokens
# from fairseq.models.roberta import RobertaHubInterface
from typing import List, Tuple, Union, Dict, Optional, Callable, Any, cast
from ctc_score import StyleTransferScorer
from bert_score import BERTScorer
from transformers import (LogitsProcessor, 
                          LogitsProcessorList, 
                          TextClassificationPipeline,
                          AutoModelForSequenceClassification, 
                          AutoTokenizer)


from datasets import load_metric
from transformers import (
    pipeline,
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
# from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

from modules import gpt2 as gpt2_modules
from sql.types import FloatTensor
from sql import utils as sql_utils
from sql import misc_utils

import math
from math import cos, pi
import random
import itertools

try:
    from detoxify import Detoxify
except ModuleNotFoundError:
    Detoxify = None

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
            prompt_dataset_basepath: str = '/data/mingkai/prompt-generation/dirty-code/rl-prompt',
            # TST-specific parameters
            tst_clf_basepath: str = '/data/mingkai/prompt-generation/soft-Q-learning-for-text-generation/experiments/yelp_sentiment_classifier',
            tst_n_repeats: Optional[int] = 4,
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
        dataset_clf_tokenizers = {'yelp': 'bert-base-uncased',
                                  'shakespeare': 'bert-base-uncased',}
        dataset_clf_paths = {'yelp': tst_clf_basepath + "/results-bert-base/checkpoint-10410",
                             'shakespeare': tst_clf_basepath + f"/shakespeare-bert-base-uncased-train-100-{self.seed}"}
        
        self.lower = ('lower' in self.dataset) or (self.dataset == 'yelp')
        self._classifier = pipeline(
            "sentiment-analysis",
            model=dataset_clf_paths[self.dataset],
            tokenizer=dataset_clf_tokenizers[self.dataset],
            device=reward_device)

        self._tst_templates = ['{prompt} "{sentence_1}" "']
        self._counter = 0
        self.tokens_explored = set()
        self.n_repeats = tst_n_repeats
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
            filepath_train_0 = os.path.join(self.basepath, "data/yelp-gpt2-control-only/raw-prep/sentiment.train.0.preprocess")
            filepath_train_1 = os.path.join(self.basepath, "data/yelp-gpt2-control-only/raw-prep/sentiment.train.1.preprocess")
            filepath_dev_0 = os.path.join(self.basepath, "data/yelp-gpt2-control-only/raw-prep/sentiment.dev.0.preprocess")
            filepath_dev_1 = os.path.join(self.basepath, "data/yelp-gpt2-control-only/raw-prep/sentiment.dev.1.preprocess")
            filepath_test_ref_0 = os.path.join(self.basepath, "data/yelp-gpt2-control-only/raw-prep/sentiment.test_ref.0.preprocess")
            filepath_test_ref_1 = os.path.join(self.basepath, "data/yelp-gpt2-control-only/raw-prep/sentiment.test_ref.1.preprocess")
            
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
            seed_dic = {0:f'100-100', 1:f'100-13', 2:f'100-21', 3:f'100-42', 4:f'100-87'}
            filepath_train = os.path.join(self.basepath, f'clf-tasks/100-shot/{self.dataset}/{seed_dic[self.seed]}/train.tsv')
            filepath_dev = os.path.join(self.basepath, f'clf-tasks/100-shot/{self.dataset}/{seed_dic[self.seed]}/dev.tsv')
            filepath_test = os.path.join(self.basepath, f'clf-tasks/100-shot/{self.dataset}/{seed_dic[self.seed]}/test.tsv')
            
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
            prompt_dataset_basepath: str = '/data/mingkai/prompt-generation/dirty-code/rl-prompt',
            # Classification-specific parameters
            clf_kshot: int = 16,
            clf_num_classes: int = 2,
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
        self.num_classes = clf_num_classes
        self.task_lm = prompt_task_lm
        print('Task LM:', self.task_lm)
        if 'gpt' in self.task_lm: # left-to-right LM
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm, pad_token='<|endoftext|>')
            self._generator = GPT2LMHeadModel.from_pretrained(self.task_lm).to(self.device)
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id
        elif 'bert' in self.task_lm: # MLM
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
            self._generator = AutoModelForMaskedLM.from_pretrained(self.task_lm).to(self.device)

        self._templates = self.load_templates()
        self._inputs = self._load_inputs()
        task_name = self.dataset
        if task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
            self._inputs_idx = {('train', 'LABEL_0'): 0, 
                                ('train', 'LABEL_1'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0}
        elif task_name == 'agnews':
            self._inputs_idx = {('train', 'LABEL_0'): 0,
                                ('train', 'LABEL_1'): 0,
                                ('train', 'LABEL_2'): 0,
                                ('train', 'LABEL_3'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0,
                                ('infer', 'LABEL_2'): 0,
                                ('infer', 'LABEL_3'): 0
                                }
        elif task_name in ['sst-5', 'yelp-5']:
            self._inputs_idx = {('train', 'LABEL_0'): 0, 
                                ('train', 'LABEL_1'): 0,
                                ('train', 'LABEL_2'): 0,
                                ('train', 'LABEL_3'): 0,
                                ('train', 'LABEL_4'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0,
                                ('infer', 'LABEL_2'): 0,
                                ('infer', 'LABEL_3'): 0,
                                ('infer', 'LABEL_4'): 0,
                                }

        self._counter = 0
        if task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
            self.pos_verbalizer_candidate = ['\u0120positive', '\u0120great'   ,'\u0120good', '\u0120wonderful', '\u0120delicious', '\u0120dog', '\u0120cat', '\u0120terrible', '\u0120yes']
            self.neg_verbalizer_candidate = ['\u0120negative', '\u0120terrible','\u0120bad' , '\u0120bad', '\u0120bad', '\u0120cat', '\u0120dog', '\u0120great', '\u0120no']
            self.pos_verbalizer_candidate_nospace = ['postive', 'great', 'good', 'wonderful']
            self.neg_verbalizer_candidate_nospace = ['negative', 'terrible', 'bad', 'bad']
            self.pos_verbalizer = self.pos_verbalizer_candidate[1]
            self.neg_verbalizer = self.neg_verbalizer_candidate[1]
        
            self.pos_id = self._tokenizer.convert_tokens_to_ids(self.pos_verbalizer)
            self.neg_id = self._tokenizer.convert_tokens_to_ids(self.neg_verbalizer)
        elif task_name == 'agnews': # news topic classification dataset (news articles)
            self.world_verbalizer = ['\u0120Global', 'world', 'World']
            self.sports_verbalizer = ['\u0120Athletics', 'sport', 'Sports']
            self.business_verbalizer = ['\u0120Finance', 'business', 'Business']
            self.technology_verbalizer = ['\u0120Technology', 'technology', 'Tech']
            
            self.world_id = self._tokenizer.convert_tokens_to_ids(self.world_verbalizer[0])
            self.sports_id = self._tokenizer.convert_tokens_to_ids(self.sports_verbalizer[0])
            self.business_id = self._tokenizer.convert_tokens_to_ids(self.business_verbalizer[0])
            self.technology_id = self._tokenizer.convert_tokens_to_ids(self.technology_verbalizer[0])
        elif task_name in ['sst-5', 'yelp-5']:
            self.verbalizer_1 = '\u0120terrible'
            self.verbalizer_2 = '\u0120bad'
            self.verbalizer_3 = '\u0120okay'
            self.verbalizer_4 = '\u0120good'
            self.verbalizer_5 = '\u0120great'

            self.id_1 = self._tokenizer.convert_tokens_to_ids(self.verbalizer_1)
            self.id_2 = self._tokenizer.convert_tokens_to_ids(self.verbalizer_2)
            self.id_3 = self._tokenizer.convert_tokens_to_ids(self.verbalizer_3)
            self.id_4 = self._tokenizer.convert_tokens_to_ids(self.verbalizer_4)
            self.id_5 = self._tokenizer.convert_tokens_to_ids(self.verbalizer_5)

    def load_templates(self) -> List[str]:
        self._task_name = self.dataset
        if 'bert' not in self.task_lm:
            # Template for left-to-right LMs like GPT-2
            temp_template = "{sentence_1} {prompt}" # TODO
        else:
            if self._task_name in ['SST-2', 'yelp-2', 'mr', 'cr', 'sst-5', 'yelp-5']:
                temp_template = "{sentence_1} {prompt} <mask> ."            
            elif self._task_name in ['agnews']: # TODO
                temp_template = "<mask> {prompt} {sentence_1}"

        return [temp_template]
    
    def _load_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        inputs = {}
        
        assert self.dataset_seed in [0, 1, 2, 3, 4]
        assert self.kshot in [16]
        task_name = self.dataset
        if task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
            train_pos_dict, train_neg_dict, dev_pos_dict, dev_neg_dict = self._load_few_shot_examples()
        elif task_name == 'agnews':
            (train_world_dict, train_sports_dict, train_business_dict, train_technology_dict, 
            dev_world_dict, dev_sports_dict, dev_business_dict, dev_technology_dict) = self._load_few_shot_examples()
        elif task_name in ['sst-5', 'yelp-5']:
            (train_1_dict, train_2_dict, train_3_dict, train_4_dict, train_5_dict, 
            dev_1_dict, dev_2_dict, dev_3_dict, dev_4_dict, dev_5_dict) = self._load_few_shot_examples()
        
        if task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
            inputs[('train', 'LABEL_0')] = list(train_pos_dict.keys())#sentences_train_1[:kshot]
            inputs[('train', 'LABEL_1')] = list(train_neg_dict.keys())#sentences_train_0[:kshot]
            inputs[('infer', 'LABEL_0')] = list(dev_pos_dict.keys())#sentences_train_1[-kshot:]
            inputs[('infer', 'LABEL_1')] = list(dev_neg_dict.keys())#sentences_train_0[-kshot:]
            inputs['train'] = [(text,1) for text in list(train_pos_dict.keys())] + [(text, 0) for text in list(train_neg_dict.keys())]
            inputs['infer'] = [(text,1) for text in list(dev_pos_dict.keys())] + [(text, 0) for text in list(dev_neg_dict.keys())]
            random.Random(0).shuffle(inputs['train'])
            random.Random(0).shuffle(inputs['infer'])
        elif task_name == 'agnews': # news topic classification
            inputs[('train', 'LABEL_0')] = list(train_world_dict.keys())#sentences_train_1[:kshot]
            inputs[('train', 'LABEL_1')] = list(train_sports_dict.keys())#sentences_train_0[:kshot]
            inputs[('train', 'LABEL_2')] = list(train_business_dict.keys())#sentences_train_1[:kshot]
            inputs[('train', 'LABEL_3')] = list(train_technology_dict.keys())#sentences_train_0[:kshot]
            
            inputs[('infer', 'LABEL_0')] = list(dev_world_dict.keys())#sentences_train_1[:kshot]
            inputs[('infer', 'LABEL_1')] = list(dev_sports_dict.keys())#sentences_train_0[:kshot]
            inputs[('infer', 'LABEL_2')] = list(dev_business_dict.keys())#sentences_train_1[:kshot]
            inputs[('infer', 'LABEL_3')] = list(dev_technology_dict.keys())

            inputs['train'] = [(text, 0) for text in list(train_world_dict.keys())] + [(text, 1) for text in list(train_sports_dict.keys())] + [(text, 2) for text in list(train_business_dict.keys())] + [(text, 3) for text in list(train_technology_dict.keys())]
            inputs['infer'] = [(text, 0) for text in list(dev_world_dict.keys())] + [(text, 1) for text in list(dev_sports_dict.keys())] + [(text, 2) for text in list(dev_business_dict.keys())] + [(text, 3) for text in list(dev_technology_dict.keys())]
            random.Random(0).shuffle(inputs['train'])
            random.Random(0).shuffle(inputs['infer'])
        elif task_name in ['sst-5', 'yelp-5']:
            inputs[('infer', 'LABEL_0')] = list(dev_1_dict.keys())#sentences_train_1[:kshot]
            inputs[('infer', 'LABEL_1')] = list(dev_2_dict.keys())#sentences_train_0[:kshot]
            inputs[('infer', 'LABEL_2')] = list(dev_3_dict.keys())#sentences_train_1[:kshot]
            inputs[('infer', 'LABEL_3')] = list(dev_4_dict.keys())
            inputs[('infer', 'LABEL_4')] = list(dev_5_dict.keys())

            inputs[('train', 'LABEL_0')] = list(train_1_dict.keys())#sentences_train_1[:kshot]
            inputs[('train', 'LABEL_1')] = list(train_2_dict.keys())#sentences_train_0[:kshot]
            inputs[('train', 'LABEL_2')] = list(train_3_dict.keys())#sentences_train_1[:kshot]
            inputs[('train', 'LABEL_3')] = list(train_4_dict.keys())
            inputs[('train', 'LABEL_4')] = list(train_5_dict.keys())

            inputs['train'] = [(text, 0) for text in list(train_1_dict.keys())] + [(text, 1) for text in list(train_2_dict.keys())] + [(text, 2) for text in list(train_3_dict.keys())] + [(text, 3) for text in list(train_4_dict.keys())] + [(text, 4) for text in list(train_5_dict.keys())]
            inputs['infer'] = [(text, 0) for text in list(dev_1_dict.keys())] + [(text, 1) for text in list(dev_2_dict.keys())] + [(text, 2) for text in list(dev_3_dict.keys())] + [(text, 3) for text in list(dev_4_dict.keys())] + [(text, 4) for text in list(dev_5_dict.keys())]
            random.Random(0).shuffle(inputs['train'])
            random.Random(0).shuffle(inputs['infer'])
            
        
        return inputs

    def _load_few_shot_examples(self):
        base_path = os.path.join(self.dataset_basepath, f'tasks/{str(self.kshot)}-shot/')
        seed_dic = {0:'16-100', 1:'16-13', 2:'16-21', 3:'16-42', 4:'16-87'}
        
        dataset_path = os.path.join(base_path, self.dataset, seed_dic[self.dataset_seed])
        train_tsv = os.path.join(dataset_path, 'train.tsv')
        dev_tsv = os.path.join(dataset_path, 'dev.tsv')
        task_name = self.dataset
        if task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
            # read train, binary sentiment at first
            train_pos_dict = {}
            train_neg_dict = {}
            # read dev, SST at first
            dev_pos_dict = {}
            dev_neg_dict = {}
        elif task_name == 'agnews':
            train_world_dict = {}
            train_sports_dict = {}
            train_business_dict = {}
            train_technology_dict = {}

            dev_world_dict = {}
            dev_sports_dict = {}
            dev_business_dict = {}
            dev_technology_dict = {}
        elif task_name in ['sst-5', 'yelp-5']:
            train_5_dict = {}
            train_4_dict = {}
            train_3_dict = {}
            train_2_dict = {}
            train_1_dict = {}
            
            dev_5_dict = {}
            dev_4_dict = {}
            dev_3_dict = {}
            dev_2_dict = {}
            dev_1_dict = {}

        # loading train examples
        with open(train_tsv, 'r') as f:
            train = f.readlines()
        for i,line in enumerate(train):
            if i==0:
                continue
            line = line.strip('\n')
            
            label = int(line.strip('\n').split('\t')[1])
            text = line.split('\t')[0] # single sentence
            if task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
                if label == 0:
                    train_neg_dict[text] = label
                elif label == 1:
                    train_pos_dict[text] = label
            elif task_name == 'agnews':
                if label == 0:
                    train_world_dict[text] = label
                elif label == 1:
                    train_sports_dict[text] = label
                elif label == 2:
                    train_business_dict[text] = label
                elif label == 3:
                    train_technology_dict[text] = label
            elif task_name in ['sst-5', 'yelp-5']:
                if label == 0:
                    train_1_dict[text] = label
                elif label == 1:
                    train_2_dict[text] = label
                elif label == 2:
                    train_3_dict[text] = label
                elif label == 3:
                    train_4_dict[text] = label
                elif label == 4:
                    train_5_dict[text] = label

        # loading dev examples
        with open(dev_tsv, 'r') as f:
            dev = f.readlines()
        for i,line in enumerate(dev):
            if i==0:
                continue
            line = line.strip('\n')
            
            label = int(line.strip('\n').split('\t')[1])
            text = line.split('\t')[0] 

            if task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
                if label == 0:
                    dev_neg_dict[text] = label
                elif label == 1:
                    dev_pos_dict[text] = label
            elif task_name == 'agnews':
                if label == 0:
                    dev_world_dict[text] = label
                elif label == 1:
                    dev_sports_dict[text] = label
                elif label == 2:
                    dev_business_dict[text] = label
                elif label == 3:
                    dev_technology_dict[text] = label
            elif task_name in ['sst-5', 'yelp-5']:
                if label == 0:
                    dev_1_dict[text] = label
                elif label == 1:
                    dev_2_dict[text] = label
                elif label == 2:
                    dev_3_dict[text] = label
                elif label == 3:
                    dev_4_dict[text] = label
                elif label == 4:
                    dev_5_dict[text] = label
                    
        if task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
            return train_pos_dict, train_neg_dict, dev_pos_dict, dev_neg_dict # dict: "text": label
        elif task_name == 'agnews':
            return train_world_dict, train_sports_dict, train_business_dict, train_technology_dict, dev_world_dict, dev_sports_dict, dev_business_dict, dev_technology_dict
        elif task_name in ['sst-5', 'yelp-5']:
            return train_1_dict, train_2_dict, train_3_dict, train_4_dict, train_5_dict, dev_1_dict, dev_2_dict, dev_3_dict, dev_4_dict, dev_5_dict

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._tokenizer.convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, source_strings: List[str], prompt_strings: List[str]) -> List[str]:
        template = self._templates[0]
        return [
            template.format(sentence_1=s, prompt=p) for s_1, p
            in zip(source_strings, prompt_strings) for s in s_1
        ], 0

    def _get_few_shot_inputs(self, mode: str, target_labels: List[str], kshot: int): 
        # per batch
        inputs = []
        self._task_name = self.dataset
        if self._task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
            p_data = self._inputs[(mode, 'LABEL_0')]
            n_data = self._inputs[(mode, 'LABEL_1')]
        elif self._task_name == 'agnews':
            world_data = self._inputs[(mode, 'LABEL_0')]
            sports_data = self._inputs[(mode, 'LABEL_1')]
            business_data = self._inputs[(mode, 'LABEL_2')]
            technology_data = self._inputs[(mode, 'LABEL_3')]
        elif self._task_name in ['sst-5', 'yelp-5']:
            data_1 = self._inputs[(mode, 'LABEL_0')]
            data_2 = self._inputs[(mode, 'LABEL_1')]
            data_3 = self._inputs[(mode, 'LABEL_2')]
            data_4 = self._inputs[(mode, 'LABEL_3')]
            data_5 = self._inputs[(mode, 'LABEL_4')]

        for i, label in enumerate(target_labels): # current input per batch
            idx = self._inputs_idx[(mode, label)] 
            if self._task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
                if mode == 'train':
                    inputs.append([p_data[idx], n_data[idx]])
                elif mode == 'infer':
                    inputs.append([p_data[idx], n_data[idx]])
            elif self._task_name == 'agnews':
                if mode == 'train':
                    inputs.append([world_data[idx], sports_data[idx], business_data[idx], technology_data[idx]])
                elif mode == 'infer':
                    inputs.append([world_data[idx], sports_data[idx], business_data[idx], technology_data[idx]])
            elif self._task_name in ['sst-5', 'yelp-5']:
                inputs.append([data_1[idx], data_2[idx], data_3[idx], data_4[idx], data_5[idx]])
                
            idx += 1
            if self._task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
                idx %= len(p_data)
            elif self._task_name == 'agnews':
                idx %= len(world_data)
            elif self._task_name in ['sst-5', 'yelp-5']:
                idx %= len(data_1)

            if idx == 0:
                random.Random(0).shuffle(self._inputs[(mode, 'LABEL_0')])
                random.Random(0).shuffle(self._inputs[(mode, 'LABEL_1')])
                if self._task_name == 'agnews':
                    random.Random(0).shuffle(self._inputs[(mode, 'LABEL_2')])
                    random.Random(0).shuffle(self._inputs[(mode, 'LABEL_3')])
                elif self._task_name in ['sst-5', 'yelp-5']:
                    random.Random(0).shuffle(self._inputs[(mode, 'LABEL_2')])
                    random.Random(0).shuffle(self._inputs[(mode, 'LABEL_3')])
                    random.Random(0).shuffle(self._inputs[(mode, 'LABEL_4')])

            self._inputs_idx[(mode, label)] = idx
        
        return inputs

    def _get_probs(self, texts, prompt_length=0, locations = 0):
        # for MLM, add mask token
        encoded_input  = self._tokenizer(texts, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=True)
        batch_size = len(texts)
        self._task_name = self.dataset
        if self._task_name in ['SST-2', 'yelp-2', 'mr', 'cr', 'sst-5', 'yelp-5']:
            seq_len = torch.ne(encoded_input.input_ids, self._tokenizer.pad_token_id).sum(-1) - 1
            if 'bert' in self.task_lm:
                seq_len = seq_len - 2
        elif self._task_name == 'agnews':
            seq_len = torch.tensor([1 for _ in range(batch_size)])
            '''
            seq_len = torch.ne(encoded_input.input_ids, self._tokenizer.pad_token_id).sum(-1) - 1
            if 'bert' in self.LM_type.lower():
                seq_len = seq_len - 2
            '''
        with torch.no_grad():
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
        if self._task_name == 'agnews' and mode == 'train': # batch size
            world_index = [i * 4 for i in range(len(prompts))]
            sports_index = [i + 1 for i in world_index]
            business_index = [i + 2 for i in world_index]
            technology_index = [i + 3 for i in world_index]
        elif self._task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
            pos_index = [i * 2 for i in range(len(prompts))]
            neg_index = [i + 1 for i in pos_index]
        elif self._task_name in ['yelp-5', 'sst-5']:
            #print(formatted_prompts)
            first_index = [i * 5 for i in range(len(prompts))]
            second_index = [i + 1 for i in first_index]
            third_index = [i + 2 for i in first_index]
            fourth_index = [i + 3 for i in first_index]
            fifth_index = [i + 4 for i in first_index]


        for batch_index in range(len(prompts)): # 1-shot: z-score always work, not needs to change code bone
            # new
            if self._task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
                Combine=True
                if not Combine or mode=='infer':
                    pos_logits = probs[batch_index * 2, [self.pos_id, self.neg_id]]
                    neg_logits = probs[batch_index * 2 + 1, [self.pos_id, self.neg_id]]
                    pos_probs = torch.softmax(pos_logits, -1)
                    neg_probs = torch.softmax(neg_logits, -1)
                    # UnBounded
                    # pos_reward = ((pos_pos_prob - pos_neg_prob) / (pos_neg_prob))
                    # neg_reward = ((neg_neg_prob - neg_pos_prob) / (neg_pos_prob))
        
                    # new
                    pos_pos_prob, pos_neg_prob = pos_probs[0], pos_probs[1]
                    neg_pos_prob, neg_neg_prob = neg_probs[0], neg_probs[1]
                    #print(pos_pos_prob,neg_neg_prob)
            
                    # Bounded
                    pos_reward = torch.exp(pos_pos_prob)#-torch.exp(1-pos_pos_prob) #/ (pos_neg_prob + pos_pos_prob))
                    neg_reward = torch.exp(neg_neg_prob)#-torch.log(1-neg_neg_prob) #/ (neg_pos_prob + neg_neg_prob))
        
                    # balanced reward
                    pos_ratio = pos_pos_prob/pos_neg_prob
                    neg_ratio = neg_neg_prob/neg_pos_prob

                    # bias exists, like our intermediate region in TST
                    # 7/3 * 7/3 , 0, 
                    if pos_pos_prob >= 0.5 and neg_neg_prob >= 0.5:# w/o shaping, [2+]
                        acc = torch.tensor(1).float().cuda()
                        if pos_pos_prob >=0.6 and neg_neg_prob >=0.6:
                            reward = 2*torch.log(pos_ratio*neg_ratio) # 5
                        else:
                            reward = 2*torch.log(pos_ratio*neg_ratio)# / (torch.exp(torch.tensor(2,dtype=torch.float32))) 
                    elif (pos_pos_prob-0.5) * (neg_neg_prob-0.5)<0: # 
                    #print(pos_pos_prob, neg_neg_prob)
                        acc = torch.tensor(0.5).cuda()
        
                        if (pos_pos_prob - 0.5) < 0:
                            reward = 2*torch.clip(torch.log(pos_ratio),min=-5)
                        elif (neg_neg_prob-0.5) < 0:
                            reward = 2*torch.clip(torch.log(neg_ratio), min=-5)
                    elif pos_pos_prob < 0.5 and neg_neg_prob < 0.5: # [-24,-11]
                        acc = torch.tensor(0.0).cuda()
                        reward = 2*torch.clip(torch.log(pos_ratio),min=-5) + 2*torch.clip(torch.log(neg_ratio), min=-5)#8*torch.clip(torch.log(pos_pos_prob*neg_neg_prob),min=-3) # clip:  #torch.tensor(-5).to('cuda')
                    average_reward = reward
                else:
                    current_pt = [prompt_strings[batch_index] for _ in range(len(source_strings))]
                    # formatted_current_prompt_sentences, _ = self._format_prompts(source_strings, current_pt, PAIR)
                    formatted_current_prompt_sentences, _ = self._format_prompts(source_strings, current_pt)
                    current_probs = self._get_probs(formatted_current_prompt_sentences, prompt_length)
                    sample_Pair=False
                    if sample_Pair:
                        sample_pos_index = random.sample(pos_index, 1)
                        sample_neg_index = random.sample(neg_index, 1)
                    pos_logits = current_probs[pos_index][:,[self.pos_id, self.neg_id]]
                    neg_logits = current_probs[neg_index][:,[self.pos_id, self.neg_id]]
                    pos_probs = torch.softmax(pos_logits,-1)
                    neg_probs = torch.softmax(neg_logits,-1)
                    print(neg_probs.shape)
                    pos_prob = pos_probs[:,0]
                    neg_prob = neg_probs[:,1]
                    
                    #print(pos_prob[0], neg_prob[0])
                    #print(pos_prob[1], neg_prob[1])
                    # current prediced probs
                    pos_predict_best_prob = torch.max(pos_probs, dim=-1).values
                    neg_predict_best_prob = torch.max(neg_probs, dim=-1).values
                    # 
                    pos_predict_sec_prob = torch.topk(pos_probs, k=2,dim=-1).values[:,1]
                    neg_predict_sec_prob = torch.topk(neg_probs, k=2,dim=-1).values[:,1]
                    #
                    pos_flag = (pos_prob - pos_predict_best_prob >=0)
                    neg_flag = (neg_prob - neg_predict_best_prob >=0)

                    class_flag = pos_flag.int() + neg_flag.int()

                    acc = torch.sum(class_flag.float())
                    acc = acc/(2*pos_flag.shape[0])
                    
                    reward_pos = torch.where(pos_prob - pos_predict_best_prob >=0, 2*(pos_predict_best_prob-pos_predict_sec_prob), 1.8*(pos_prob-pos_predict_best_prob))
                    reward_neg = torch.where(neg_prob - neg_predict_best_prob >=0, 2*(neg_predict_best_prob-neg_predict_sec_prob), 1.8*(neg_prob-neg_predict_best_prob))
                    average_reward = torch.mean(reward_pos + reward_neg) * 50

                if mode == 'train':
                    quantities_to_log["acc"].append(acc.item())
                    quantities_to_log["reward"].append(average_reward.item())
                else:
                    quantities_to_log["reward"].append(reward.item())
            
            elif self._task_name == 'agnews':
                if mode == 'train':
                    sep=False
                    if not sep:
                        # learned prompts
                        # current_pt -> prob
                        current_pt = [prompt_strings[batch_index] for _ in range(len(source_strings))]
                        # formatted_current_prompt_sentences, _ = self._format_prompts(source_strings, current_pt, PAIR)
                        formatted_current_prompt_sentences, _ = self._format_prompts(source_strings, current_pt)
                        current_probs = self._get_probs(formatted_current_prompt_sentences, prompt_length)
                        # perform sampling
                        '''
                        sample_world_index = random.sample(world_index, 1)
                        sample_sports_index = random.sample(sports_index, 1)
                        sample_business_index = random.sample(business_index, 1)
                        sample_technology_index = random.sample(technology_index, 1)
                        '''
                        average_reward = torch.tensor(0).float().cuda()
                        world_logits = current_probs[world_index][:, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                        sports_logits = current_probs[sports_index][:, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                        business_logits = current_probs[business_index][:, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                        technology_logits = current_probs[technology_index][:, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                    else:
                        world_logits = probs[batch_index * 4, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                        sports_logits = probs[batch_index * 4 + 1, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                        business_logits = probs[batch_index * 4 + 2, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                        technology_logits = probs[batch_index * 4 + 3, [self.world_id, self.sports_id, self.business_id, self.technology_id]]

                    world_probs = torch.softmax(world_logits, -1)
                    sports_probs = torch.softmax(sports_logits, -1)
                    tech_probs = torch.softmax(technology_logits, -1)
                    business_probs = torch.softmax(business_logits, -1)
                    if not sep:
                        world_prob = world_probs[:,0] # :,
                        sports_prob = sports_probs[:,1]
                        tech_prob = tech_probs[:,3]
                        business_prob = business_probs[:,2]
                    else:
                        world_prob = world_probs[0]
                        sports_prob = sports_probs[1]
                        tech_prob = tech_probs[3]
                        business_prob = business_probs[2]

                    # current predicted probs
                    world_predict_best_prob = torch.max(world_probs, dim=-1).values
                    sports_predict_best_prob = torch.max(sports_probs, dim=-1).values
                    tech_predict_best_prob = torch.max(tech_probs, dim=-1).values
                    business_predict_best_prob = torch.max(business_probs, dim=-1).values
                    # current predicted labels
                    world_predict_label = torch.argmax(world_probs, dim=-1)
                    sports_predict_label = torch.argmax(sports_probs, dim=-1)
                    tech_predict_label = torch.argmax(tech_probs, dim=-1)
                    business_predict_label = torch.argmax(business_probs, dim=-1)
                    # whether predicted correctly
                    wd_index = torch.where(world_predict_label == 0, True, False)
                    sp_index = torch.where(sports_predict_label == 1, True, False) 
                    tc_index = torch.where(tech_predict_label == 3, True, False)
                    bs_index = torch.where(business_predict_label == 2, True, False)
                    # acc
                    wd_index_val = wd_index * torch.ones(wd_index.shape).cuda()
                    sp_index_val = sp_index * torch.ones(wd_index.shape).cuda()
                    tc_index_val = tc_index * torch.ones(wd_index.shape).cuda()
                    bs_index_val = bs_index * torch.ones(wd_index.shape).cuda()

                    avg_wd_index = torch.mean(wd_index_val.float())
                    avg_sp_index = torch.mean(sp_index_val.float())
                    avg_tc_index = torch.mean(tc_index_val.float())
                    avg_bs_index = torch.mean(bs_index_val.float())
                    acc = torch.sum(wd_index) + torch.sum(sp_index) + torch.sum(tc_index) + torch.sum(bs_index)
                    if not sep:
                        acc = acc/(len(wd_index)*4)
                    else:
                        acc = acc

                    Complicated = False
                    if Complicated and sep:
                        # neg: [-10,0]
                        reward_world = torch.where(wd_index == True, 5 * world_prob, 3*torch.clip(torch.log(world_prob), min=-6)) # pos: neglect, since we have acc based
                        reward_sports = torch.where(sp_index == True, 5 * sports_prob, 3*torch.clip(torch.log(sports_prob), min=-6))
                        reward_tech = torch.where(tc_index == True, 5 * tech_prob, 3*torch.clip(torch.log(tech_prob), min=-6))
                        reward_business = torch.where(bs_index == True, 5 * business_prob, 3*torch.clip(torch.log(business_prob), min=-6))
                    elif not Complicated and sep:
                        reward_world = torch.where(world_prob-world_predict_best_prob>=0, world_prob * 10, 15*(world_prob-world_predict_best_prob))
                        reward_sports = torch.where(sports_prob-sports_predict_best_prob>=0, sports_prob * 10, 15*(sports_prob-sports_predict_best_prob))
                        reward_tech = torch.where(tech_prob-tech_predict_best_prob>=0, tech_prob * 10, 15*(tech_prob-tech_predict_best_prob))
                        reward_business = torch.where(business_prob-business_predict_best_prob>=0, business_prob * 10, 15*(business_prob-business_predict_best_prob))

                    # batch size vector
                    # pos
                    #wd_index = torch.gt(reward_world, 0)
                    #sp_index = torch.gt(reward_sports, 0)
                    #tc_index = torch.gt(reward_tech, 0)
                    #bs_index = torch.gt(reward_business, 0)
                    
                    count_based = False
                    if count_based:
                        avg_world_reward = torch.mean(reward_world)
                        avg_sports_reward = torch.mean(reward_sports)
                        avg_tech_reward = torch.mean(reward_tech)
                        avg_business_reward = torch.mean(reward_business)

                        if True:
                            #acc = torch.sum(wd_index) + torch.sum(sp_index) + torch.sum(tc_index) + torch.sum(bs_index)
                            #acc = acc/(len(wd_index)*4)
                            #print(acc) world_predict_best_prob
                            world_predict_sec_prob = torch.topk(world_probs, k=2, dim=-1).values[1]
                            sports_predict_sec_prob = torch.topk(sports_probs, k=2,dim=-1).values[1]
                            tech_predict_sec_prob = torch.topk(tech_probs, k=2,dim=-1).values[1]
                            business_predict_sec_prob = torch.topk(business_probs, k=2,dim=-1).values[1]
                            world_ratio = world_predict_best_prob/world_predict_sec_prob
                            sports_ratio = sports_predict_best_prob/sports_predict_sec_prob
                            tech_ratio = tech_predict_best_prob/tech_predict_sec_prob
                            business_ratio = business_predict_best_prob/business_predict_sec_prob
                            # variation
                            reward_world = torch.where(world_prob-world_predict_best_prob>=0, torch.clip(world_ratio,max=8) * 3, 20*(world_prob-world_predict_best_prob))
                            reward_sports = torch.where(sports_prob-sports_predict_best_prob>=0, torch.clip(sports_ratio,max=8) * 3, 20*(sports_prob-sports_predict_best_prob))
                            reward_tech = torch.where(tech_prob-tech_predict_best_prob>=0, torch.clip(tech_ratio,max=8) * 3, 20*(tech_prob-tech_predict_best_prob))
                            reward_business = torch.where(business_prob-business_predict_best_prob>=0, torch.clip(business_ratio,max=8) * 3, 20*(business_prob-business_predict_best_prob))
                            reward = (reward_world + reward_sports + reward_tech + reward_business)/4
                            
                            average_reward = torch.mean(reward)
                        # sum over


                    else:
                        #whole_index = wd_index * sp_index * tc_index * bs_index
                        #reward_world = torch.where(whole_index < 0, reward_world, reward_world - 3.5)
                        #reward_sports = torch.where(whole_index < 0, reward_sports, reward_sports - 3.5)
                        #reward_tech = torch.where(whole_index < 0, reward_tech, reward_tech - 3.5)
                        #reward_business = torch.where(whole_index < 0, reward_business, reward_business - 3.5)
                        world_predict_sec_prob = torch.topk(world_probs, k=2, dim=-1).values[:,1] # :,
                        sports_predict_sec_prob = torch.topk(sports_probs, k=2,dim=-1).values[:,1]
                        tech_predict_sec_prob = torch.topk(tech_probs, k=2,dim=-1).values[:,1]
                        business_predict_sec_prob = torch.topk(business_probs, k=2,dim=-1).values[:,1]
                        # reward, BIAS: 5*torch.clip(world_predict_best_prob/world_predict_sec_prob, max=5
                        world_flag = (world_prob-world_predict_best_prob>=0)
                        sports_flag = (sports_prob-sports_predict_best_prob>=0)
                        tech_flag = (tech_prob-tech_predict_best_prob>=0)
                        business_flag = (business_prob-business_predict_best_prob>=0)
                        # reward
                        class_flag = world_flag.int() + sports_flag.int() + tech_flag.int() + business_flag.int()
                        # print('class_flag', class_flag)
                        
                        reward_world = torch.where(world_prob-world_predict_best_prob>=0, 2*(world_predict_best_prob-world_predict_sec_prob), 1.8*(world_prob-world_predict_best_prob))
                        #reward = torch.where(class_flag >=2, reward, )
                        #average_reward += reward
                        reward_sports = torch.where(sports_prob-sports_predict_best_prob>=0, 2*(sports_predict_best_prob-sports_predict_sec_prob), 1.8*(sports_prob-sports_predict_best_prob))
                        reward_tech = torch.where(tech_prob-tech_predict_best_prob>=0, 2*(tech_predict_best_prob-tech_predict_sec_prob), 1.8*(tech_prob-tech_predict_best_prob))
                        reward_business = torch.where(business_prob-business_predict_best_prob>=0, 2*(business_predict_best_prob-business_predict_sec_prob), 1.8*(business_prob-business_predict_best_prob))
                        average_reward = torch.mean(reward_world + reward_sports + reward_tech + reward_business) * 25 #acc * 20#torch.mean(reward)#reward * torch.exp(acc) #* torch.log(10 * acc)#torch.mean(reward)
                        
                        #average_reward = torch.mean(reward)
                        #if acc > 0.7:
                        #    average_reward += 5
                elif mode == 'infer':
                    world_logits = probs[batch_index * 4, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                    sports_logits = probs[batch_index * 4 + 1, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                    business_logits = probs[batch_index * 4 + 2, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                    technology_logits =probs[batch_index * 4 + 3, [self.world_id, self.sports_id, self.business_id, self.technology_id]]
                    
                    world_probs = torch.softmax(world_logits, -1)
                    sports_probs = torch.softmax(sports_logits, -1)
                    tech_probs = torch.softmax(technology_logits, -1)
                    business_probs = torch.softmax(business_logits, -1)
                    # probs shape
                    #print(tech_probs.shape)
                    # prob
                    world_prob = world_probs[0]
                    sports_prob = sports_probs[1]
                    tech_prob = tech_probs[3]
                    business_prob = business_probs[2]
                    # predicted label
                    world_predict_label = torch.argmax(world_probs, dim=-1)
                    sports_predict_label = torch.argmax(sports_probs, dim=-1)
                    tech_predict_label = torch.argmax(tech_probs, dim=-1)
                    business_predict_label = torch.argmax(business_probs, dim=-1)

                if mode == 'train':
                    quantities_to_log["avg_world_reward"].append(torch.mean(reward_world).item())
                    quantities_to_log["avg_sports_reward"].append(torch.mean(reward_sports).item())
                    quantities_to_log["avg_tech_reward"].append(torch.mean(reward_tech).item())
                    quantities_to_log["avg_business_reward"].append(torch.mean(reward_business).item())

                    quantities_to_log["avg_reward"].append(average_reward.item())
            elif self._task_name in ['sst-5', 'yelp-5']:
                if mode =='train':
                    current_pt = [prompt_strings[batch_index] for _ in range(len(source_strings))]
                    formatted_current_prompt_sentences,_ = self._format_prompts(source_strings, current_pt, PAIR)
                    current_probs = self._get_probs(formatted_current_prompt_sentences, prompt_length)
                    # probs
                    probs_1 = torch.softmax(current_probs[first_index][:, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]], -1)
                    probs_2 = torch.softmax(current_probs[second_index][:, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]], -1)
                    probs_3 = torch.softmax(current_probs[third_index][:, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]], -1)
                    probs_4 = torch.softmax(current_probs[fourth_index][:, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]], -1)
                    probs_5 = torch.softmax(current_probs[fifth_index][:, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]], -1)
                    # prob
                    prob_1 = probs_1[:, 0]
                    prob_2 = probs_2[:, 1]
                    prob_3 = probs_3[:, 2]
                    prob_4 = probs_4[:, 3]
                    prob_5 = probs_5[:, 4]
                    # current best
                    predict_best_prob1 = torch.max(probs_1, dim=-1).values
                    predict_best_prob2 = torch.max(probs_2, dim=-1).values
                    predict_best_prob3 = torch.max(probs_3, dim=-1).values
                    predict_best_prob4 = torch.max(probs_4, dim=-1).values
                    predict_best_prob5 = torch.max(probs_5, dim=-1).values
                    # current labels
                    predict_label1 = torch.argmax(probs_1, dim=-1)
                    predict_label2 = torch.argmax(probs_2, dim=-1)
                    predict_label3 = torch.argmax(probs_3, dim=-1)
                    predict_label4 = torch.argmax(probs_4, dim=-1)
                    predict_label5 = torch.argmax(probs_5, dim=-1)
                    # whether predict correctly
                    index_1 = torch.where(predict_label1 == 0, True, False)
                    index_2 = torch.where(predict_label2 == 1, True, False)
                    index_3 = torch.where(predict_label3 == 2, True, False)
                    index_4 = torch.where(predict_label4 == 3, True, False)
                    index_5 = torch.where(predict_label5 == 4, True, False)
                    # acc
                    index1_val = index_1 * torch.ones(index_1.shape).cuda()
                    index2_val = index_2 * torch.ones(index_2.shape).cuda()
                    index3_val = index_3 * torch.ones(index_3.shape).cuda()
                    index4_val = index_4 * torch.ones(index_4.shape).cuda()
                    index5_val = index_5 * torch.ones(index_5.shape).cuda()

                    acc = torch.sum(index_1) + torch.sum(index_2) + torch.sum(index_3) + torch.sum(index_4) + torch.sum(index_5)
                    acc = acc/(len(index_1)*5)
                    
                    predict_sec_prob1 = torch.topk(probs_1, k=2, dim=-1).values[:,1]
                    predict_sec_prob2 = torch.topk(probs_2, k=2, dim=-1).values[:,1]
                    predict_sec_prob3 = torch.topk(probs_3, k=2, dim=-1).values[:,1]
                    predict_sec_prob4 = torch.topk(probs_4, k=2, dim=-1).values[:,1]
                    predict_sec_prob5 = torch.topk(probs_5, k=2, dim=-1).values[:,1]
                    # reward
                    reward_1 = torch.where(prob_1 - predict_best_prob1>=0, 2*(predict_best_prob1-predict_sec_prob1), 1.8*(prob_1-predict_best_prob1))
                    reward_2 = torch.where(prob_2 - predict_best_prob2>=0, 2*(predict_best_prob2-predict_sec_prob2), 1.8*(prob_2-predict_best_prob2))
                    reward_3 = torch.where(prob_3 - predict_best_prob3>=0, 2*(predict_best_prob3-predict_sec_prob3), 1.8*(prob_3-predict_best_prob3))
                    reward_4 = torch.where(prob_4 - predict_best_prob4>=0, 2*(predict_best_prob4-predict_sec_prob4), 1.8*(prob_4-predict_best_prob4))
                    reward_5 = torch.where(prob_5 - predict_best_prob5>=0, 2*(predict_best_prob5-predict_sec_prob5), 1.8*(prob_5-predict_best_prob5))
                    
                    average_reward = torch.mean(reward_1+reward_2+reward_3+reward_4+reward_5) * 20
                
                    quantities_to_log["avg_1_reward"].append(torch.mean(reward_1).item())
                    quantities_to_log["avg_2_reward"].append(torch.mean(reward_2).item())
                    quantities_to_log["avg_3_reward"].append(torch.mean(reward_3).item())
                    quantities_to_log["avg_4_reward"].append(torch.mean(reward_4).item())
                    quantities_to_log["avg_5_reward"].append(torch.mean(reward_5).item())

                    quantities_to_log["avg_reward"].append(average_reward.item())
                elif mode == 'infer':
                    first_logit = probs[batch_index * 5, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]]
                    second_logit = probs[batch_index * 5 + 1, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]]
                    third_logit = probs[batch_index * 5 + 2, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]]
                    fourth_logit = probs[batch_index * 5 + 3, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]]
                    fifth_logit = probs[batch_index * 5 + 4, [self.id_1, self.id_2, self.id_3, self.id_4, self.id_5]]

                    first_probs = torch.softmax(first_logit, -1)
                    second_probs = torch.softmax(second_logit, -1)
                    third_probs = torch.softmax(third_logit, -1)
                    fourth_probs = torch.softmax(fourth_logit, -1)
                    fifth_probs = torch.softmax(fifth_logit, -1)

                    first_prob = first_probs[0]
                    second_prob = second_probs[1]
                    third_prob = third_probs[2]
                    fourth_prob = fourth_probs[3]
                    fifth_prob = fifth_probs[4]

                    # predict label
                    first_predict_label = torch.argmax(first_probs, dim=-1)
                    second_predict_label = torch.argmax(second_probs, dim=-1)
                    third_predict_label = torch.argmax(third_probs, dim=-1)
                    fourth_predict_label = torch.argmax(fourth_probs, dim=-1)
                    fifth_predict_label = torch.argmax(fifth_probs, dim=-1)

            if mode == 'train':
                input_rewards['a'] += [average_reward.item()]
            
            self._counter += 1
            if mode == 'train':
                if self._task_name in ['SST-2', 'yelp-2', 'mr', 'cr', 'RTE', 'MRPC']:
                    print(prompts[batch_index], '\n', 
                    formatted_prompts[batch_index * 2], '\n', 
                    formatted_prompts[batch_index * 2 + 1], '\n', 
                    'acc:', acc.item(), '|',
                    'Reward:', round(average_reward.item(), 2))
                elif self._task_name == 'agnews':
                    print(prompts[batch_index], '\n',
                    formatted_prompts[batch_index * 4], '\n',
                    formatted_prompts[batch_index * 4 + 1], '\n',
                    formatted_prompts[batch_index * 4 + 2], '\n',
                    formatted_prompts[batch_index * 4 + 3], '\n',
                    #'world prob:', world_prob.item(), '|',
                    #'sports prob:', sports_prob.item(), '|',
                    #'business prob:', business_prob.item(), '|',
                    #'tech prob:', tech_prob.item(), '|',
                    'Avg Reward:', round(average_reward.item(), 2), '|',
                    'Acc:', acc.item())
                elif self._task_name in ['sst-5', 'yelp-5']:
                    print(prompts[batch_index], '\n',
                    formatted_prompts[batch_index * 5], '\n',
                    formatted_prompts[batch_index * 5 + 1], '\n',
                    formatted_prompts[batch_index * 5 + 2], '\n',
                    formatted_prompts[batch_index * 5 + 3], '\n',
                    formatted_prompts[batch_index * 5 + 4], '\n',
                    #'1 prob:', prob_1.item(), '|',
                    #'2 prob:', prob_2.item(), '|',
                    #'3 prob:', prob_3.item(), '|',
                    #'4 prob:', prob_4.item(), '|',
                    'Acc:', acc.item(), '|',
                    'Reward:', round(average_reward.item(), 2))
            
                rewards.append(average_reward)
            if mode == 'infer': # val score
                if self._task_name in ['SST-2', 'yelp-2', 'mr', 'cr']:
                    pos_acc = torch.tensor(1) if pos_pos_prob >= 0.5 else torch.tensor(0)
                    neg_acc = torch.tensor(1) if neg_neg_prob >= 0.5 else torch.tensor(0)

                    rewards.append((pos_acc+neg_acc)/2)
                elif self._task_name == 'agnews':
                    world_acc = torch.tensor(1) if world_predict_label == 0 else torch.tensor(0)
                    sports_acc = torch.tensor(1) if sports_predict_label == 1 else torch.tensor(0)
                    business_acc = torch.tensor(1) if business_predict_label == 2 else torch.tensor(0)
                    tech_acc = torch.tensor(1) if tech_predict_label == 3 else torch.tensor(0)
                    
                    rewards.append(torch.tensor((world_acc+sports_acc+business_acc+tech_acc)/4).float())
                elif self._task_name in ['sst-5', 'yelp-5']:
                    acc_1 = torch.tensor(1) if first_predict_label == 0 else torch.tensor(0)
                    acc_2 = torch.tensor(1) if second_predict_label == 1 else torch.tensor(0)
                    acc_3 = torch.tensor(1) if third_predict_label == 2 else torch.tensor(0)
                    acc_4 = torch.tensor(1) if fourth_predict_label == 3 else torch.tensor(0)
                    acc_5 = torch.tensor(1) if fifth_predict_label == 4 else torch.tensor(0)

                    rewards.append(torch.tensor((acc_1+acc_2+acc_3+acc_4+acc_5)/5).float())

            
        rewards_tensor = torch.stack(rewards)
        
        if mode=='train':
            batch_zscore = True
            if batch_zscore:
                input_reward_means = {k:np.mean(v) for k,v in input_rewards.items()}
                input_reward_stds = {k:np.std(v) for k,v in input_rewards.items()}
                # not source strings
                idx_means = torch.tensor(input_reward_means['a']).float()
                idx_stds = torch.tensor(input_reward_stds['a']).float()
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

