# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import os
import json
import torch
import torch.nn as nn
import texar.torch as tx
from functools import partial
from typing import List, Tuple, Union, Dict, Optional, Callable, Any, cast

from configs.models import (
    config_model_transformers_small)

from sql.utils import ForwardMode
from sql.types import (
    BatchType,
    FloatTensor,
    LongTensor)

from transformers import pipeline, AutoTokenizer
import random
import itertools

def _build_gpt2_vocab_mlp(out_dim, in_dim=768, device=0): 
    
    W1 = nn.Linear(in_dim, 2048)
    A1 = nn.ReLU()
    W2 = nn.Linear(2048, out_dim)
    return nn.Sequential(W1, A1, W2)

def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    r"""Adapted from
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L63-L77
    """
    if k == 0:
        # no truncation
        return logits

    values, _ = torch.topk(logits, k=k)
    min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float('-inf')), logits)

def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    r"""Adapted from
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317#file-top-k-top-p-py-L16-L27"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the
    # threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    for idx in range(logits.size(0)):
        batch_indices = sorted_indices[idx, sorted_indices_to_remove[idx]]
        logits[idx, batch_indices] = float("-inf")
    return logits

class GPT2ConditionedMLP(nn.Module): 
    tmp_input = 'classification'

    def __init__(self, 
                # General model parameters
                 train_data: tx.data.PairedTextData,
                 max_source_length: int,
                 max_decoding_length: int,
                 config_name: str,
                 # General prompt parameters
                 prompt_dataset: Optional[str] = None,
                 prompt_dataset_seed: Optional[int] = None,
                 prompt_dataset_basepath: str = '/data/mingkai/prompt-generation/dirty-code/rl-prompt',
                 # MLP-specific parameters
                 mlp_policy_lm: str = 'distilgpt2',
                 mlp_input_specific: bool = False,
                 mlp_logit_bias: float = 0,
                 mlp_n_repeats: int = 4,
                 mlp_fluent_prompt: bool = False,
                 **kwargs) -> None: 
        super().__init__()
        
        if config_name not in ['gpt2_conditioned_mlp']: 
            raise ValueError
            
        self.device = 0 # TODO

        self.max_source_length = max_source_length
        self.max_decoding_length = max_decoding_length
        
        self.source_vocab = train_data.source_vocab
        self.target_vocab = train_data.target_vocab
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id
        
        self.dataset = prompt_dataset
        self.seed = prompt_dataset_seed
        if prompt_dataset_basepath == '.': # TODO
            self.basepath = os.path.abspath(os.path.join('.', os.pardir, os.pardir, os.pardir))
        else:
            self.basepath = prompt_dataset_basepath
        self.n_repeats = mlp_n_repeats
        self.fluent = mlp_fluent_prompt
        self.logit_bias = mlp_logit_bias
        
        model = mlp_policy_lm
        print('Policy LM:', model)
        self.tokenizer = AutoTokenizer.from_pretrained(model, pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=model,
                                  device=self.device)
        for param in self.generator.model.parameters():
            param.requires_grad = False
        
        model_sizes = {'distilgpt2': 768,
                       'gpt2-medium': 1024,
                       'gpt2-large': 1280}
        model_dim = model_sizes[model]
        self.mlp = _build_gpt2_vocab_mlp(model_dim, in_dim=model_dim).to(self.device)
        self._mlp_forward = self._gpt2_vocab_mlp_forward
        self.valid_token_ids = None
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(init_weights)

        self.input_specific = mlp_input_specific
        if self.input_specific:
            self._tst_inputs = self._load_tst_inputs()
            self._tst_inputs_idx = {('train', 'LABEL_0'): 0, 
                                    ('train', 'LABEL_1'): 0,
                                    ('infer', 'LABEL_0'): 0,
                                    ('infer', 'LABEL_1'): 0}
    
    
    def _gpt2_vocab_mlp_forward(self, state): 
        
        mlp_output = self.mlp(state)
        
        logits = self.generator.model.lm_head(mlp_output)
        if self.valid_token_ids is not None: 
            logits = logits[:, self.valid_token_ids]
            
        if self.fluent: 
            plm_logits = self.generator.model.lm_head(state)
            values, _ = torch.topk(plm_logits, k=20)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                plm_logits < min_values,
                torch.full_like(logits, float('-inf')), logits)
        zeros = torch.ones_like(logits)[:, :4] * float('-inf')
        modified_logits = torch.cat([zeros, logits], dim=-1)
        return modified_logits
    
    def _load_tst_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        tst_inputs = {}
        print(os.path.abspath(self.basepath))
        assert self.dataset in ['yelp', 'shakespeare'], self.dataset
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
                sentences_test_0 = [line.strip() for line in f]
            with open(filepath_test_ref_1) as f: 
                sentences_test_1 = [line.strip() for line in f]
            test_size = 16
                
        elif self.dataset in ['shakespeare']:
            seed_dic = {0:f'100-100', 1:f'100-13', 2:f'100-21', 3:f'100-42', 4:f'100-87'}
            
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
            
            if self.dataset == 'shakespeare' and self.seed in [0, 1, 2, 3, 4]: 
                new_train_0 = [sent for sent in sentences_train_0 if len(self.tokenizer(sent)['input_ids']) < 30]
                new_train_1 = [sent for sent in sentences_train_1 if len(self.tokenizer(sent)['input_ids']) < 30]
                new_dev_0 = [sent for sent in sentences_dev_0 if len(self.tokenizer(sent)['input_ids']) < 30]
                new_dev_1 = [sent for sent in sentences_dev_1 if len(self.tokenizer(sent)['input_ids']) < 30]
                
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
            
        idx = 0
        
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
        
        tst_inputs[('test', 'LABEL_0')] = sentences_test_1
        tst_inputs[('test', 'LABEL_1')] = sentences_test_0
        
        return tst_inputs
        
    def decode_teacher_forcing(self,
                               batch: BatchType,
                               last_token_hidden_state,
                               past_key_values) \
    -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        
        state = last_token_hidden_state
        sample_ids, sample_logits = batch['target_text_ids'][:, 1:], []
        
        for i in range(self.max_decoding_length): 
            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias
            
            actions = sample_ids[:, i]
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
            sample_logits.append(logits.unsqueeze(dim=1))
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            token_encoding = (self.generator
                               .tokenizer(tokens, 
                                          padding=True,
                                          return_tensors='pt')
                               .to(self.device))
            input_ids = token_encoding['input_ids']
            input_lengths = token_encoding['attention_mask'].sum(dim=1)

            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            state = next_outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                   (input_lengths - 1)]
            past_key_values = next_outputs.past_key_values
            
        sample_logits = torch.cat(sample_logits, dim=1)
            
        decoder_output = tx.modules.TransformerDecoderOutput(
            logits=sample_logits,
            sample_id=sample_ids
        )
        return decoder_output, None
    
    def decode_sampling(self,
                        batch: BatchType,
                        last_token_hidden_state,
                        past_key_values,
                        top_k: Optional[int] = None,
                        top_p: Optional[float] = None) \
    -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        if top_k is not None and top_p is not None:
            raise ValueError

        state = last_token_hidden_state
        prompt_tokens, sample_ids, sample_logits = [], [], []
        for i in range(self.max_decoding_length): 
            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias
            # print(logits[:, 4:].min().item(), logits.max().item())
            
            if top_k is not None: sampling_logits = _top_k_logits(logits, k=top_k)
            elif top_p is not None: sampling_logits = _top_p_logits(logits, p=top_p)
            else: sampling_logits = logits
            
            actions = (torch.distributions.categorical
                       .Categorical(logits=sampling_logits)
                       .sample())
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
            
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            
            token_encoding = (self.generator
                               .tokenizer(tokens, 
                                          padding=True,
                                          return_tensors='pt')
                               .to(self.device))
            input_ids = token_encoding['input_ids']
            input_lengths = token_encoding['attention_mask'].sum(dim=1)

            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            state = next_outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                   (input_lengths - 1)]
            past_key_values = next_outputs.past_key_values
            
        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
            
        decoder_output = tx.modules.TransformerDecoderOutput(
            logits=sample_logits,
            sample_id=sample_ids
        )
        return (decoder_output, 
                torch.tensor([self.max_decoding_length \
                              for _ in range(sample_ids.shape[0])]).to(self.device)
               )

    def decode_greedy(
            self,
            batch: BatchType,
            last_token_hidden_state,
            past_key_values,
            corruption_p: Optional[float] = None,
            input_mode='train',
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        state = last_token_hidden_state
        prompt_tokens, sample_ids, sample_logits = [], [], []
        for i in range(self.max_decoding_length): 
            logits = self._mlp_forward(state) # [batch_size, vocab_size]
            
            actions = logits.argmax(dim=-1) # [batch_size]
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
            
            sample_ids.append(actions.unsqueeze(dim=1)) # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1)) # [batch_size, 1, vocab_size]
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            token_encoding = (self.generator
                               .tokenizer(tokens, 
                                          padding=True,
                                          return_tensors='pt')
                               .to(self.device))
            input_ids = token_encoding['input_ids']
            input_lengths = token_encoding['attention_mask'].sum(dim=1)

            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            state = next_outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                   (input_lengths - 1)]
            past_key_values = next_outputs.past_key_values
            
        sample_ids = torch.cat(sample_ids, dim=1) # [batch_size, prompt_length]
        sample_logits = torch.cat(sample_logits, dim=1) # [batch_size, prompt_length, vocab_size]
            
        decoder_output = tx.modules.TransformerDecoderOutput(
            logits=sample_logits,
            sample_id=sample_ids
        )
        return {
                "sample_id": (
                    decoder_output
                    .sample_id
                    .unsqueeze(dim=-1)
                )
            }
    
    def _get_inputs(self, mode: str, target_labels: List[str]): 
        inputs = []
        indices = []
        
        for i, label in enumerate(target_labels): 
            if self.input_specific:
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
                
                if idx == 0: 
                    data_idx = np.arange(len(data) // self.n_repeats) * self.n_repeats
                    random.Random(0).shuffle(data_idx)
                    self._tst_inputs[(mode, label)] = \
                        (list(itertools.chain(*[[data[i] for _ in range(self.n_repeats)]
                                                for i in data_idx])))
                    
                self._tst_inputs_idx[(mode, label)] = idx
            else: 
                inputs.append(self.tmp_input)
                indices.append(int(i))
        
        return indices, inputs
    
    def forward(self,
                batch: BatchType,
                mode: ForwardMode,
                **kwargs) -> Union[Tuple[tx.modules.TransformerDecoderOutput, 
                                         LongTensor], 
                                   Dict]:        
                                   
        target_labels = [t[0] for t in batch['source_text']]
        if mode in [ForwardMode.INFER]: 
            input_mode = 'infer'
        else: 
            input_mode = 'train'
        indices, input_texts = self._get_inputs(input_mode, target_labels)
        
        token_encoding = (self.generator
                          .tokenizer(input_texts, 
                                     padding=True,
                                     return_tensors='pt')
                          .to(self.device))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        outputs = (self.generator.model
                   .transformer(input_ids, use_cache=True))
        last_token_hidden_state = outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                            (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        
        if mode in [ForwardMode.MLE, ForwardMode.SQL_OFF_GT]:
            return self.decode_teacher_forcing(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values,
                **kwargs)

        if mode in [ForwardMode.PG, ForwardMode.SQL_ON]:
            return self.decode_sampling(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values,
                **kwargs)

        if mode in [ForwardMode.INFER]:
            return self.decode_greedy(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values,
                input_mode=input_mode,
                **kwargs)

        raise ValueError(f"Unknown mode {mode}")


class Transformer(nn.Module):
    r"""A standalone sequence-to-sequence Transformer model, from "Attention
    Is All You Need". The Transformer model consists of the word embedding
    layer, position embedding layer, an encoder and a decoder. Both encoder
    and decoder are stacks of self-attention layers followed by feed-forward
    layers. See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    for the full description of the model.
    """

    def __init__(
            self,
            train_data: tx.data.PairedTextData,
            max_source_length: int,
            max_decoding_length: int,
            config_name: str,
    ) -> None:
        super().__init__()

        if config_name not in ["transformer_small"]:
            raise ValueError

        if config_name == "transformer_small":
            config_model = config_model_transformers_small

        self.config_model = config_model
        self.max_source_length = max_source_length
        self.max_decoding_length = max_decoding_length

        self.source_vocab = train_data.source_vocab
        self.target_vocab = train_data.target_vocab
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id

        self.source_embedder = tx.modules.WordEmbedder(
            vocab_size=self.source_vocab_size,
            hparams=self.config_model.emb)

        self.target_embedder = tx.modules.WordEmbedder(
            vocab_size=self.target_vocab_size,
            hparams=self.config_model.emb)

        self.source_pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.max_source_length,
            hparams=self.config_model.position_embedder_hparams)

        self.target_pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.max_decoding_length,
            hparams=self.config_model.position_embedder_hparams)

        self.encoder = tx.modules.TransformerEncoder(
            hparams=self.config_model.encoder)
        self.decoder = tx.modules.TransformerDecoder(
            token_pos_embedder=partial(
                self._embedding_fn,
                source_or_target="target"),
            vocab_size=self.target_vocab_size,
            output_layer=self.target_embedder.embedding,
            hparams=self.config_model.decoder)

    def _embedding_fn(
            self,
            tokens: LongTensor,
            positions: LongTensor,
            source_or_target: str,
    ) -> FloatTensor:
        if source_or_target not in ["source", "target"]:
            raise ValueError

        if source_or_target == "source":
            word_embed = self.source_embedder(tokens)
            pos_embed = self.source_pos_embedder(positions)
        if source_or_target == "target":
            word_embed = self.target_embedder(tokens)
            pos_embed = self.target_pos_embedder(positions)

        scale = self.config_model.hidden_dim ** 0.5
        return word_embed * scale + pos_embed

    def decode_teacher_forcing(
            self,
            batch: BatchType,
            memory: FloatTensor
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        decoder_outputs = self.decoder(
            memory=memory,
            memory_sequence_length=batch["source_length"],
            inputs=batch["target_text_ids"][:, :-1],
            sequence_length=batch["target_length"] - 1,
            decoding_strategy="train_greedy")

        # label_lengths = (labels != 0).long().sum(dim=1)
        # We don't really need `sequence_lengths` here
        return decoder_outputs, None

    def decode_greedy(
            self,
            batch: BatchType,
            memory: FloatTensor,
            corruption_p: Optional[float] = None,
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if corruption_p is not None:
            raise NotImplementedError("Deprecated")

        return self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_greedy",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=min(
                self.max_decoding_length,
                batch["target_length"].max().item() - 1))

    def decode_sampling(
            self,
            batch: BatchType,
            memory: FloatTensor,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        if top_k is not None and top_p is not None:
            raise ValueError

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if top_k is not None:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                top_k=top_k)

        if top_p is not None:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                p=top_p)

        decoder_output = self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_sample",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=min(
                self.max_decoding_length,
                batch["target_length"].max().item() - 1))
        
        sample_logits = decoder_output[0].logits
        print(sample_logits.min().item(), sample_logits.max().item())
        return decoder_output

    def decode_beam_search(
            self,
            batch: BatchType,
            memory: FloatTensor,
            beam_width: int,
            corruption_p: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:

        # Only greedy decoding is support for this as of now.
        if corruption_p is not None:
            if beam_width != 1:
                raise NotImplementedError

        # when `beam_width in [None, 1]`, `self.decoder`
        # will switch to default decoding mode, which is
        # not necessarily what we want. Instead, let's
        # explicitly call greedy-decoding.
        # https://sourcegraph.com/github.com/asyml/texar-pytorch/-/blob/texar/torch/modules/decoders/rnn_decoders.py#L717:9
        if beam_width > 1:

            start_tokens = memory.new_full(
                batch["target_length"].size(),
                self.bos_token_id,
                dtype=torch.int64)

            return self.decoder(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                memory=memory,
                memory_sequence_length=batch["source_length"],
                beam_width=beam_width,
                max_decoding_length=self.max_decoding_length)

        else:
            infer_outputs, _ = self.decode_greedy(
                batch=batch,
                memory=memory,
                corruption_p=corruption_p)

            return {
                "sample_id": (
                    infer_outputs
                    .sample_id
                    .unsqueeze(dim=-1)
                )
            }

    def forward(
            self,
            batch: BatchType,
            mode: ForwardMode,
            **kwargs,
    ) -> Union[Tuple[tx.modules.TransformerDecoderOutput, LongTensor], Dict]:

        # Text sequence length excluding padding
        if not (batch["source_length"] == (batch["source_text_ids"] != 0).int().sum(dim=1)).all():
            raise ValueError

        positions: LongTensor = (
            torch.arange(
                batch["source_length"].max(),  # type: ignore
                dtype=torch.long,
                device=batch["source_text_ids"].device)
            .unsqueeze(0)
            .expand(batch["source_text_ids"].size(0), -1)
        )

        encoder_output = self.encoder(
            inputs=self._embedding_fn(
                tokens=batch["source_text_ids"],
                positions=positions,
                source_or_target="source"),
            sequence_length=batch["source_length"])

        if mode in [ForwardMode.MLE, ForwardMode.SQL_OFF_GT]:
            return self.decode_teacher_forcing(
                batch=batch,
                memory=encoder_output)

        if mode in [ForwardMode.PG, ForwardMode.SQL_ON]:
            return self.decode_sampling(
                batch=batch,
                memory=encoder_output,
                **kwargs)

        if mode in [ForwardMode.INFER]:
            return self.decode_beam_search(
                batch=batch,
                memory=encoder_output,
                **kwargs)

        raise ValueError(f"Unknown mode {mode}")
