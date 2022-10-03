import sys
sys.path.append('..')
import hydra
from typing import Optional, Tuple, List
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          GPT2LMHeadModel,
                          AutoModelForMaskedLM)

SUPPORTED_LEFT_TO_RIGHT_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                               'gpt2-large', 'gpt2-xl']
SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large']


class PromptedClassificationEvaluator:
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str],
        prompt: str
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.task_lm = task_lm
        print("Task LM:", self.task_lm)
        if is_mask_lm is None: 
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if 'bert' in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm  
        if self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm,
                                truncation_side="left")
            self._generator = (AutoModelForMaskedLM
                               .from_pretrained(self.task_lm)
                               .to(self.device))
        else:
            assert self.task_lm in SUPPORTED_LEFT_TO_RIGHT_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.task_lm, pad_token='<|endoftext|>')
            self._generator = (GPT2LMHeadModel
                               .from_pretrained(self.task_lm,
                                   truncation_side="left")
                               .to(self.device))
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id
        self.num_classes = num_classes
        self.verbalizers = verbalizers

        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v)
                               for v in self.verbalizers]
        if template is None:
            self.template = self.load_default_template()  # prompt templates
        else:
            self.template = template

        self.prompt = prompt

    # Adapted from
    # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(
            input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def load_default_template(self) -> List[str]:
        if self.is_mask_lm:
            template = "{sentence_1} {prompt} <mask> ."
        else:
            # Template for left-to-right LMs like GPT-2
            template = "{sentence_1} {prompt}"

        return template

    @torch.no_grad()
    def _get_logits(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        # for MLM, add mask token
        batch_size = len(texts)
        encoded_inputs = self._tokenizer(texts, padding='longest',
                                         truncation=True, return_tensors="pt",
                                         add_special_tokens=True)

        if self.is_mask_lm:
            # self.ensure_exactly_one_mask_token(encoded_inputs) TODO
            token_logits = self._generator(
                **encoded_inputs.to(self.device)).logits
            mask_token_indices = \
                self._get_mask_token_index(encoded_inputs['input_ids'])
            out_logits = token_logits[range(batch_size), mask_token_indices, :]
        else:
            token_logits = self._generator(
                **encoded_inputs.to(self.device)).logits
            input_lengths = encoded_inputs['attention_mask'].sum(dim=1)
            out_logits = token_logits[range(batch_size), input_lengths - 1, :]

        return out_logits

    def _format_prompts(
        self,
        prompts: List[str],
        source_strs: List[str]
    ) -> List[str]:
        return [self.template.format(sentence_1=s_1, prompt=prompt)
                for s_1, prompt in zip(source_strs, prompts)]

    def forward(
        self,
        dataloader
    ) -> float:
        num_of_examples = dataloader.dataset.__len__()
        correct_sum = 0
        for i, batch in enumerate(dataloader):
            inputs = batch['source_texts']  # List
            targets = batch['class_labels']  # Tensor
            batch_size = targets.size(0)
            current_prompts = [self.prompt for _ in range(batch_size)]
            formatted_templates = self._format_prompts(current_prompts, inputs)
            all_logits = self._get_logits(formatted_templates)
            class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
            # Get labels
            predicted_labels = torch.argmax(class_probs, dim=-1)
            label_agreement = torch.where(
                targets.cuda() == predicted_labels, 1, 0)
            # Compute accuracy
            correct_sum += label_agreement.sum()
        accuracy = correct_sum/num_of_examples
        return accuracy
