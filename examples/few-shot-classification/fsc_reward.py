import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import defaultdict
from rlprompt.rewards import BaseReward

SUPPORTED_LEFT_TO_RIGHT_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                               'gpt2-large', 'gpt2-xl']
SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large']


class PromptedClassificationReward(BaseReward):
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        compute_zscore: bool,
        incorrect_coeff: float, # lambda_1 in paper
        correct_coeff: float, # lambda_2 in paper
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str]
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.task_lm = task_lm
        if is_mask_lm is None: 
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if 'bert' in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm  
        print('Task LM:', self.task_lm)
        if self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
            self._generator = (AutoModelForMaskedLM
                               .from_pretrained(self.task_lm)
                               .to(self.device))
        else:
            assert self.task_lm in SUPPORTED_LEFT_TO_RIGHT_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.task_lm, pad_token='<|endoftext|>')
            self._generator = (GPT2LMHeadModel
                               .from_pretrained(self.task_lm)
                               .to(self.device))
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id


        self.compute_zscore = compute_zscore
        self.incorrect_coeff = incorrect_coeff
        self.correct_coeff = correct_coeff
        self.num_classes = num_classes
        self.verbalizers = verbalizers
        print('Verbalizers:', self.verbalizers)
        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v)
                               for v in self.verbalizers]
        if template is None:
            self.template = self.load_default_template()  # prompt templates
        else: 
            self.template = template
        self._counter = 0

    def load_default_template(self) -> str:
        if self.is_mask_lm:
            mask_token = self._tokenizer.mask_token
            template = f"{{sentence_1}} {{prompt}} {mask_token} ."
        else:
            # Template for left-to-right LMs like GPT-2
            template = "{sentence_1} {prompt}"
        return template

    def forward(
        self,
        source_texts: List[str],
        class_labels: List[int],
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        assert mode in ["train", "infer"]
        
        if mode == "train":
            self._counter += 1

        # Process prompts and verbalizer indices
        prompt_tokens = output_tokens
        prompt_strings = self._convert_tokens_to_string(prompt_tokens)
        batch_size = len(source_texts)

        rewards: List[torch.Tensor] = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for i, prompt in enumerate(prompt_strings):
            # Compute LM logits
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self._format_prompts(source_texts,
                                                       current_prompts)
            all_logits = self._get_logits(formatted_templates)
            # [batch_size, vocab_size]
            class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
            # [batch_size, num_classes]

            # Get label and maximum not-label probabilities
            label_probs = class_probs[range(batch_size), class_labels]
            # [batch_size, 1]
            not_label_probs = torch.where(
                class_probs == label_probs.unsqueeze(1),
                torch.Tensor([-1]).to(self.device), class_probs)
            # [batch_size, num_classes]
            max_not_label_probs, _ = torch.max(not_label_probs, -1)
            # [batch_size, 1]

            # Compute piecewise gap reward
            gap = (label_probs - max_not_label_probs)
            correct = (gap > 0).long()
            gap_rewards = gap * (self.correct_coeff * correct \
                                 + self.incorrect_coeff * (1 - correct))
            reward = gap_rewards.mean().detach()

            # Log quantities such as accuracy and class-wise reward
            acc = correct.float().mean()
            quantities_to_log['acc'] = acc
            for c in range(self.num_classes):
                class_idx = np.array(class_labels) == c
                class_rewards = gap_rewards[class_idx]
                quantities_to_log[f"gap_reward_class_{c}"].append(
                    class_rewards.mean().item())
            quantities_to_log['gap_reward'].append(reward.item())
            rewards.append(reward)

            # keep track of rewards for z-score normalization
            input_rewards['z'] += [reward.item()]

            # Print examples
            print_strs = [self._counter, '|', prompt, '\n']
            for c in range(self.num_classes):
                class_example_idx = np.where(np.array(class_labels) == c)[0][0]
                class_example = formatted_templates[class_example_idx]
                class_example_probs = class_probs[class_example_idx, :].tolist()
                class_example_probs = [round(prob, 2) \
                                       for prob in class_example_probs]
                print_strs += ['Class', c, 'Example:', 
                               class_example, '|',
                               'Probs:', class_example_probs, '\n']
            print_strs += ['Accuracy:', acc.item(), '|',
                           'Reward:', round(reward.item(), 2)]
            print(*print_strs)
        rewards_tensor = torch.stack(rewards)

        # z-score normalization (2nd stage)
        if mode == 'train' and self.compute_zscore:
            input_reward_means = {k: np.mean(v)
                                  for k, v in input_rewards.items()}
            input_reward_stds = {k: np.std(v)
                                 for k, v in input_rewards.items()}
            # not source strings
            idx_means = torch.tensor(input_reward_means['z']).float()
            idx_stds = torch.tensor(input_reward_stds['z']).float()
            rewards_tensor = (rewards_tensor - idx_means)/(idx_stds + 1e-4)
            for i in range(rewards_tensor.size(0)):
                quantities_to_log['resized_reward'].append(
                    rewards_tensor[i].item())
        elif mode == 'infer':  # Optional: Predict Val Prompts
            score = rewards_tensor.mean().item()
            print('Our Prompt:')
            print(prompt_strings, score)

        rewards_log = dict(
            (reward_key, torch.mean(torch.tensor(reward_vals)))
            for reward_key, reward_vals in quantities_to_log.items())

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    # Adapted from
    # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(
            input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def ensure_exactly_one_mask_token(
        self,
        model_inputs: Dict[str, torch.Tensor]
    ) -> None:
        for input_ids in model_inputs["input_ids"]:
            masked_index = self._get_mask_token_index(input_ids)
            numel = np.prod(masked_index.shape)
            assert numel == 1

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
            token_logits = self._generator(**encoded_inputs.to(self.device)).logits
            mask_token_indices = \
                self._get_mask_token_index(encoded_inputs['input_ids'])
            out_logits = token_logits[range(batch_size), mask_token_indices, :]
        else:
            token_logits = self._generator(**encoded_inputs.to(self.device)).logits
            input_lengths = encoded_inputs['attention_mask'].sum(dim=1)
            out_logits = token_logits[range(batch_size), input_lengths - 1, :]

        return out_logits

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self._tokenizer.convert_tokens_to_string(s)
                for s in tokens]

    def _format_prompts(
        self,
        source_strs: List[str],
        prompt_strs: List[str],
    ) -> List[str]:
        return [self.template.format(sentence_1=s_1, prompt=p)
                for s_1, p in zip(source_strs, prompt_strs)]
