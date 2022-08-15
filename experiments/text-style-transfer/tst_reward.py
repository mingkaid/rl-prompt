import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from typing import List, Tuple, Union, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer
from bert_score import BERTScorer
from collections import defaultdict

from rlprompt.rewards import BaseReward

# Magic numbers and variables
DEFAULT_TEMPLATE = '{prompt} "{sentence_1}" "'
DEFAULT_END_PUNCT = '"'
SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                 'gpt2-large', 'gpt2-xl']
DEFAULT_PAD_TOKEN = '<|endoftext|>'
DEFAULT_PAD_TOKEN_ID = 50256
DEFAULT_TOP_K = 10
DEFAULT_STYLE_BATCH_SIZE = 32


class PromptedTextStyleTransferReward(BaseReward):
    def __init__(
        self,
        task_lm: str,
        style_classifier_path: str,
        num_repeats: int,  # Num of repetitions for each example
        num_samples: int,  # Num of samples from which to take the output
        num_bootstraps: int,  # Num of bootstraps to reduce reward randomness
        compute_zscore: bool,  # Whether to compute z-score of rewards
        lower_outputs: bool  # Whether to convert all outputs to lower case
    ):
        generator_device = 0  # TODO
        reward_device = 0  # TODO

        # Loading generator model
        assert task_lm in SUPPORTED_LMS
        generator_model = task_lm
        print('Task LM:', generator_model)
        tokenizer = AutoTokenizer.from_pretrained(generator_model,
                                                  pad_token=DEFAULT_PAD_TOKEN)
        self._generator = pipeline("text-generation",
                                   model=generator_model,
                                   tokenizer=tokenizer,
                                   device=generator_device)
        self.num_samples = num_samples
        self.num_bootstraps = num_bootstraps
        self.top_k = DEFAULT_TOP_K  # TODO
        self.pad_token_id = DEFAULT_PAD_TOKEN_ID  # TODO

        # Loading reward models
        self._style_classifier = pipeline("sentiment-analysis",
                                          model=style_classifier_path,
                                          tokenizer='bert-base-uncased', # TODO
                                          device=reward_device)
        self.style_batch_size = DEFAULT_STYLE_BATCH_SIZE  # TODO
        # We use BERTScore which computes the equivalent of the CTC
        # content/semantic preservation metric
        self._bert_scorer = BERTScorer('roberta-large',
                                       device=reward_device,
                                       rescale_with_baseline=True,
                                       lang='en')

        # Misc. generation params
        self._generator_template = DEFAULT_TEMPLATE  # TODO: Allow options
        self._generator_end_punct = DEFAULT_END_PUNCT  # TODO: Allow options
        self._lower_outputs = lower_outputs

        # Misc. training details
        self.num_repeats = num_repeats
        self._counter = 0
        self.tokens_explored = set()
        self.compute_zscore = compute_zscore

    def forward(
        self,
        source_texts: List[str],
        target_labels: List[str],
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError
        if mode == 'train':
            self._counter += 1
            source_strs = self._repeat_texts(source_texts)
            target_labels = self._repeat_texts(target_labels)
        elif mode == "infer":
            source_strs = source_texts

        prompt_tokens = output_tokens
        self.tokens_explored = \
            self.tokens_explored.union(*[set(p) for p in prompt_tokens])
        prompt_strs = self._convert_tokens_to_string(prompt_tokens)
        assert len(prompt_strs) == len(source_strs)
        formatted_prompts = self._format_prompts(source_strs, prompt_strs)

        n_reward = self.num_samples
        k_reward = self.num_bootstraps
        N = n_reward * k_reward
        X = MyDataset(formatted_prompts)
        generator_outputs: List[List[Dict[str, Any]]] = \
            self._generator(X,
                            # max_length=60, # Default max_length is 50
                            pad_token_id=self.pad_token_id,
                            top_k=self.top_k,
                            num_return_sequences=N,
                            # Only return generated text, without the prompt
                            return_full_text=False)

        rewards: List[torch.Tensor] = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for batch_index, outputs in enumerate(generator_outputs):
            generated_texts = []
            for output in outputs:
                text = output["generated_text"]
                generated_texts.append(self._postprocess_output(text))

            # Content/semantic preservation
            input_texts = [source_strs[batch_index] for _ in generated_texts]
            content_rewards = \
                self._compute_content_preservation_rewards(input_texts,
                                                           generated_texts)
            content_score = torch.tensor(content_rewards).float().mean()
            quantities_to_log['mean_content'].append(content_score)

            # Style probability:
            labels = [target_labels[batch_index] for _ in generated_texts]
            style_probs = self._compute_style_prob_rewards(generated_texts,
                                                           labels)
            prob = torch.tensor(style_probs).float().mean()
            quantities_to_log['mean_style'].append(prob)

            # Sum content and style rewards together
            sum_rewards = [(c + 100 * p) / 2
                           for c, p in zip(content_rewards, style_probs)]

            # Bootstrap the max reward for k times and average
            bootstrap_max_rewards: List[float] = \
                self._boostrap_max_rewards_k_times(sum_rewards, k_reward)
            # Average boostrap max rewards as the final reward
            reward = torch.Tensor(bootstrap_max_rewards).float().mean()

            # Keep track of each input's max rewards to compute z-score
            input_text = source_strs[batch_index]
            input_rewards[input_text] += bootstrap_max_rewards

            # Take the max of the sub-list rewards as the one to
            # print as an output example
            max_reward = max(bootstrap_max_rewards)
            top_index = sum_rewards.index(max_reward)
            top_content = torch.tensor(content_rewards[top_index]).float()
            top_prob = torch.tensor(style_probs[top_index]).float()

            # Log relevant quantities
            quantities_to_log["sum_reward"].append(reward)
            mean_reward = torch.tensor(sum_rewards).float().mean()
            quantities_to_log["mean_reward"].append(mean_reward)
            quantities_to_log["top_content"].append(top_content)
            quantities_to_log["top_style"].append(top_prob)

            print(self._counter, '|',
                  prompt_tokens[batch_index], '|',
                  formatted_prompts[batch_index], '|',
                  generated_texts[top_index], '|',
                  'Top Content:', round(content_rewards[top_index], 2), '|',
                  'Top Style:',
                  round(100 * style_probs[top_index], 2), '|',
                  'Top Reward:', round(sum_rewards[top_index], 2), '|',
                  'Reward:', round(reward.item(), 2))
            rewards.append(reward)

        rewards_tensor = torch.stack(rewards)

        if mode == "train" and self.compute_zscore:
            input_max_reward_means = {k: np.mean(v)
                                      for k, v in input_rewards.items()}
            input_max_reward_stds = {k: np.std(v)
                                     for k, v in input_rewards.items()}
            idx_means = torch.tensor([input_max_reward_means[s]
                                     for s in source_strs]).float()
            idx_stds = torch.tensor([input_max_reward_stds[s]
                                    for s in source_strs]).float()
            print(idx_means)
            print(idx_stds)

            rewards_tensor = (rewards_tensor - idx_means) / (idx_stds + 1e-4)

        quantities_to_log["num_tokens_explored"].append(
            torch.tensor(len(self.tokens_explored)).float())
        quantities_to_log["reward_var"].append(
            torch.var(torch.tensor(rewards)))
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    def _boostrap_max_rewards_k_times(
        self,
        rewards: List[float],
        k: int
    ) -> List[float]:
        # Segment list rewards into k equal sub-lists
        l = len(rewards)
        assert l % k == 0, f'l={l}, k={k}'
        segmented_rewards = [rewards[i*l//k:(i+1)*l//k]
                             for i in range(k)]  # [k, l/k]
        # We use different rewards for each bootstrap for now
        bootstrap_rewards = segmented_rewards

        # For each sub-list, take the max as the sub-reward
        values, indices = (torch.tensor(bootstrap_rewards)
                           .float().max(axis=1))
        # Take numbers from the original list to avoid numerical issues
        bootstrap_max_rewards = [bootstrap_rewards[i][index]
                                 for i, index in enumerate(indices)]

        return bootstrap_max_rewards

    def _compute_content_preservation_rewards(
        self,
        input_texts: List[str],
        output_texts: List[str],
    ) -> List[float]:
        assert len(input_texts) == len(output_texts)
        # We use BERTScore F1 to compute the equivalent of CTC metric
        ctc_scores = self._bert_scorer.score(output_texts,
                                             input_texts)[2]
        content_rewards = [max(s, 0) for s in (ctc_scores * 100).tolist()]
        return content_rewards

    def _compute_style_prob_rewards(
        self,
        texts: List[str],
        labels: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        if batch_size is None:
            batch_size = self.style_batch_size
        X = MyDataset(texts)
        style_probs = []
        for i, c in enumerate(self._style_classifier(X, batch_size=batch_size,
                                                     truncation=True)):
            label = labels[i]
            prob = ((c['label'] == label) * c['score']
                    + (c['label'] != label) * (1 - c['score']))
            style_probs.append(prob)
        return style_probs

    def _repeat_texts(
        self,
        texts: List[str],
        num_repeats: Optional[int] = None
    ) -> List[str]:
        if num_repeats is None:
            num_repeats = self.num_repeats
        return list(itertools.chain(*[[s for _ in range(num_repeats)]
                                      for s in texts]))

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self._generator.tokenizer.convert_tokens_to_string(s)
                for s in tokens]

    def _format_prompts(
        self,
        source_strs: List[str],
        prompt_strs: List[str],
        template: Optional[str] = None,
    ) -> List[str]:
        if template is None:
            template = self._generator_template

        return [template.format(sentence_1=s_1, prompt=p)
                for s_1, p in zip(source_strs, prompt_strs)]

    def _postprocess_output(
        self,
        text: List[str],
        end_punct: Optional[str] = None,
    ) -> List[str]:
        if end_punct is None:
            end_punct = self._generator_end_punct

        try:
            end = text.index(end_punct)
        except ValueError:
            end = len(text)
        text = text[:end].strip()

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
        if self._lower_outputs:
            text = text.lower()

        return text


class MyDataset(Dataset):
    def __init__(self, x):
        self.samples = x

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
