import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline
from bert_score import BERTScorer
from typing import Tuple, List, Union

class TextStyleTransferOutputSelector: 
    def __init__(
        self, 
        style_classifier: str,
        style_tokenizer: str,
        style_batch_size: int,
        device_id: int
    ): 
        self.device = device_id
        self.style_classifier = pipeline("sentiment-analysis",
                                         model=style_classifier,
                                         tokenizer=style_tokenizer,
                                         device=self.device)
        self.style_batch_size = style_batch_size
        self.bert_scorer = BERTScorer('roberta-large', 
                                      device=self.device, 
                                      rescale_with_baseline=True, 
                                      lang='en')

    def compute_sample_rewards(
        self, 
        source_text: str, 
        generated_texts: List[str], 
        target_label: str
    ) -> Tuple[List[float]]: 
        srcs = [source_text for _ in generated_texts]
        hypos = generated_texts

        # Content preservation reward
        ctc_scores = self.bert_scorer.score(hypos, srcs)[2]
        content_rewards = [max(s, 0) * 100 for s in ctc_scores.tolist()]

        # Style probablility reward
        hypo_dataset = ListDataset(hypos)
        batch_size = self.style_batch_size
        style_rewards = []
        for i, c in enumerate(self.style_classifier(hypo_dataset,
                                                    batch_size=batch_size, 
                                                    truncation=True)):
            prob = ((c['label'] == target_label) * c['score']
                    + (c['label'] != target_label) * (1 - c['score']))
            style_rewards.append(prob * 100)
        sum_rewards = [(c + s) / 2 \
                       for c, s in zip(content_rewards, style_rewards)]

        return sum_rewards, content_rewards, style_rewards
        
    def select_outputs_batch(
        self, 
        source_texts: List[str], 
        generated_texts: List[List[str]], 
        target_labels: List[str]
    ) -> Tuple[Union[List[str], List[float]]]:
        output_texts = []
        output_rewards = []
        output_contents = []
        output_styles = []
        for src, hypos, label in tqdm(zip(source_texts, generated_texts, 
                                          target_labels),
                                      total=len(source_texts)):
            hypos = [h for h in hypos if len(h) > 0]
            rewards, contents, styles = self.compute_sample_rewards(src, hypos, 
                                                                    label)
            max_reward = torch.tensor(rewards).float().max()
            top_index = rewards.index(max_reward)

            output_texts.append(hypos[top_index])
            output_rewards.append(rewards[top_index])
            output_contents.append(contents[top_index])
            output_styles.append(styles[top_index])
            
        return output_texts, output_rewards, output_contents, output_styles


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __getitem__(self, index):
        return self.data_list[index]
    
    def __len__(self):
        return len(self.data_list)