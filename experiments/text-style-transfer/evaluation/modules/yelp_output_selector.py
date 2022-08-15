from transformers import AutoTokenizer, pipeline, GPT2LMHeadModel
import numpy as np
import sacrebleu as scb
from collections import defaultdict 
import torch
import scipy.stats as stats
from tqdm import tqdm
import pandas as pd
from bert_score import BERTScorer
from torch.utils.data import Dataset
import re
import json
from transformers.pipelines.pt_utils import KeyDataset
import fire


class SampleDataset(Dataset):
    def __init__(self, x):
        self.samples = x
        
    def __getitem__(self,index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)
    
    
class YelpOutputSelector(): 
    def __init__(self, 
                 device_id,
                 tst_dataset,
                 tst_data_seed): 
        self.device = device_id
        self.dataset = tst_dataset
        self.seed = tst_data_seed
        
        dataset_clf_basepath = "/emnlp-2022-code/experiments/yelp_sentiment_classifier"
        dataset_clf_tokenizers = {'yelp': 'bert-base-uncased',
                                  'shakespeare': 'bert-base-uncased',}
        dataset_clf_paths = {'yelp': dataset_clf_basepath + "/results-bert-base/checkpoint-10410",
                             'shakespeare': dataset_clf_basepath + f"/shakespeare-bert-base-uncased-train-100-{self.seed}"}
        
        self.classifier = pipeline("sentiment-analysis",
                                   model=dataset_clf_paths[self.dataset],
                                   tokenizer=dataset_clf_tokenizers[self.dataset],
                                   device=self.device)
        
        self.bert_scorer = BERTScorer('roberta-large', 
                                      device=self.device, 
                                      rescale_with_baseline=True, 
                                      lang='en')
        
        
    def sample_generate(self, task_name, sample_size, top_k=None, top_p=None, **kwargs): 
        assert task_name in ['pos2neg', 'neg2pos']
        
        outputs = []
        for sample_id, gen, ref, target_label in self._model_generate(task_name, 
                                                              sample_size, 
                                                              top_k, 
                                                              top_p, 
                                                              **kwargs): 
            output = self._select_output(gen, ref, target_label, sample_id=sample_id)
            outputs.append(output)
        return outputs
    
    def select_output(self,
                      pos2neg_hypos_path,
                      neg2pos_hypos_path): 
        all_hypos = {'pos2neg': ('LABEL_0', [json.loads(row.strip()) for row in open(pos2neg_hypos_path)]),
                     'neg2pos': ('LABEL_1', [json.loads(row.strip()) for row in open(neg2pos_hypos_path)])}
        
        selected_outputs = []
        for task, (target_label, task_hypos) in all_hypos.items():
            print(f'selecting {task}')
            for i, row in tqdm(enumerate(task_hypos), total=len(task_hypos)): 
                output_row = {'target_label': target_label,
                              'src': row['src'],
                              'idx': row['idx']}

                ref_texts = [row['src'] for i, hypo in enumerate(row['hypos']) if len(row['hypos'][i]) > 0]
                hypo_texts = [hypo for i, hypo in enumerate(row['hypos']) if len(row['hypos'][i]) > 0]

                if len(hypo_texts) == 0: 
                    output_row.update({'train_reward': 0,
                                       'selected_hypo': ''})
                    selected_outputs.append(output_row)
                    continue

                bleus = [scb.sentence_bleu(hypothesis=x,
                                           references=[y])
                         for x, y in zip(hypo_texts,
                                         ref_texts)]
                bleu_rewards = [b.score for b in bleus]

                bertscore_f1 = self.bert_scorer.score(hypo_texts, ref_texts)[2]
                bertscore_rewards = [max(b, 0) for b in (bertscore_f1 * 100).tolist()]

                probs = []
                for c in self.classifier(SampleDataset(hypo_texts), batch_size=32, truncation=True): 
                    prob = ((c['label'] == target_label) * c['score'] + \
                            (c['label'] != target_label) * (1 - c['score']))
                    probs.append(prob * 100)

                recon_weight = 1
                style_weight = 1
                sum_rewards = [(recon_weight * r + style_weight * p) / (recon_weight + style_weight) \
                               for r, p in zip(bertscore_rewards, probs)]
                max_sum_reward = torch.tensor(sum_rewards).float().max()
                output_row['train_reward'] = round(float(max_sum_reward), 1)

                top_index = sum_rewards.index(max_sum_reward)
                output_row['selected_hypo'] = hypo_texts[top_index]
                output_row['self_bertscore'] = round(float(bertscore_rewards[top_index]), 1)
                output_row['self_bleu'] = round(float(bleu_rewards[top_index]), 1)
                output_row['train_style'] = round(float(probs[top_index]), 1)
                selected_outputs.append(output_row)
            
        return selected_outputs
    
    def select_and_save(self,
                        pos2neg_hypos_path,
                        neg2pos_hypos_path,
                        save_path): 
        selected_outputs = self.select_output(pos2neg_hypos_path,
                                              neg2pos_hypos_path)
        
        with open(save_path, 'w') as fw: 
            for row in selected_outputs: 
                fw.write(json.dumps(row) + '\n')
                
if __name__ == '__main__':
    fire.Fire(YelpOutputSelector)