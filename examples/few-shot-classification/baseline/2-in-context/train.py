import sys
sys.path.append("..")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml
import argparse
from argparse import Namespace

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM
)
from utils.model_util import get_optimizer_and_scheduler
from utils.dataset import FewShotDataset
from utils.processors import compute_metrics_mapping

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='data path')
parser.add_argument('--template', type=str, help='Template string')
parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--truncate_head', type=bool, default=True)
parser.add_argument('--skip_u0120', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()


args.task_name = args.task.lower()
args.mapping = args.label_map
args.data_dir = f"../../data/16-shot/{args.task_name}/16-{args.seed}"
args.use_demo = True
args.prompt = True
args.max_seq_length = 512
args.num_sample = 1
args.overwrite_cache = None
args.first_sent_limit = None
args.other_sent_limit = None
print(args)


seed = args.seed 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


device = torch.device(f'cuda:{args.gpu_id}')
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
generator = AutoModelForMaskedLM.from_pretrained('roberta-large').to(device)


for param in generator.parameters():
    param.requires_grad = False

    
testset = FewShotDataset(args, tokenizer=tokenizer, mode="test")
testloader = DataLoader(testset, collate_fn=testset.collate_fn, num_workers=4, pin_memory=True, batch_size=32, shuffle=False)
metrics_fn = compute_metrics_mapping[args.task_name]


pred_labels, true_labels = [], []
for batch in tqdm(testloader):
    labels = torch.empty_like(batch['input_ids']).fill_(-100).long().to(device)
    labels[range(labels.shape[0]), batch['mask_pos'].squeeze()] = batch['labels_tokens'].to(device)

    with torch.no_grad():
        output = generator(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=labels
        )

    loss, logits = output.loss, output.logits
    logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
    pred_labels += logits[:, testset.get_labels_tok()].argmax(1).tolist()
    true_labels += batch['labels'].squeeze().tolist()
    
dev_metrics = metrics_fn(args.task_name, np.array(pred_labels), np.array(true_labels))
dev_metrics = dev_metrics['f1'] if 'f1' in dev_metrics else dev_metrics['acc']
print(f'FewShot Performance: {dev_metrics}')

os.makedirs(os.path.join('result', args.task_name), exist_ok=True)
with open(os.path.join('result', args.task_name, f'result-{seed}.txt'), 'a') as f:
    f.write(f'FewShot Metrics: {dev_metrics}\n')
    