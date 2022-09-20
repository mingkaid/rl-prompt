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
args.use_demo = False
args.prompt = True
args.max_seq_length = 512
args.overwrite_cache = None
args.first_sent_limit = None
args.other_sent_limit = None
print(args)

epochs = 100
eval_period = 100
global_step = 0
max_grad_norm = 1.0
lr = 1e-5

seed = args.seed 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device(f'cuda:{args.gpu_id}')
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
generator = AutoModelForMaskedLM.from_pretrained('roberta-large').to(device)

trainset = FewShotDataset(args, tokenizer=tokenizer, mode="train")
devset = FewShotDataset(args, tokenizer=tokenizer, mode="dev")
testset = FewShotDataset(args, tokenizer=tokenizer, mode="test")
metrics_fn = compute_metrics_mapping[args.task_name]


best_metrics = -float('inf')
best_loss = float('inf')
os.makedirs(os.path.join('result', args.task_name), exist_ok=True)

trainloader = DataLoader(trainset, collate_fn=trainset.collate_fn, num_workers=4, pin_memory=True, batch_size=16, shuffle=False)
devloader = DataLoader(devset, collate_fn=devset.collate_fn, num_workers=4, pin_memory=True, batch_size=16, shuffle=False)
testloader = DataLoader(testset, collate_fn=testset.collate_fn, num_workers=4, pin_memory=True, batch_size=16, shuffle=False)
optimizer, scheduler = get_optimizer_and_scheduler(
        "adamw",
        generator,
        learning_rate=lr,
        warmup_steps=0,
        num_training_steps=epochs)


for epoch in range(epochs):
    train_loss = []
    for batch in trainloader:
        global_step += 1
        labels = torch.empty_like(batch['input_ids']).fill_(-100).long().to(device)
        labels[range(labels.shape[0]), batch['mask_pos'].squeeze()] = batch['labels_tokens'].to(device)
        
        output = generator(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=labels
        )
        loss, logits = output.loss, output.logits
        loss.backward()
        train_loss.append(loss.item())
        
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
        optimizer.step()   
        generator.zero_grad()
        if scheduler is not None:
            scheduler.step()
            
        if global_step % eval_period == 0:
            generator.eval()
            dev_loss = []
            pred_labels, true_labels = [], []

            for batch in devloader:
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

                dev_loss.append(loss.item())
                pred_labels += logits[:, devset.get_labels_tok()].argmax(1).tolist()
                true_labels += batch['labels'].squeeze().tolist()

            dev_metrics = metrics_fn(args.task_name, np.array(pred_labels), np.array(true_labels))
            dev_metrics = dev_metrics['f1'] if 'f1' in dev_metrics else dev_metrics['acc']
            dev_loss = np.mean(dev_loss)
            generator.train()

            if dev_loss < best_loss or dev_metrics > best_metrics:

                if dev_loss < best_loss:
                    best_loss = dev_loss

                if dev_metrics > best_metrics:
                    best_metrics = dev_metrics

                model_name = f"model-{seed}.pt"
                generator.save_pretrained(os.path.join('result', args.task_name, model_name))

                print(f'Step {global_step} save model with loss {dev_loss} metrics {dev_metrics}')
                with open(os.path.join('result', args.task_name, f'result-{seed}.txt'), 'a') as f:
                    f.write(f'Step: {global_step} | Loss: {dev_loss} | Metrics: {dev_metrics}\n')

            print(f'Step {global_step} train_loss {np.mean(train_loss)}', end='\r')


generator = AutoModelForMaskedLM.from_pretrained(os.path.join('result', args.task_name, model_name)).to(device)
pred_labels, true_labels = [], []
for batch in tqdm(testloader):
    with torch.no_grad():
        logits = generator(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
        ).logits.cpu()

    logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
    pred_labels += logits[:, testset.get_labels_tok()].argmax(1).tolist()
    true_labels += batch['labels'].squeeze().tolist()

metrics = metrics_fn(args.task_name, np.array(pred_labels), np.array(true_labels))
with open(os.path.join('result', args.task_name, f'result-{seed}.txt'), 'a') as f:
    f.write(f'Full Test Metrics: {metrics}\n')

print(f'Finish {args.task_name}-{seed}')
print('*' * 10)
