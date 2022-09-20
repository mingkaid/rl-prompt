import os
import pdb
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from utils.processors import compute_metrics_mapping

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, required=True, help='data path')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--text_attr', type=str, default='sentence')
parser.add_argument('--label_attr', type=str, default='label')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--max_length', type=int, default=128)
args = parser.parse_args()
print(args)


save_dir = os.path.join('result', args.task_name)
os.makedirs(save_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=args.num_classes)


def preprocess(batch):
    texts = ((batch[args.text_attr],) )
    new_batch = tokenizer(*texts, padding="max_length", truncation=True, max_length=args.max_length)
    new_batch['labels'] = [label_to_id[label] for label in batch[args.label_attr]]
    return new_batch

def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    pred_labels = np.argmax(logits, axis=-1)
    metrics = metrics_fn(args.task_name, np.array(pred_labels), np.array(true_labels))
    return metrics

dataset_dir = f"../../data/16-shot/{args.task_name}/16-{args.seed}/"
dataset = load_dataset(os.path.abspath(dataset_dir))

label_list = list(sorted(set([dataset['train'][i]['label'] for i in range(len(dataset['train']))])))
label_to_id = {v: i for i, v in enumerate(label_list)}
preprocess_dataset = dataset.map(preprocess, batched=True)


training_args = TrainingArguments(
    output_dir=save_dir, 
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    num_train_epochs=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="acc"
)

metrics_fn = compute_metrics_mapping[args.task_name]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=preprocess_dataset['train'],
    eval_dataset=preprocess_dataset['validation'],
    compute_metrics=compute_metrics,
)

trainer.train()
test_pred = trainer.predict(preprocess_dataset["test"])
with open(os.path.join(save_dir, f'result-{seed}.txt'), 'a') as f:
    f.write(f'{test_pred.metrics}\n')

