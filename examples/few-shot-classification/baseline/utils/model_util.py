import numpy as np
import os
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel

def load_checkpoint(gpt2, checkpoint=None,
                    prompt_tune=False, head_tune=False, transform_tune=False,
                    n_prefix=20,
                    mapping=None):

    def convert_to_single_gpu(state_dict):
        def _convert(key):
            if key.startswith('module.'):
                return key[7:]
            return key
        return {_convert(key):value for key, value in state_dict.items()}

    if checkpoint is not None and not prompt_tune and not head_tune and not transform_tune:
        assert os.path.exists(checkpoint)
        model = GPT2LMHeadModel.from_pretrained(
            gpt2,
            state_dict=convert_to_single_gpu(torch.load(checkpoint)))
        return model

    model = GPT2LMHeadModel.from_pretrained(gpt2)

    if checkpoint is not None:
        assert os.path.exists(checkpoint)
        if prompt_tune:
            set_extra_embeddings(model, n_prefix=n_prefix)
            weight = torch.load(checkpoint)["transformer.wte.new_embed.weight"]
            model.transformer.wte.new_embed._load_from_state_dict(
                {"weight": weight}, "", None, True, [], [], "")

        elif head_tune:
            weight = torch.load(checkpoint)["lm_head.my_lm_head.weight"]
            set_separate_lm_head(model, mapping=mapping)
            model.lm_head.my_lm_head._load_from_state_dict(
                {"weight": weight}, "", None, True, [], [], "")

        elif transform_tune:
            weight = torch.load(checkpoint)["lm_head.transform.weight"]
            set_transformed_lm_head(model)
            model.lm_head.transform._load_from_state_dict(
                {"weight": weight}, "", None, True, [], [], "")

        else:
            raise NotImplementedError()

    return model

def get_dataloader(inputs, batch_size, is_training):

    shape = inputs["input_ids"].shape
    for v in inputs.values():
        assert v.shape==shape

    if "labels" in inputs:
        dataset = TensorDataset(inputs["input_ids"],
                                inputs["attention_mask"],
                                inputs["token_type_ids"],
                                inputs["labels"])

    else:
        dataset = TensorDataset(inputs["input_ids"],
                                inputs["attention_mask"],
                                inputs["token_type_ids"])

    if is_training:
        sampler=RandomSampler(dataset)
    else:
        sampler=SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

def get_optimizer_and_scheduler(optimizer_name,
                                model,
                                learning_rate=1e-5,
                                warmup_proportion=0.01,
                                warmup_steps=50,
                                weight_decay=0.0,
                                adam_epsilon=1e-8,
                                num_training_steps=1000):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    #optimizer_grouped_parameters = [p for n, p in named_parameters]

    if optimizer_name=="adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters,
                              lr=learning_rate,
                              relative_step=False,
                              warmup_init=False)
        scheduler = None
    elif optimizer_name=="adamw":
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=learning_rate,
                          eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_training_steps)
    else:
        raise NotImplementedError()
    return optimizer, scheduler


class MyEmbedding(torch.nn.Module):

    # this is for prompt tuning

    def __init__(self, embed, n_prefix):
        super().__init__()
        self.embed = embed
        self.new_embed = torch.nn.Embedding(n_prefix, embed.embedding_dim)

        # following Lester et al. 2021 in initializing using the top 5000 random vocabs
        indices = np.random.permutation(range(5000))[:n_prefix]
        init_weight = self.embed.state_dict()["weight"][indices]
        self.new_embed._load_from_state_dict({"weight": init_weight},
                                             "", None, True, [], [], "")

    def forward(self, input):
        return F.embedding(
            input,
            torch.cat([self.embed.weight, self.new_embed.weight], 0),
            self.embed.padding_idx,
            self.embed.max_norm,
            self.embed.norm_type,
            self.embed.scale_grad_by_freq,
            self.embed.sparse)

class MyEmbedding2(torch.nn.Module):

    def __init__(self, embed, mapping):
        super().__init__()
        self.my_embed = torch.nn.Embedding(len(mapping), embed.embedding_dim)
        indices = [mapping[i] for i in range(len(mapping))]
        init_weight = embed.state_dict()["weight"][indices]
        self.my_embed._load_from_state_dict({"weight": init_weight},
                                            "", None, True, [], [], "")

    def forward(self, input):
        return self.my_embed(input)


class MyLMHead(torch.nn.Module):

    def __init__(self, lm_head, mapping):
        super().__init__()
        self.my_lm_head = torch.nn.Linear(lm_head.in_features, len(mapping), bias=False)

        indices = [mapping[i] for i in range(len(mapping))]
        init_weight = lm_head.state_dict()["weight"][indices]
        self.my_lm_head._load_from_state_dict({"weight": init_weight},
                                              "", None, True, [], [], "")

    def forward(self, input):
        return self.my_lm_head(input)

class MyLMHeadWithTransform(torch.nn.Module):

    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head
        self.transform = torch.nn.Linear(lm_head.in_features,
                                         lm_head.in_features, bias=False)
        init_weight = torch.eye(lm_head.in_features)
        self.transform._load_from_state_dict({"weight": init_weight},
                                              "", None, True, [], [], "")

    def forward(self, input):
        return self.lm_head(self.transform(input))


def set_extra_embeddings(model, n_prefix):
    model.transformer.set_input_embeddings(
        MyEmbedding(model.transformer.wte, n_prefix))

def set_separate_lm_head(model, mapping):
    model.set_output_embeddings(
        MyLMHead(model.lm_head, mapping))

def set_separate_embeddings(model, mapping):
    model.set_input_embeddings(
        MyEmbedding2(model.transformer.wte, mapping))

def set_transformed_lm_head(model):
    model.set_output_embeddings(
        MyLMHeadWithTransform(model.lm_head))
