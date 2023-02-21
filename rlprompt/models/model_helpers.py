from dataclasses import dataclass
from typing import Optional
from rlprompt.models import BaseModel, LMAdaptorModel, SinglePromptModel, InputConditionedPromptModel

def make_lm_adaptor_model(config: "DictConfig") -> LMAdaptorModel:
    return LMAdaptorModel(config.policy_lm,
                          config.hidden_size,
                          config.logit_bias,
                          config.fluent,
                          config.fluent_top_k,
                          config.max_decoding_length,
                          config.eos_token_id)


def make_single_prompt_model(model: BaseModel,
                             config: "DictConfig") -> SinglePromptModel:
    return SinglePromptModel(model,
                             config.prompt_length,
                             config.prompt_train_batch_size,
                             config.prompt_infer_batch_size,
                             config.source_str)


def make_input_conditioned_prompt_model(model: BaseModel,
                                        config: "DictConfig") -> InputConditionedPromptModel:
    return InputConditionedPromptModel(model,
                                       config.prompt_length,
                                       config.source_train_reps,
                                       config.source_infer_reps)


@dataclass
class LMAdaptorModelConfig:
    policy_lm: str = "distilgpt2"
    # Name of the backbone pretrained LM
    hidden_size: int = 2048
    # Dimension for the hidden state of the enclosed adaptor MLP
    logit_bias: float = 0.0
    # Added to all prompt token logits. Set negative value to encourage exploration.
    fluent: bool = False
    # if True, constrain tokens to be from those with top-k probability under
    # a GPT-2 model
    fluent_top_k: int = 20
    # k for top-k probability above
    max_decoding_length: int = 5
    # Max output token length for the model
    eos_token_id: Optional[int] = None
    # The end-of-sentence token id, set to None for fixed-length prompts


@dataclass
class SinglePromptModelConfig:
    prompt_length: int = 5
    prompt_train_batch_size: int = 8
    prompt_infer_batch_size: int = 8
    source_str: str = "<|endoftext|>"

    
@dataclass
class InputConditionedPromptModelConfig:
    prompt_length: int = 5
    source_train_reps: int = 1
    source_infer_reps: int = 1
