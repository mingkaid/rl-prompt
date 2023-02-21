import os
import sys
sys.path.append('..')
import dataclasses
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import json
from tqdm import tqdm

from transformers import AutoTokenizer

from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.models import (LMAdaptorModelConfig, InputConditionedPromptModelConfig,
                             make_lm_adaptor_model, make_input_conditioned_prompt_model)
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)

from tst_helpers import (PromptedTextStyleTransferRewardConfig,
                         TextStyleTransferDatasetConfig,
                         make_prompted_text_style_transfer_reward,
                         make_text_style_transfer_datasets,
                         get_style_classifier)


@dataclass
class InputConditionedPromptInferenceConfig:
    output_save_path: str = "./output_prompts.txt"


# Compose default config
config_list = [PromptedTextStyleTransferRewardConfig,
               TextStyleTransferDatasetConfig, LMAdaptorModelConfig,
               InputConditionedPromptModelConfig, SQLModuleConfig, 
               InputConditionedPromptInferenceConfig, TrainerConfig]
cs = compose_hydra_config_store('base_tst', config_list)


@hydra.main(version_base=None, config_path="./", 
            config_name="input_conditioned_tst_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    train_dataset, val_dataset, test_dataset = \
        make_text_style_transfer_datasets(config)
    print('Test Size:', len(test_dataset))
    print('Examples:', test_dataset[:5])

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_input_conditioned_prompt_model(policy_model, config)
    # config.style_classifier = get_style_classifier('train', config)
    # reward = make_prompted_text_style_transfer_reward(config)
    reward = None
    algo_module = make_sql_module(prompt_model, reward, config)

    # config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, test_dataset, config)
    
    eval_dataloader = trainer._get_eval_dataloader(test_dataset)
    model = trainer.module.eval()
    hypos = []
    for batch in tqdm(eval_dataloader):
        infer_outputs: Dict[str, Union[torch.Tensor, List[List[str]]]]
        infer_outputs = model.infer(batch)
        hypos += infer_outputs['sample_tokens']
        
    tokenizer = AutoTokenizer.from_pretrained(config.task_lm)
    prompt_strings = [tokenizer.convert_tokens_to_string(t) for t in hypos]
    
    with open(config.output_save_path, 'w') as fw: 
        for s in prompt_strings: 
            fw.write(s + '\n')

if __name__ == "__main__":
    main()
