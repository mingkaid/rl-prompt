import os
import dataclasses
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.models import (LMAdaptorModelConfig, InputConditionedPromptModelConfig,
                             make_lm_adaptor_model, make_input_conditioned_prompt_model)
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)

import sys
sys.path.append('..')
from tst_helpers import (PromptedTextStyleTransferRewardConfig,
                         TextStyleTransferDatasetConfig,
                         make_prompted_text_style_transfer_reward,
                         make_text_style_transfer_datasets,
                         get_style_classifier)


# Compose default config
config_list = [PromptedTextStyleTransferRewardConfig,
                TextStyleTransferDatasetConfig, LMAdaptorModelConfig,
                InputConditionedPromptModelConfig, SQLModuleConfig, TrainerConfig]
cs = compose_hydra_config_store('base_tst', config_list)


@hydra.main(version_base=None, config_path="./", 
            config_name="input_conditioned_tst_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    train_dataset, val_dataset, test_dataset = \
        make_text_style_transfer_datasets(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_input_conditioned_prompt_model(policy_model, config)
    config.style_classifier = get_style_classifier('train', config)
    config.style_classifier = os.path.join('..', config.style_classifier)
    reward = make_prompted_text_style_transfer_reward(config)
    algo_module = make_sql_module(prompt_model, reward, config)

    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    trainer.train(config=config)


if __name__ == "__main__":
    main()
