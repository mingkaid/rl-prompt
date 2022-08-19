import hydra
import dataclasses
import sys
sys.path.append('../..')
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from fsc_helpers import (PromptedClassificationRewardConfig,
                         FewShotClassificationDatasetConfig,
                         make_prompted_classification_reward,
                         make_few_shot_classification_dataset)
from rlprompt.utils.utils import colorful_print
from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model)
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.trainers import TrainerConfig, make_trainer


# Compose default config
config_fields = []
for config_cls in [PromptedClassificationRewardConfig,
                   FewShotClassificationDatasetConfig, LMAdaptorModelConfig,
                   SinglePromptModelConfig, SQLModuleConfig, TrainerConfig]:
    for config_field in dataclasses.fields(config_cls):
        config_fields.append((config_field.name, config_field.type,
                              config_field))
Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="base_fsc", node=Config)


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')

    (train_dataset, val_dataset, test_dataset,
     num_classes, verbalizers, template) = \
        make_few_shot_classification_dataset(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    reward = make_prompted_classification_reward(num_classes, verbalizers, 
                                                 template, config)
    algo_module = make_sql_module(prompt_model, reward, config)

    # Hack for few-shot classification - Each batch contains all examples
    config.train_batch_size = len(train_dataset)
    config.eval_batch_size = len(val_dataset)
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    trainer.train(config=config)


if __name__ == "__main__":
    main()
