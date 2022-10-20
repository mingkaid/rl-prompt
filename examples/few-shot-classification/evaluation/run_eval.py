import hydra
import sys
sys.path.append("..")
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlprompt.utils.utils import colorful_print
from fsc_helpers import (make_few_shot_classification_dataset,
                         get_dataset_verbalizers)
from fsc_evaluator import PromptedClassificationEvaluator


@hydra.main(version_base=None, config_path="./", config_name="eval_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')

    (train_dataset, val_dataset, test_dataset,
     num_classes, verbalizers, template) = \
        make_few_shot_classification_dataset(config)
    print('Test Size', len(test_dataset))
    print('Examples:', test_dataset[:5])
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=32,
                             drop_last=False)

    is_mask_lm = True if 'bert' in config.task_lm else False
    verbalizers = get_dataset_verbalizers(config.dataset)
    num_classes = len(verbalizers)
    if config.dataset == 'agnews' and is_mask_lm:
        template = "<mask> {prompt} {sentence_1}"
    elif config.dataset == 'dbpedia' and is_mask_lm:
        template = "{prompt} <mask> : {sentence_1}"
    else: 
        template = None
    # Below are some example prompts:
    # Alert Blog Dialogue Diary Accountability (82% for agnews)
    # Absolutely VERY absolute VERY absolute (92% for sst-2)
    tester = PromptedClassificationEvaluator(
        task_lm=config.task_lm,
        is_mask_lm=config.is_mask_lm,
        num_classes=num_classes,
        verbalizers=verbalizers,
        template=template,
        prompt=config.prompt
    )

    acc = tester.forward(test_loader)
    colorful_print(f"prompt: {config.prompt}, accuracy: {acc}", fg='red')


if __name__ == "__main__":
    main()
