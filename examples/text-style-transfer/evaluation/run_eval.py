import sys
import os
sys.path.append('..')  # A hack
import json
import pandas as pd
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from tst_evaluator import TextStyleTransferEvaluator
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)
from tst_modules import PromptedGenerator, TextStyleTransferOutputSelector
from tst_helpers import (TextStyleTransferDatasetConfig,
                         PromptedTextStyleTransferRewardConfig,
                         load_text_style_transfer_test_data,
                         get_style_classifier)
import hydra
import dataclasses
from dataclasses import dataclass


@dataclass
class TextStyleTransferEvaluationConfig:
    prompt_0_to_1: Optional[str] = None
    prompt_1_to_0: Optional[str] = None


# Compose default config
config_list = [TextStyleTransferDatasetConfig, 
               PromptedTextStyleTransferRewardConfig,
               TextStyleTransferEvaluationConfig]
cs = compose_hydra_config_store('base_eval', config_list)


ppl_lm_dict = {'yelp': './ppl/gpt2-yelp',
               'shakespeare': './ppl/gpt2-shakespeare'}
@hydra.main(version_base=None, config_path="./", config_name="eval_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    if config.prompt_0_to_1 is None and config.prompt_1_to_0 is None: 
        raise ValueError('Need to supply at least one prompt')
    output_dir = get_hydra_output_dir()

    device_id = 0
    generator = PromptedGenerator(config.task_lm, config.template, 
                                  config.end_punct, config.pad_token, 
                                  device_id, config.lower_outputs, 
                                  config.control_output_length)
    train_style_classifier = \
        os.path.join('..', get_style_classifier('train', config))
    selector = TextStyleTransferOutputSelector(train_style_classifier, 
                                               config.style_tokenizer, 
                                               config.style_batch_size, 
                                               device_id)

    all_source_texts = []
    all_target_labels = []
    all_ref_texts = []
    all_output_texts = []
    all_rewards = []
    all_contents = []
    all_styles = []
    for direction, prompt in zip(['0_to_1', '1_to_0'],
                                 [config.prompt_0_to_1, config.prompt_1_to_0]):
        if prompt is None:
            continue  # Skip direction if prompt not supplied
        print('Direction:', direction, 'Prompt:', prompt)
        config.direction = direction
        source_texts, target_labels, ref_texts = \
            load_text_style_transfer_test_data(config)
        print('Test Size:', len(source_texts))
        print('Examples:', source_texts[:5])

        top_p = 1.0
        generated_texts = generator.sample_generate_batch(
            prompt, source_texts, config.num_samples, config.task_top_k, top_p)
        output_texts, rewards, contents, styles = selector.select_outputs_batch(
            source_texts, generated_texts, target_labels)

        all_source_texts += source_texts
        all_target_labels += target_labels
        all_ref_texts += ref_texts
        all_output_texts += output_texts
        all_rewards += rewards
        all_contents += contents
        all_styles += styles
    del generator
    del selector

    test_style_classifier = \
        os.path.join('..', get_style_classifier('test', config))
    evaluator = TextStyleTransferEvaluator(test_style_classifier,
                                           ppl_lm_dict[config.dataset])
    (content, style, fluency, joint_score, gm, bleu, bertscore, ppl) = \
        evaluator.evaluate_output(all_source_texts, all_output_texts,
                                  all_target_labels, all_ref_texts)
    del evaluator
    print('Printing Aggregate Scores')
    print('Content:', content, 'Style:', style, 'Fluency:', fluency,
          'Joint:', joint_score, 'GM:', gm, 'BLEU:', bleu,
          'BERTScore:', bertscore, 'PPL:', ppl)

    summary_data = {'content': content, 'style': style, 'fluency': fluency,
                    'joint': joint_score, 'gm': gm, 'bleu': bleu,
                    'bertscore': bertscore, 'ppl': ppl}
    output_data = {'source_text': all_source_texts,
                   'target_label': all_target_labels,
                   'ref_texts': all_ref_texts, 'output_text': all_output_texts, 
                   'reward': all_rewards, 'content': all_contents, 
                   'style': all_styles}
    output_data_df = pd.DataFrame(output_data)
    summary_path = os.path.join(output_dir, 'summary.json')
    output_path = os.path.join(output_dir, 'outputs.tsv')
    json.dump(summary_data, open(summary_path, 'w'))
    output_data_df.to_csv(output_path, index=False, sep='\t')
    print(f'Outputs saved at {output_dir}')


if __name__ == "__main__":
    main()
