import sys
import os
sys.path.append('..')  # A hack
import json
from tqdm import tqdm
import pandas as pd
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)

from tst_modules import PromptedGenerator, TextStyleTransferOutputSelector
from tst_helpers import (TextStyleTransferDatasetConfig,
                         PromptedTextStyleTransferRewardConfig,
                         load_text_style_transfer_test_data,
                         get_style_classifier)
from evaluation.tst_evaluator import TextStyleTransferEvaluator

import hydra
import dataclasses
from dataclasses import dataclass


@dataclass
class TextStyleTransferInputConditionedEvaluationConfig:
    prompts_0_to_1_path: Optional[str] = None
    prompts_1_to_0_path: Optional[str] = None


# Compose default config
config_list = [TextStyleTransferDatasetConfig, 
               PromptedTextStyleTransferRewardConfig,
               TextStyleTransferInputConditionedEvaluationConfig]
cs = compose_hydra_config_store('base_eval', config_list)


ppl_lm_dict = {'yelp': '../evaluation/ppl/gpt2-yelp',
               'shakespeare': '../evaluation/ppl/gpt2-shakespeare'}
@hydra.main(version_base=None, config_path="./", 
            config_name="input_conditioned_eval_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    prompts_supplied = False
    
    if config.prompts_0_to_1_path is not None: 
        prompts_0_to_1 = (open(config.prompts_0_to_1_path, 'r').read().strip()
                          .split('\n'))
        prompts_supplied = True
    else: 
        prompts_0_to_1 = None
        
    if config.prompts_1_to_0_path is not None: 
        prompts_1_to_0 = (open(config.prompts_1_to_0_path, 'r').read().strip()
                          .split('\n'))
        prompts_supplied = True
    else: 
        prompts_1_to_0 = None
        
    if not prompts_supplied: 
        raise ValueError('Need to supply prompts for at least one direction')
    
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
    for direction, prompts in zip(['0_to_1', '1_to_0'],
                                 [prompts_0_to_1, prompts_1_to_0]):
        if prompts is None:
            continue  # Skip direction if prompt not supplied
        print('Direction:', direction, 'Prompt Example:', prompts[0])
        config.direction = direction
        source_texts, target_labels, ref_texts = \
            load_text_style_transfer_test_data(config)
        print('Test Size:', len(source_texts))
        print('Examples:', source_texts[:5])

        top_p = 1.0
        assert len(prompts) == len(source_texts)
        all_generated_texts = []
        for i, (prompt, source_text) in tqdm(enumerate(zip(prompts, 
                                                           source_texts)),
                                             total=len(source_texts)):
            generated_texts = generator.sample_generate(
                prompt, source_text, config.num_samples, 
                config.task_top_k, top_p)
            all_generated_texts.append(generated_texts)
        generated_texts = all_generated_texts
    
        # generated_texts = generator.sample_generate_batch(
        #     prompt, source_texts, config.num_samples, config.task_top_k, top_p)
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
