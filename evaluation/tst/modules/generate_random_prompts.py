import os
import sys
sql_path = '/emnlp-2022-code/'
sys.path.insert(0, sql_path)
import torch.nn.functional as F
from glob import glob
from tqdm import trange
from experiments.translation import *
from experiments.translation import _modify_model_config
import omegaconf
from sql.utils import ForwardMode
from transformers import AutoTokenizer
import fire

def generate_prompts(task, ckpt_path_short, device_id, tst_dataset, tst_data_seed): 
    assert task in ['neg2pos', 'pos2neg']
    config = omegaconf.OmegaConf.load(os.path.join(sql_path, "configs/config.yaml"))
    config = config.translation
    if task == 'pos2neg': 
        config.task_name = "prompt_tst.yelp_gpt2_vocab_negative"
    elif task == 'neg2pos': 
        config.task_name = "prompt_tst.yelp_gpt2_vocab_positive"
    config.architecture = "gpt2_conditioned_mlp"
    config.reward_name = None
#     config.checkpoint_path = os.path.join(sql_path, 'outputs/',
#                                           ckpt_path_short) if len(ckpt_path_short) > 0 else None
    config.tst_dataset = tst_dataset
    config.tst_data_seed = tst_data_seed
    config.checkpoint_path = os.path.join(sql_path, 'outputs/', ckpt_path_short)
    
    device = torch.device(f'cuda:{device_id}')
    # Modify the model config
    _modify_model_config(config)
    # Choosing task dynamically
    config_data: Any = importlib.import_module(
        f"configs.data.{config.task_name}")
    # Prepare data, model, and ops
    train_data, val_data, test_data, data_iterator = prepare_data(
        config_data=config_data, device=device)
    model = prepare_model(
        config=config,
        train_data=train_data,
        max_source_length=getattr(
            config_data,
            "max_source_length",
            config_data.max_decoding_length),
        max_decoding_length=config_data.max_decoding_length,
        device=device_id)
    
    model._model._tst_inputs[('infer', 'LABEL_0')] = model._model._tst_inputs[('test', 'LABEL_0')]
    model._model._tst_inputs[('infer', 'LABEL_1')] = model._model._tst_inputs[('test', 'LABEL_1')]
    print(model._model.fluent)
    print(len(model._model._tst_inputs[('infer', 'LABEL_0')]),
          len(model._model._tst_inputs[('infer', 'LABEL_1')]))
    
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    
    data_iterator.switch_to_test_data()
    all_prompts = []
    for step, batch in enumerate(data_iterator):
        infer_outputs, _ = model(
            mode=ForwardMode.INFER,
            batch=batch)
        output_ids = infer_outputs["sample_id"][:, :, 0].cpu()

        # Here `target_texts` are list of string of all references
        source_texts = [
            " ".join(text) for text in
            tx.utils.strip_special_tokens(
                batch["source_text"],
                is_token_list=True)]
        # target_texts = _get_list_of_targets_from_batch(batch)
        output_texts = tx.data.vocabulary.map_ids_to_strs(
            ids=output_ids, vocab=val_data.target_vocab)

        prompts = [tokenizer.convert_tokens_to_string(text.split()) for text in output_texts]
        all_prompts += prompts
        
    if task == 'pos2neg': 
        cutoff = len(model._model._tst_inputs[('test', 'LABEL_0')])
        print(cutoff)
    elif task == 'neg2pos': 
        cutoff = len(model._model._tst_inputs[('test', 'LABEL_1')])
        print(cutoff)
    del model
    
    return all_prompts[:cutoff]

def main(task, ckpt_path_short, device_id, output_path, tst_dataset, tst_data_seed): 
    all_prompts = generate_prompts(task, ckpt_path_short, device_id, tst_dataset, tst_data_seed)
    with open(output_path, 'w') as fw: 
        for prompt in all_prompts: 
            fw.write(prompt + '\n')

if __name__ == '__main__':
    fire.Fire(main)