# Text Style Transfer with Input-Conditioned Prompts

In this folder, we provide the code for optimizing input-conditioned prompts,
which generally gives better performance. The process for training and 
evaluation is almost the same as optimizing a single prompt, except during
inference, you would need to keep a copy of the learned prompt model to 
generate one prompt for each input. 

## Training Commands
```
python run_input_conditioned_tst.py \
    dataset=[yelp, shakespeare] \
    dataset_seed=[0, 1, 2 (optional, skip for yelp)] \
    direction=[0_to_1, 1_to_0] \
    prompt_length=[any integer (optional, default:5)] \
    task_lm=[distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    lower_outputs=[true, false] \
    random_seed=[any integer (optional)]
```

You can find the saved checkpoints at 
`outputs/[experiment-date]/[experiment-time]`

## Evaluation Commands

To generate the input-conditioned prompts during inference, run the following
command:
```
python infer_input_conditioned_tst_prompt.py \ 
    dataset=[yelp, shakespeare] \
    dataset_seed=[0, 1, 2 (optional, skip for yelp)] \
    direction=[0_to_1, 1_to_0] \
    checkpoint_path=[path/to/checkpoint.pth] \
    output_save_path=[path/for/saving/your/outputs]
```

After that, you can download the evaluation classifiers/language models using
the commands provided in the previous folder, and run the evaluation script 
with the command below:
```
python run_input_conditioned_eval.py \
    dataset=[yelp, shakespeare] \
    dataset_seed=[0, 1, 2 (skip for yelp)] \
    task_lm=[distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    prompts_0_to_1_path=[path/to/0-to-1/prompts (optional)] \
    prompts_1_to_0_path=[path/to/1-to-0/prompts (optional)]
```
The outputs will be saved at `evaluation/outputs/[evaluation-date]/[evaluation-time]`