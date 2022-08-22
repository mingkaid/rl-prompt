# Prompted Few-Shot Classification Example

The script below runs a 16-shot classification experiment, with options for `task_lm` and `dataset`. For each dataset, we provide 5 different 16-shot training sets, toggled by `dataset_seed`.
```
python run_fsc.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews, sst-5, yelp-5] \
    dataset_seed=[0, 1, 2, 3, 4] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    prompt_length=[any integer (optional, default:5)] \
    max_decoding_length=[same integer as prompt_length (optional)] \
    random_seed=[any integer (optional)]
```

## Evaluation

After you train a prompt, you can evaluate it on a given dataset with the following commands
```
cd evaluation
python run_eval.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews, sst-5, yelp-5] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    prompt=[any prompt in string form, e.g. "Absolutely"]
```

For a quick start, you may try the following examples: 

| Model | Prompt | Dataset | Accuracy (%) | 
| ------ | ------- | ------------ | ---------- |
| roberta-large | Absolutely VERY absolute VERY absolute | SST-2 | 92.7 |
| roberta-large | Alert Blog Dialogue Diary Accountability | AG News | 82.0 |
