# RLPrompt: Optimizing Discrete Text Prompts With Reinforcement Learning

This repo contains the code for reproducing the RL-based discrete prompt optimization results described in [this paper](https://arxiv.org/abs/2205.12548). 

We will keep updating the codebase to be easier to use and repurpose for your own tasks, so please stay tuned by starring or watching our repo! 

## Getting Started

* Extensive [recent work](https://arxiv.org/abs/2107.13586) has shown that *prompting* pre-trained LMs with specific text can steer them to perform various NLP tasks, without needing to update the model
* Previous work has typically tuned soft prompts with gradient-based optimization or searched for discrete text prompts using various heuristics
* In our paper, we propose to formulate discrete prompt optimization as an RL problem, and train a policy network to generate the prompt that optimizes a reward function
* Compared to typical soft prompts, our discrete prompts are lightweight, interpretable, and transferrable across model types (e.g., RoBERTa to GPT-2) and sizes (e.g., small to large)
* Check out more analyses at our paper [here](https://arxiv.org/abs/2205.12548)

![](figure.png)

## Setting Up (Minimal)

To run the codebase, we provide a minimal set of required configurations: 
* Python >= 3.7
* PyTorch >= 1.10.1 (install from the [official website](https://pytorch.org/get-started/locally/))

Install other libraries by
```
pip install -r requirements-minimal.txt
```

## Setting Up (Reproducing Results)

To exactly reproduce our environment, use
* Python == 3.8
* PyTorch == 1.10.1

Install the full dependencies by 
```
pip install -r requirements-full.txt
```

## Usage

Our policy network is in `modules/models.py`, and we combine prompts with LMs to compute the reward function in `sql/rewards.py`

Below are the commands we use to run the experiments for 1) few-shot text classification and 2) text style transfer, as described in our paper.

### Few-Shot Text Classification
The command below runs a 16-shot classification experiment. You can toggle the dataset by the options for `prompt_dataset`

For each dataset, we provide 5 different 16-shot training sets, toggled by `prompt_dataset_seed`
```
python run_experiments.py \
translation.task_name="prompt.classification_gpt2_vocab_16shot_5token" \
translation.architecture="gpt2_conditioned_mlp" \
translation.save_dir=./outputs \
translation.num_epochs=1200 \
translation.training_mode="sql-onpolicy" \
translation.num_batches_per_epoch=10 \
translation.save_frequency=2 \
translation.reward_old_min=0 \
translation.reward_old_max=1 \
translation.reward_shaping_min=0 \
translation.reward_shaping_max=3 \
translation.top_k=256 \
translation.random_seed=2 \
translation.learning_rate=5e-5 \
translation.target_learning_rate=1e-3  \
translation.reward_name="prompted-classification" \
translation.prompt_task_lm=[distilroberta-base,roberta-large] \
translation.prompt_dataset=[SST-2,yelp-2,mr,cr,agnews,sst-5,yelp-5] \
translation.prompt_dataset_seed=[0,1,2,3,4]
```

### Text Style Transfer
We experiment on Yelp for sentiment and Shakespeare for authorship transfer, respectively. 
For Shakespeare, we test for [few-shot text style transfer](https://arxiv.org/abs/2010.03802) by only training on 100 examples per style. 
Like for few-shot classification, we provide 3 different training sets and their corresponding style classifiers.

You can download the training style classifiers by running the script below
```
python download_tst_classifiers.py --model_name [yelp_train,
                                                 shakespeare_train_100_0,
                                                 shakespeare_train_100_1,
                                                 shakespeare_train_100_2]
```

#### Yelp
`tst_gpt2_vocab_positive_5token` is for negative-to-positive transfer, and `tst_gpt2_vocab_negative_5token` is for positive-to-negative.
```
python run_experiments.py \
translation.task_name="prompt.tst_gpt2_vocab_[positive,negative]_5token" \
translation.architecture="gpt2_conditioned_mlp" \
translation.save_dir=./outputs \
translation.num_epochs=240 \
translation.training_mode="sql-onpolicy" \
translation.num_batches_per_epoch=50 \
translation.save_frequency=2 \
translation.reward_old_min=0 \
translation.reward_old_max=1 \
translation.reward_shaping_min=-20 \
translation.reward_shaping_max=80 \
translation.top_k=50 \
translation.reward_name="prompted-text-style-transfer" \
translation.random_seed=2 \
translation.learning_rate=[1e-4,5e-5] \
translation.target_learning_rate=1e-3  \
translation.mlp_input_specific=true \
translation.mlp_logit_bias=-10 \
translation.prompt_task_lm=[distilgpt2,gpt2,gpt2-medium,gpt2-large,gpt2-xl] \
translation.prompt_dataset='yelp' 
```

#### Shakespeare (100-Shot)
`tst_gpt2_vocab_positive_5token` is for old-to-modern transfer, and `tst_gpt2_vocab_negative_5token` is for modern-to-old

Select `prompt_dataset_seed` to toggle the training set
```
python run_experiments.py \
translation.task_name="prompt.tst_gpt2_vocab_positive_5token" \
translation.architecture="gpt2_conditioned_mlp" \
translation.save_dir=./outputs \
translation.num_epochs=240 \
translation.training_mode="sql-onpolicy" \
translation.num_batches_per_epoch=50 \
translation.save_frequency=2 \
translation.reward_old_min=0 \
translation.reward_old_max=1 \
translation.reward_shaping_min=-20 \
translation.reward_shaping_max=80 \
translation.top_k=50 \
translation.reward_name="prompted-text-style-transfer" \
translation.random_seed=2 \
translation.learning_rate=5e-5 \
translation.target_learning_rate=1e-3  \
translation.mlp_input_specific=true \
translation.mlp_logit_bias=-10 \
translation.prompt_task_lm=gpt2-xl \
translation.prompt_dataset='shakespeare' \
translation.prompt_dataset_seed=[0,1,2]
```

#### Evaluation
You can download the evaluation style classifiers as below
```
python download_tst_classifiers.py --model_name [yelp_test,shakespeare_test_all]
```
And the language models for perplexity as below
```
python download_ppl_lms.py --model_name [yelp,shakespeare]
```
And use the evaluation scripts in `evaluation/tst/modules` in the order below
```
generate_prompts.py -> prompted_gpt2.py -> yelp_output_selector.py -> yelp_evaluator.py
```
We will provide more directions for this part soon. Please reach out if you have any questions while using it. 