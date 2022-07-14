#!/bin/bash
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
translation.reward_shaping_max=5 \
translation.top_k=256 \
translation.random_seed=$4 \
translation.learning_rate=5e-5 \
translation.target_learning_rate=1e-3  \
translation.reward_name="prompted-classification" \
translation.prompt_task_lm=$1 \
translation.prompt_dataset=$2 \
translation.prompt_dataset_seed=$3
