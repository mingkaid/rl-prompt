#!/bin/bash
python run_experiments.py \
translation.task_name="prompt.tst_gpt2_vocab_$2_5token" \
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
translation.prompt_task_lm=$1 \
translation.prompt_dataset=$3 \
translation.prompt_dataset_seed=$4 