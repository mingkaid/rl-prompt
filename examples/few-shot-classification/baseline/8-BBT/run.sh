for task in 'sst-2' 'mr' 'cr' 'yelp-2' 'sst-5' 'yelp-5' 'agnews' 'yahoo' 'dbpedia' 'trec' 'subj'
do
    for seed in 13 21 42 87 100
    do
        python bbt.py \
          --task_name $task \
          --n_prompt_tokens 50 \
          --intrinsic_dim 500 \
          --k_shot 16 \
          --device "cuda:0" \
          --seed $seed \
          --loss_type "ce" \
          --cat_or_add "add" \
          --budget 8000 \
          --print_every 50 \
          --eval_every 100
    done
done
