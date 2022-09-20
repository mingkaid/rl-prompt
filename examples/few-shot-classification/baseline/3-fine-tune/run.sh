gpu_id=7
for seed in 13 21 42 87 100
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "sst-2" \
        --seed $seed \
        --num_classes 2
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "mr" \
        --seed $seed \
        --num_classes 2
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "cr" \
        --seed $seed \
        --num_classes 2
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "yelp-2" \
        --seed $seed \
        --num_classes 2    
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "sst-5" \
        --seed $seed \
        --num_classes 5 \
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "yelp-5" \
        --seed $seed \
        --num_classes 5 \
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "agnews" \
        --seed $seed \
        --num_classes 4
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "dbpedia" \
        --seed $seed \
        --num_classes 14           
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "yahoo" \
        --seed $seed \
        --num_classes 10
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "trec" \
        --seed $seed \
        --num_classes 6
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --task_name "subj" \
        --seed $seed \
        --num_classes 2
done