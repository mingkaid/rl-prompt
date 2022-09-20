gpu_id=0
for seed in 13 21 42 87 100
do
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task sst-2 \
        --template '<s> {sentence} [T] [T] [T] [T] [T] [P] . </s>' \
        --label-map '{"0": ["Ġterrible"], "1": ["Ġgreat"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seed
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task cr \
        --template '<s> {sentence} [T] [T] [T] [T] [T] [P] . </s>' \
        --label-map '{"0": ["Ġterrible"], "1": ["Ġgreat"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seed
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task mr \
        --template '<s> {sentence} [T] [T] [T] [T] [T] [P] . </s>' \
        --label-map '{"0": ["Ġterrible"], "1": ["Ġgreat"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seed
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task yelp-2 \
        --template '<s> {text} [T] [T] [T] [T] [T] [P] . </s>' \
        --label-map '{"0": ["Ġterrible"], "1": ["Ġgreat"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seed
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task sst-5 \
        --template '<s> {sentence} [T] [T] [T] [T] [T] [P] . </s>' \
        --label-map '{"0": ["Ġterrible"], "1": ["Ġbad"], "2": ["Ġokay"], "3": ["Ġgood"], "4": ["Ġgreat"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seed
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task yelp-5 \
        --template '<s> {sentence} [T] [T] [T] [T] [T] [P] . </s>' \
        --label-map '{"0": ["Ġterrible"], "1": ["Ġbad"], "2": ["Ġokay"], "3": ["Ġgood"], "4": ["Ġgreat"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seed
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task agnews \
        --template '<s> [T] [T] [T] [T] [T] [P] {text} . </s>' \
        --label-map '{"0": ["ĠWorld"], "1": ["ĠSports"], "2": ["ĠBusiness"], "3": ["ĠTech"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seed
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task yahoo \
        --template '<s> [T] [T] [T] [T] [T] [P] {sentence} . </s>' \
        --label-map '{"0": ["Ġculture"], "1": ["Ġscience"], "2": ["Ġhealth"], "3": ["Ġeducation"], "4": ["Ġcomputer"], "5": ["Ġsports"], "6": ["Ġbusiness"], "7": ["Ġmusic"], "8": ["Ġfamily"], "9": ["Ġpolitics"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seed
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task dbpedia \
        --template '<s> [T] [T] [T] [T] [T] [P] {sentence} . </s>' \
        --label-map '{"0": ["ĠCompany"], "1": ["ĠEducation"], "2": ["ĠArtist"], "3": ["ĠSports"], "4": ["ĠOffice"], "5": ["ĠTransportation"], "6": ["ĠBuilding"], "7": ["ĠNatural"], "8": ["ĠVillage"], "9": ["ĠAnimal"], "10": ["ĠPlant"], "11": ["ĠAlbum"], "12": ["ĠFilm"], "13": ["ĠWritten"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seeds
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task trec \
        --template '<s> [T] [T] [T] [T] [T] [P] {sentence} . </s>' \
        --label-map '{"0": ["ĠDescription"], "1": ["ĠEntity"], "2": ["ĠExpression"], "3": ["ĠHuman"], "4": ["ĠLocation"], "5": ["ĠNumber"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seeds
    CUDA_VISIBLE_DEVICES=$gpu_id python -m autoprompt.create_trigger \
        --task subj \
        --template '<s> {sentence} [T] [T] [T] [T] [T] [P] . </s>' \
        --label-map '{"0": ["Ġsubjective"], "1": ["Ġobjective"]}' \
        --num-cand 100 \
        --accumulation-steps 1 \
        --bsz 32 \
        --eval-size 16 \
        --iters 5 \
        --model-name roberta-large \
        --seed $seeds
done