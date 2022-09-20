gpu_id=7
for seed in 13
do
    python train.py \
        --task "sst-2" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "mr" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "cr" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "yelp-2" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "sst-5" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "yelp-5" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "agnews" \
        --template "*cls**mask*_News: *sent_0**sep+*" \
        --label-map "{0:'World',1:'Sports',2:'Business',3:'Tech'}" \
        --seed $seed \
        --truncate_head true \
        --skip_u0120 true \
        --gpu_id $gpu_id

    python train.py \
        --task "yahoo" \
        --template "*cls*Topic *mask*: *sent_0**sep+*" \
        --label-map "{0:'culture',1:'science',2:'health',3:'education',4:'computer',5:'sports',6:'business',7:'music',8:'family',9:'politics'}" \
        --seed $seed \
        --truncate_head false \
        --skip_u0120 true \
        --gpu_id $gpu_id \
        
    python train.py \
        --task "dbpedia" \
        --template "*cls*[Category:*mask*]*sent_0**sep+*" \
        --label-map "{0:'Company',1:'Education',2:'Artist',3:'Sports',4:'Office',5:'Transportation',6:'Building',7:'Natural',8:'Village',9:'Animal',10:'Plant',11:'Album',12:'Film',13:'Written'}" \
        --seed $seed \
        --truncate_head false \
        --gpu_id $gpu_id \
        
    python train.py \
        --task "trec" \
        --template "*cls**mask*:*+sent_0**sep+*" \
        --label-map "{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}" \
        --seed $seed \
        --truncate_head false \
        --gpu_id $gpu_id \

    python train.py \
        --task "subj" \
        --template "*cls**sent_0*_This_is*mask*.*sep+*" \
        --label-map "{0:'subjective',1:'objective'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        
done
