gpu_id=7
for seed in 13
do
    python train.py \
        --task "sst-2" \
        --template "*cls*In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. *sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "mr" \
        --template "*cls*In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. *sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "cr" \
        --template "*cls*In this task, you are given sentences from customer reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. *sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "yelp-2" \
        --template "*cls*In this task, you are given Yelp reviews. The task is to classify a review as \"great\" if the overall sentiment of the review is positive or as \"terrible\" if the overall sentiment of the review is negative. *sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "sst-5" \
        --template "*cls*In this task, you are given sentences from movie reviews. Based on the given review, classify it to one of the five classes: (1) terrible, (2) bad, (3) okay, (4) good, and (5) great. *sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "yelp-5" \
        --template "*cls*In this task, you are given Yelp reviews. Based on the given review, classify it to one of the five classes: (1) terrible, (2) bad, (3) okay, (4) good, and (5) great. *sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id

    python train.py \
        --task "agnews" \
        --template "*cls*In this task, you . In this task is to classify the article to one of the four topics 'World', 'Sports', 'Business', 'Tech'. *mask*_News: *sent_0**sep+*" \
        --label-map "{0:'World',1:'Sports',2:'Business',3:'Tech'}" \
        --seed $seed \
        --truncate_head true \
        --skip_space true \
        --gpu_id $gpu_id

    python train.py \
        --task "yahoo" \
        --template "*cls*You are given a passage. Using the information present in the passage, you need to classify it into one of the 10 topics: 0 - 'Culture', 1 - 'Science', 2 - 'Health', 3 - 'Education', 4 - 'Computers', 5 - 'Sports', 6 - 'Business', 7 - 'Music', 8 - 'Family', 9 - 'Politics'. Topic *mask*: *sent_0**sep+*" \
        --label-map "{0:'culture',1:'science',2:'health',3:'education',4:'computer',5:'sports',6:'business',7:'music',8:'family',9:'politics'}" \
        --seed $seed \
        --truncate_head false \
        --skip_space true \
        --gpu_id $gpu_id \
        
    python train.py \
        --task "dbpedia" \
        --template "*cls*You are given a passage. Using the information present in the passage, you need to classify it into one of the 10 topics: 0 - 'Culture', 1 - 'Science', 2 - 'Health', 3 - 'Education', 4 - 'Computers', 5 - 'Sports', 6 - 'Business', 7 - 'Music', 8 - 'Family', 9 - 'Politics'. [Category:*mask*]*sent_0**sep+*" \
        --label-map "{0:'Company',1:'Education',2:'Artist',3:'Sports',4:'Office',5:'Transportation',6:'Building',7:'Natural',8:'Village',9:'Animal',10:'Plant',11:'Album',12:'Film',13:'Written'}" \
        --seed $seed \
        --truncate_head false \
        --gpu_id $gpu_id \
                
    python train.py \
        --task "trec" \
        --template "*cls*You are given a question. You need to detect which category better describes the question. Answer with \"Description\", \"Entity\", \"Expression\", \"Human\", \"Location\", and \"Number\". *mask*:*+sent_0**sep+*" \
        --label-map "{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}" \
        --seed $seed \
        --truncate_head false \
        --gpu_id $gpu_id \

    python train.py \
        --task "subj" \
        --template  "*cls*In this task, you are given sentences from reviews. The task is to classify a sentence as \"subjective\" if the opinion of the sentence is subjective or as \"objective\" if the opinion of the sentence is objective. *sent_0*_This_is*mask*.*sep+*" \
        --label-map "{0:'subjective',1:'objective'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        
done
