import os
import pandas as pd
for MR in ['./16-42/', './16-100/', './16-21/', './16-13/', './16-87/']:
    MR_dev = os.path.join(MR, 'dev.csv')
    MR_train = os.path.join(MR, 'train.csv')
    mr_dev=pd.read_csv(MR_dev)
    mr_train=pd.read_csv(MR_train)
    mr_dev_first = mr_dev.columns.tolist()
    mr_train_first = mr_train.columns.tolist()
    mr_rest_dev = mr_dev.values.tolist()
    mr_rest_train = mr_train.values.tolist()
    mr_rest_dev.append(mr_dev_first)
    mr_rest_train.append(mr_train_first)
    tar = open(MR+'/dev.tsv','w')
    tar_train = open(MR+'/train.tsv','w')
    tar.write('sentence\tlabel\n')
    tar_train.write('sentence\tlabel\n')
    for line in mr_rest_dev:
        label = line[0]
        text = line[1]
        tar.write(text+'\t'+str(label)+'\n')
    tar.close()
    for line in mr_rest_train:
        label = line[0]
        text = line[1]
        tar_train.write(text+'\t'+str(label)+'\n')
    tar_train.close()
