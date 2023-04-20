import pandas as pd
import os
from simpletransformers.classification import MultiLabelClassificationModel
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import MultiLabelClassificationArgs
from sklearn.metrics import f1_score, recall_score, precision_score
# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
from transformers import pipeline
from data import make_data, creat_task3_multi, make_dataframe_task3, add_appeals, TASK3_LABELS, TASK3_LABELS_EN

def add_appeals(df3):
    
    classifier_cred = pipeline("text-classification",model='appeal_model/final_ethos_s1_std', top_k=1, truncation=True)
    pred_cred = pd.Series(classifier_cred(list(df3['text'])))
    pred_cred = pred_cred.apply(lambda x: x[0].get('label'))
    df3['pred_cred']=pd.Series(list(pred_cred), index=df3.index)
    print(df3['pred_cred'].sum())

    classifier_emo = pipeline("text-classification",model='appeal_model/final_pathos_s1_std', top_k=1, truncation=True)
    pred_emo = pd.Series(classifier_emo(list(df3['text'])))
    pred_emo = pred_emo.apply(lambda x: x[0].get('label'))
    df3['pred_emo']=pd.Series(list(pred_emo), index=df3.index)
    print(df3['pred_emo'].sum())

    classifier_log = pipeline("text-classification",model='appeal_model/final_logos_s1_std', top_k=1, truncation=True)
    pred_log = pd.Series(classifier_log(list(df3['text'])))
    pred_log = pred_log.apply(lambda x: x[0].get('label'))
    df3['pred_log']=pd.Series(list(pred_log), index=df3.index)
    print(df3['pred_log'].sum())


    df3['pred_emoN'] = df3['pred_emo'].apply(lambda x: 'emotional appeal ' if x==1 else '')
    df3['pred_credN'] = df3['pred_cred'].apply(lambda x: 'credibility attack ' if x==1 else '')
    df3['pred_logN'] = df3['pred_log'].apply(lambda x: 'logical fallacy ' if x==1 else '')

    df3['text'] = df3.apply(lambda x: x.pred_emoN + x.pred_credN +x.pred_logN + ':' + x.text, axis=1)
    
        
    return df3


def main():
    
    
    df3 = creat_task3_multi(['en'])
    #print(df3.iloc[0].text)
    #print(classifier_cred(df3.iloc[0].text)[0][0].get('label'))
    #print(classifier_cred('he is like hitler')[0][0].get('label'))
    df3 = add_appeals(df3)
    df3.to_csv('train_appeals.csv', sep='\t')
    
    
    path_folder = os.getcwd()
    dev_folder = path_folder + '/data/{}/dev-articles-subtask-3/'.format('en')
    df_dev = make_dataframe_task3(dev_folder)

    df_dev = add_appeals(df_dev)
    df_dev.to_csv('dev_appeals.csv')
    
    
if __name__ == "__main__":
    print('hello')
    main()