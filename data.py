import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from transformers import pipeline
#from dataclasses import dataclass, field
#from transformers import HfArgumentParser
#from sklearn.metrics import f1_score, recall_score, precision_score
# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.

TASK3_LABELS_EN = ['Appeal_to_Authority', 'Appeal_to_Fear-Prejudice',
       'Appeal_to_Hypocrisy', 'Appeal_to_Popularity',
       'Causal_Oversimplification', 'Conversation_Killer', 'Doubt',
       'Exaggeration-Minimisation', 'False_Dilemma-No_Choice',
       'Flag_Waving', 'Guilt_by_Association', 'Loaded_Language',
       'Name_Calling-Labeling', 'Obfuscation-Vagueness-Confusion',
       'Red_Herring', 'Repetition', 'Slogans', 'Straw_Man',
       'Whataboutism']
TASK3_LABELS = ['Appeal_to_Authority', 'Appeal_to_Fear-Prejudice',
       'Appeal_to_Hypocrisy', 'Appeal_to_Popularity', 'Appeal_to_Time',
       'Appeal_to_Values', 'Causal_Oversimplification',
       'Consequential_Oversimplification', 'Conversation_Killer', 'Doubt',
       'Exaggeration-Minimisation', 'False_Dilemma-No_Choice', 'Flag_Waving',
       'Guilt_by_Association', 'Loaded_Language', 'Name_Calling-Labeling',
       'Obfuscation-Vagueness-Confusion', 'Questioning_the_Reputation',
       'Red_Herring', 'Repetition', 'Slogans', 'Straw_Man', 'Whataboutism']

TASK2_LABELS = ['Capacity_and_resources', 'Crime_and_punishment', 'Cultural_identity',
       'Economic', 'External_regulation_and_reputation',
       'Fairness_and_equality', 'Health_and_safety',
       'Legality_Constitutionality_and_jurisprudence', 'Morality',
       'Policy_prescription_and_evaluation', 'Political', 'Public_opinion',
       'Quality_of_life', 'Security_and_defense']

def make_dataframe_task2(input_folder, labels_folder=None, target='type'):
    #MAKE TXT DATAFRAME
    text = []
    
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):

        iD, txt = fil[7:].split('.')[0], open(input_folder +fil, 'r', encoding='utf-8').read() 
        text.append((iD, txt))

    df_text = pd.DataFrame(text, columns=['id','text']).set_index('id')

    df = df_text

    #MAKE LABEL DATAFRAME
    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0:'id',1:target})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')

        #JOIN
        df = labels.join(df_text)[['text',target]]

    return df

def make_dataframe_task3(input_folder, labels_fn=None):
    #MAKE TXT DATAFRAME
    text = []
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        fil = fil.strip('._')
        iD = fil[7:].split('.')[0]
        lines = list(enumerate(open(input_folder+fil,'r',encoding='utf-8').read().splitlines(),1))
        text.extend([(iD,) + line for line in lines])

    df_text = pd.DataFrame(text, columns=['id','line','text'])
    df_text.id = df_text.id.apply(int)
    df_text.line = df_text.line.apply(int)
    df_text = df_text[df_text.text.str.strip().str.len() > 0].copy()
    df_text = df_text.set_index(['id','line'])
    
    df = df_text

    if labels_fn:
        #MAKE LABEL DATAFRAME
        labels = pd.read_csv(labels_fn,sep='\t',encoding='utf-8',header=None)
        labels = labels.rename(columns={0:'id',1:'line',2:'labels'})
        labels = labels.set_index(['id','line'])
        labels = labels[labels.labels.notna()].copy()

        #JOIN
        df = labels.join(df_text)[['text','labels']]

    return df

#def load_task3(lang='en'):
#    path_folder = os.getcwd()
#    path_en_tr3=path_folder+'{}/train-articles-subtask-3/'.format(lang)
#    path_en_la3=path_folder+'{}/train-labels-subtask-3.txt'.format(lang)
#    
#    df= _make_dataframe_task3(path_en_tr3,path_en_la3)
   

    #id2label = {idx:label for idx, label in enumerate(task3_labels)}
    #label2id = {label:idx for idx, label in enumerate(task3_labels)}
#    return df

def creat_task3_multi(langs):
    path_folder = os.getcwd()
    print(path_folder)
    df = pd.DataFrame()
    for i in range(len(langs)):
        
        lang=langs[i]
        path_en_tr3=path_folder+'/data/{}/train-articles-subtask-3/'.format(lang)
        path_en_la3=path_folder+'/data/{}/train-labels-subtask-3.txt'.format(lang)
        df_la = make_dataframe_task3(path_en_tr3,path_en_la3)
        df_la['lang']=lang
        if i==0:
            df['text']=df_la['text']
        
            df['labels']=df_la['labels']
            df['lang']=df_la['lang']
        else:
            df=pd.concat([df,df_la[['text','labels','lang']]])
    return df 

def create_task2_multi(langs):
    path_folder = os.getcwd()
    print(path_folder)
    df = pd.DataFrame()
    for i in range(len(langs)):
        lang=langs[i]
        path_en_tr2=path_folder+'/data/{}/train-articles-subtask-2/'.format(lang)
        path_en_la2=path_folder+'/data/{}/train-labels-subtask-2.txt'.format(lang)
        df2 =make_dataframe_task2(path_en_tr2, path_en_la2, target='frames')
        df2 = df2.rename(columns={'frames':'labels'})
        if i==0:
            df['text']=df2['text']
        
            df['labels']=df2['labels']
            
        else:
            df=pd.concat([df,df2[['text','labels']]])
    return df 

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

def _create_samples_multilabel(df: pd.DataFrame, labels_sorted, sample_size: int, seed: int) -> pd.DataFrame:
    """Samples a DataFrame to create an equal number of samples per class (when possible)."""
    print(df.columns)
    examples = []
    #column_labels = [_col for _col in df.columns.tolist() if _col != "text"]
    for label in labels_sorted:
        
        #subset = df.query(f"{label} == 1")
        subset = df[df[str(label)]==1]
        
        if len(subset) > sample_size:
            # tjek howmany already add
            
            if len(examples)>0:
                
                df_intermedia = pd.concat(examples).drop_duplicates()
                already_in = df_intermedia[label].sum()
                
                if already_in < sample_size:
                    sample_yet = sample_size - already_in
                    examples.append(subset.sample(sample_yet, random_state=42, replace=False))
            else:
                
                examples.append(subset)
        else:
            
            examples.append(subset)
    # Dropping duplicates for samples selected multiple times as they have multi labels
    return pd.concat(examples).drop_duplicates()

def _over_sampling(df: pd.DataFrame,task3_labels, q=0.40):
    sum_list=df.drop(columns=['text','labels']).sum()
    treshold= sum_list.quantile(q) #300 #sum_list.median()
    
    print(treshold)
    to_duplicate=sum_list[sum_list<treshold]
    duplicate_list = to_duplicate.index
    print(duplicate_list)
    #print(q)
    #duplicate_list=sum_list.index[0:int(q)]  
    #print(duplicate_list)
    # repeat
    #treshold= sum_list.quantile(0.75) #300 #sum_list.median()
    #to_duplicate=sum_list[sum_list<treshold]
    #duplicate_list.append(to_duplicate.index)
    
    df_copy = df
    df_copy = df.copy(deep=True)
    for label in duplicate_list:
        df_copy =pd.concat([df_copy, df[df[label]==1]])
        
    df_copy = df_copy.reset_index(drop=True)
    
    label_matrix = np.zeros((df_copy.shape[0],len(task3_labels)),int)
    for row in range(df_copy.shape[0]):
        label_matrix[row]=df_copy['labels'][row]
    df_copy = df_copy[['text','labels']]
    
    return df_copy, label_matrix
    
def make_data( df3, task3_labels, sample=None, sample_size=100, seed=42, q=0.40):
      
    #df3_1=df3.loc[indexs] 
    labels_matrix = np.zeros((df3.shape[0], len(task3_labels)),int)
    
    for idx, label in enumerate(task3_labels):
        labels_matrix[:, idx] = df3.labels.apply(lambda x: 1 if label in x else 0)

    df3 = df3.reset_index()    
    df3["label_nr1"]=pd.Series(list(labels_matrix))
    df3_t = df3[['text','label_nr1']].rename(columns={'label_nr1':'labels'})
    print(df3_t.shape)
    #if use_span:
    #    print('adding span')
    #    df3_t = add_span(df3_t,task3_labels)
    #    print(df3_t.shape)
    
    if sample is not None:
        id2label = {idx:label for idx, label in enumerate(task3_labels)}
        labels=[l.replace('-','_') for l in list(id2label.values())]
        df3_t[labels] =pd.DataFrame(df3_t.labels.tolist(), index= df3_t.index)
        
    if sample=='under':
        
        df3_t = df3_t.drop(['labels'], axis=1)
        sorted_indx =np.argsort(labels_matrix.sum(axis=0))
        sorted_labels =np.array(task3_labels)[sorted_indx]
        sorted_labels = [x.replace('-','_') for x in sorted_labels]
        print('into under')
        
        df_sample = _create_samples_multilabel(df3_t, sorted_labels, sample_size, seed)
        print(df_sample.shape)
        labels_matrix_train = labels_matrix[df_sample.index]
        print(labels_matrix_train.sum(axis=0))
        df_train = pd.DataFrame({'text': df_sample['text'].reset_index(drop=True), 'labels': pd.Series(list(labels_matrix_train))})
        
        return df_train, labels_matrix_train
    
    
    if sample=='over':
        df_train, labels_matrix_train = _over_sampling(df3_t,task3_labels,q)
        
        return df_train, labels_matrix_train
        
    return df3_t, labels_matrix



def add_span(df_train,task3_labels):
      # add span
    
    df_span = pd.read_csv('semeval_en_task3_span.csv',  sep='\t')
    labels_under =df_span.label.value_counts()[df_span.label.value_counts()>500].index
   
    df_many =df_span[(df_span.label==labels_under[0]) | (df_span.label==labels_under[1]) | (df_span.label==labels_under[2])]
    
    df_few =df_span[(df_span.label!=labels_under[0]) & (df_span.label!=labels_under[1]) & (df_span.label!=labels_under[2])]
    
    g = df_many.groupby('label')
    df_many_few = g.apply(lambda x: x.sample(300).reset_index(drop=True))
    df_semeval_train2  = pd.concat([df_many_few, df_few])
    
    df_semeval_train2=df_semeval_train2.reset_index(drop=True)
    labels_matrix = np.zeros((df_semeval_train2.shape[0], len(task3_labels)),int)
    
    for idx, label in enumerate(task3_labels):
        labels_matrix[:, idx] = df_semeval_train2.label.apply(lambda x: 1 if label in x else 0)
    
    df_semeval_train2['labels'] =pd.Series(list(labels_matrix))
    #cont7rine here
    df_train = pd.concat([df_train, df_semeval_train2[['text','labels']]])
    
    
    return df_train
    
    
                          