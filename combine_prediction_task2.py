import pandas as pd
import numpy as np
from data import TASK2_LABELS

def to_matrix_pred(df3,TASK3_LABELS):    
    df3 = df3.where(pd.notnull(df3), '')
    labels_matrix = np.zeros((df3.shape[0], len(TASK3_LABELS)),int)
    
    for idx, label in enumerate(TASK3_LABELS):
        labels_matrix[:, idx] = df3.labels.apply(lambda x: 1 if label in x else 0)
    return labels_matrix


def multi_pred(langs, modelnames, name):
    
    print(modelnames)
    LABELNAMES = TASK2_LABELS
    folder='task2'
    id2label = {idx:label for idx, label in enumerate(LABELNAMES)}
    for lang in langs:
        lang_files = []
        for modelname in modelnames:
            lang_files.append('{}/{}/pred_{}_{}.csv'.format(folder,lang,lang,modelname))

        df_1 = pd.read_csv(lang_files[0], sep='\t', header=None, names=['id','labels'])

        label_matrix=to_matrix_pred(df_1,LABELNAMES)
        df_1[lang_files[0]] = df_1['labels'] #pd.Series(list(label_matrix))
        
        for filename in lang_files[1:]:
            df_1['labels']=pd.read_csv(filename, sep='\t', header=None, names=['id','labels'])['labels']

           
            label_matrix += to_matrix_pred(df_1,LABELNAMES)
            df_1[filename]=df_1['labels']

        label_matrix = np.round(label_matrix/len(lang_files))
        pred_com=pd.Series(list(label_matrix ))
        series_labels=pred_com.apply(lambda x: [id2label[i] for i in np.where(x==1)[0]])
        df_1['labels_pred']=series_labels.apply(lambda x: ','.join(x))

        output_pred_file= '{}_ensemble_{}_{}.txt'.format(name, len(lang_files),lang)
        print(output_pred_file)
        df_1[['id','labels_pred']].to_csv(output_pred_file, sep='\t', header=None, index=None)
        output_pred_file1= '{}_ensemble_{}_{}_with_all.txt'.format(name, len(lang_files),lang)
        print(output_pred_file1)
        df_1.to_csv(output_pred_file1, sep='\t', index=None)

        
if __name__ == "__main__":
    print('hello')
    modelnames_mul = ['xlm_q020_s40', 'xlm_q030_s41', 'xlm_q040_s42', 'xlm_q050_s43', 'xlm_q060_s44'] 
    multi_pred(['en','ru','it','po','ge','fr'], modelnames_mul, 'xlm')
    
    print('english')
    modelnames = ['rob_q020_s40', 'rob_q030_s41', 'rob_q040_s42', 'rob_q050_s43', 'rob_q060_s44']
    multi_pred(['en'], modelnames, 'rob')
    
    print('english on all 10 model')
    modelnames_com = modelnames + modelnames_mul
    multi_pred(['en'], modelnames_com,'rob_xlm')