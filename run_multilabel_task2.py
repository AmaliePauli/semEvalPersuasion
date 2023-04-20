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

from data import make_data, create_task2_multi, make_dataframe_task2, TASK2_LABELS

SEED= 42
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='roberta-large', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
       default='roberta'
    )
    do_train: bool = field(
        default=True
    )
    do_eval: bool = field(
        default=True
    )
    do_predict: bool=field(
        default=False
    )
    use_span: bool=field(
        default=False
    )
        
    output: str = field(
        default='output'
    )
    sample: str = field(
         default=None
    )
    sample_size: int = field(
        default=100
    )
    q: float = field(
        default=0.4
    )
    lr: float = field(
        default=4e-5
    )
    train_seed: int =field(
        default=42
    )
    

@dataclass
class DataTrainingArguments:
    lang: str=field(
        default='en'
    )
    pred_lang: str=field(
        default='en'
    )
    lang_ekstra: str=field(
        default=None
    )     
    
        
    

def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments)) #, MultiLabelClassificationArgs))
    #model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args = parser.parse_args_into_dataclasses()

    if data_args.lang=='mul':
        lang=['en','it','ge','ru','fr','po']
    else: lang = [data_args.lang]
    
    df3 = create_task2_multi(lang)
  
    task2_labels=TASK2_LABELS
    print('lenght of labels ' + str(len(task2_labels)) )
    
    if model_args.do_eval:
            df3_train_split = df3.sample(frac=0.85, random_state=SEED)
            df3_val_split = df3.drop(df3_train_split.index)
            df_val, val_labels_matrix = make_data(df3_val_split,task2_labels)
            print('shape of val matrix ')
            print(val_labels_matrix.shape)
    else: df3_train_split = df3
    
    
        # add to df_train_split
    if data_args.lang_ekstra is not None:
        if data_args.lang_ekstra=='en':
            ekstra = ['en']
        else:
            ekstra = ['en', data_args.lang_ekstra]
        
        df_ekstra =create_task2_multi(ekstra)
        df3_train_split = pd.concat([df3_train_split,df_ekstra])
        
    
    
    df_train, provematrix = make_data(df3_train_split,task2_labels, sample=model_args.sample, sample_size=model_args.sample_size, q=model_args.q)
    print('shape of train matix')
    print(provematrix.shape)
    #df_train,_ = make_data(df3,task3_labels, sample=over)
    #df_train1, _ = add_span(df_train, task3_labels)
    #print(df_train.columns)
    #print(df_train[1:5])
    #if model_args.use_span:
    #    df_train = add_span(df_train, task3_labels)
    #    print(df_train.columns)
    #    print(df_train[-5:-1])
    id2label = {idx:label for idx, label in enumerate(task2_labels)}


    # Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(
        model_args.model_type,
        model_args.model_name_or_path,
        num_labels=len(task2_labels),
        args={
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "evaluate_during:training": True,
        "use_early_stopping": True,
        "num_train_epochs": 10,
        "max_seq_length": 512,
        "train_batch_size": 8,
        "learning_rate": model_args.lr,
        "output_dir": model_args.output,
        "use_cuda": True,
        "seed": model_args.train_seed, #np.random.randint(1000, size=1)[0],
        "cuda_device": 'cuda:0'
    },
    )
    

    # You can set class weights by using the optional weight argument
    #print(df_train.head())
    if model_args.do_train:
        # Train the model
        model.train_model(df_train, eval_df=df_val)

    # Evaluate the model
    if model_args.do_eval:
        
        
        result, model_outputs, wrong_predictions = model.eval_model(df_val)
        y_pred = np.round_(model_outputs).astype(int)
        print('shap of eval pred matrix')
        print(y_pred.shape)
        print(result)
        #print(model_outputs)
        micro_str ='f1-score_micro: {}'.format(f1_score(val_labels_matrix,y_pred,average='micro'))
        macro_str ='f1-score_macro: {}'.format(f1_score(val_labels_matrix,y_pred,average='macro'))
        print(micro_str)
        print(macro_str)
        scores=f1_score(val_labels_matrix,y_pred,average=None)
        
        dict_label={label:score for label, score in zip(task2_labels,scores)}
        print(dict_label)
        with open(model_args.output+'/results2.txt', 'w') as f:
            f.write(micro_str + '\n')
            f.write(macro_str+ '\n')
            for item in dict_label.items():
                f.write(str(item) + '\n')
    
    
    if model_args.do_predict:
        print('ready to predict')
        print(data_args.pred_lang)
        # load x_dev without labels 
        path_folder = os.getcwd()
        dev_folder = path_folder + '/data/{}/dev-articles-subtask-2/'.format(data_args.pred_lang)
        df_dev = make_dataframe_task2(dev_folder)
        df_dev=df_dev.reset_index()

        # predict
        y_pred, model_outputs = model.predict(list(df_dev['text']))
        y_pred = np.round_(model_outputs).astype(int)

        series_pred=pd.Series(list(y_pred))  
        series_labels=series_pred.apply(lambda x: [id2label[i] for i in np.where(x==1)[0]])
            
        df_dev['labels_pred']=series_labels.apply(lambda x: ','.join(x))
        
        if not os.path.exists('task2'):
               os.makedirs('task2')
        if not os.path.exists('task2/'+str(data_args.pred_lang)):
               os.makedirs('task2/'+str(data_args.pred_lang))
        
        output_pred_file = 'task2/{}/DEV_noSamp_xlm.csv'.format(data_args.pred_lang)
        
        print(output_pred_file)
        # pred_ru_models/xlm_512_mul_e10.csv

        df_dev[['id','labels_pred']].to_csv(output_pred_file, sep='\t', header=None, index=None)
    
        
if __name__ == "__main__":
    print('hello')
    main()
    
    
   