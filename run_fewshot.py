import argparse
import copy
import json
import math
import os
import pathlib
import sys
import pandas as pd
from shutil import copyfile
from typing import Dict
from warnings import simplefilter
import numpy as np
from datasets import Dataset
from evaluate import load
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from torch.utils.data import DataLoader
from typing_extensions import LiteralString

from setfit.modeling import SetFitBaseModel, SKLearnWrapper, sentence_pairs_generation_multilabel
from setfit.utils import  load_data_splits_multilabel
from setfit import SetFitModel

from data import make_dataframe_task3, creat_task3_multi, make_data, TASK3_LABELS, TASK3_LABELS_EN, add_span

LOSS_NAME_TO_CLASS = {
    "CosineSimilarityLoss": losses.CosineSimilarityLoss,
    "TripletLoss": losses.TripletLoss
}
SEED=42
TEMPLATE = "This argument is "

IDX_TEMPLATES = ['appealing to authority', 'appealing to the emotions of fear or prejudice', 'appealing hypocrisy', 'appealing to popularity','making an causal oversimplification','an conversaion killer', 'cating doubt on sombodys reputations', 'exaggeration or minimising the facts', 'presenting a false dilemma', 'promoting a country', 'casting guilt on sombody by association', 'using loaded language','name calling or labeling persons', 'vague or confussing', 'introducing irrelavant informations', 'using repetition','containing a slogan','misrepresentation of someone’s position','switching topic to distract' ]

IDX_TEMPLATES_MUL = ['appealing to authority', 'appealing to the emotions of fear or prejudice', 'appealing hypocrisy', 'appealing to popularity','appealing to time','appealing to values','making an causal oversimplification','making an consequential oversimplification','an conversaion killer', 'cating doubt on sombodys reputation', 'exaggeration or minimising the facts', 'presenting a false dilemma', 'promoting a country', 'casting guilt on sombody by association', 'using loaded language','name calling or labeling persons', 'vague or confussing', 'questioning sombodys reputation', 'introducing irrelavant informations', 'using repetition','containing a slogan','misrepresentation of someone’s position','switching topic to distract' ]




# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--sample_sizes", type=int, default=100)
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--num_itaration", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--classifier",
        default="logistic_regression"
    )
    parser.add_argument(
        "--multi_target_strategy",
        default="one-vs-rest",
        choices=["one-vs-rest", "multi-output", "classifier-chain"],
    )
    parser.add_argument("--loss", default="TripletLoss")
    parser.add_argument("--margin", default=0.5, type=float)
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--add_normalization_layer", default=False, action="store_true")
    parser.add_argument("--optimizer_name", default="AdamW")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--override_results", default=False, action="store_true")
    parser.add_argument("--lang", default='en')
    parser.add_argument("--pred_lang", default='en')
    parser.add_argument("--do_eval", type=str, default='True')
    parser.add_argument("--do_inference", type=str, default='False')
    parser.add_argument("--do_train", type=str, default='True')
    parser.add_argument("--device", default='cuda:1')
    parser.add_argument("--run_nr", type=int, default=0)
    args = parser.parse_args()

    return args

class SetFitBaseModel:
    def __init__(self,  model, max_seq_length: int, add_normalization_layer: bool) -> None:
        self.model = SentenceTransformer(model, device=DEVICE)
        self.model_original_state = copy.deepcopy(self.model.state_dict())
        self.model.max_seq_length = max_seq_length

        if add_normalization_layer:
            self.model._modules["2"] = models.Normalize()

#def sentence_generates_achor_multiple(sentences, labels, input_pair):
#train_examples = [InputExample(texts=['Anchor 1 eks', 'Positive 1', 'Negative 1']),
#    
##    for first_idx in range(len(sentences)):
#        current_sentence = sentences[first_idx]
#        sample_labels = np.where(labels[first_idx, :] == 1)[0]
#        if len(sample_labels)!=0:
#            for _lab in sample_labels:
#                # get the anchor
#                anchor = IDX_TEMPLATES[_lab]
#                # find the negativ (a sentence with label zero in the _lab index)
#                # a vecotr describing current label
#                _lab_vec=np.zeros(labels.shape[1])
#                _lab_vec[_lab]=1
#                negative_idx = np.where(labels.dot(labels[first_idx, :].T) == 0)[0]
#                negative_sentence = sentences[np.random.choice(negative_idx)]
#                # input pair
#                #train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
#                #InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])]
#                input_pair.append(InputExample(texts=[anchor, current_sentence, negative_sentence]))
#                
#    return input_pair

def sentence_generates_achor_multiple(sentences, labels, TEMPLATE, LABELS_NAMES, input_pair):
#train_examples = [InputExample(texts=['Anchor plus', 'Positive 1', 'Negative 1']),

    
    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        sample_labels = np.where(labels[first_idx, :] == 1)[0]
        if len(sample_labels)!=0:
           
            # get the multilabel anchor
            label_names = ', '.join(LABELS_NAMES[_lab] for _lab in sample_labels)
            anchor = TEMPLATE + label_names
            
            negative_idx = np.where(labels.dot(labels[first_idx, :].T) == 0)[0]
            
            negative_sentence = sentences[np.random.choice(negative_idx)]
            # input pair
            #train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
            input_pair.append(InputExample(texts=[anchor, current_sentence, negative_sentence]))
                
    return input_pair

class RunFewShot:
    def __init__(self, args: argparse.Namespace) -> None:
        # Prepare directory for results
        self.args = args

        parent_directory = pathlib.Path(__file__).parent.absolute()
        self.output_path = (
            parent_directory
            / "model_anchor"
            / "task3"
            / f"{args.model.replace('/', '-')}-{args.loss}-{args.classifier}-itaration_{args.num_itaration}-batch_{args.batch_size}-{args.exp_name}--{args.sample_sizes}-margin_{args.margin}_use_span".rstrip(
                "-"
            )
        )
        os.makedirs(self.output_path, exist_ok=True)

        # Save a copy of this training script and the run command in results directory
        train_script_path = os.path.join(self.output_path, "train_script.py")
        copyfile(__file__, train_script_path)
        with open(train_script_path, "a") as f_out:
            f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

        #self.dataset_to_metric = {dataset: "accuracy" for dataset in args.datasets}

        # Configure loss function
        self.loss_class = LOSS_NAME_TO_CLASS[args.loss]
        
        # Load SetFit Model
        self.model_wrapper = SetFitBaseModel(
            self.args.model, max_seq_length=args.max_seq_length, add_normalization_layer=args.add_normalization_layer
        )
        self.model = self.model_wrapper.model

    def get_classifier(self, sbert_model: SentenceTransformer) -> SKLearnWrapper:
        if self.args.classifier == "logistic_regression":
            if self.args.multi_target_strategy == "one-vs-rest":
                multilabel_classifier = OneVsRestClassifier(LogisticRegression())
            elif self.args.multi_target_strategy == "multi-output":
                multilabel_classifier = MultiOutputClassifier(LogisticRegression())
            elif self.args.multi_target_strategy == "classifier-chain":
                multilabel_classifier = ClassifierChain(LogisticRegression())
            return SKLearnWrapper(sbert_model, multilabel_classifier)

   

    def train(self, df_train, y_train, nr=0) -> SKLearnWrapper:
        "Trains a SetFit model on the given few-shot training data."
        print('train started')
        self.model.load_state_dict(copy.deepcopy(self.model_wrapper.model_original_state))

        x_train = df_train["text"].values  
        
        if self.loss_class is None:
            return

        # sentence-transformers adaptation
        batch_size = self.args.batch_size
        
        ## Add to loos
        if self.args.loss=='TripletLoss':
            train_loss = self.loss_class(
                model=self.model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=self.args.margin,
                        )
            train_examples = []
            for _ in range(self.args.num_itaration):
                # cahnges how to generate input pairs to fit achor
                
                train_examples = sentence_generates_achor_multiple(np.array(x_train), y_train, TEMPLATE, IDX_TEMPLATES, train_examples)
                
        
            
        else: 
            train_loss = self.loss_class(self.model)
            train_examples = []
            for _ in range(self.args.num_itaration):
                train_examples = sentence_pairs_generation_multilabel(np.array(x_train), np.array(y_train), train_examples)
    
        
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_steps = len(train_dataloader)
        

        print(f"{len(x_train)} train samples in total, {train_steps} train steps with batch size {batch_size}")

        warmup_steps = math.ceil(train_steps * 0.1)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            steps_per_epoch=train_steps,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
        )
        
        #save model
        
        
        # Train the final classifier
        classifier = self.get_classifier(self.model)
        classifier.fit(x_train, y_train)
        
        # save
        print('saving model and clf')
        #self.model.save(self.output_path)
        clf_name = 'classifier_' + str(nr) 
        clf_path = os.path.join(self.output_path, clf_name)
        classifier.save(clf_path)
        old_name = os.path.join(clf_path, 'setfit_head.pkl')
        new_name = os.path.join(clf_path, 'model_head.pkl')
        os.rename(old_name, new_name)
        
        return classifier

    def eval(self, classifier: SKLearnWrapper, df_val, y_test, metric: str, task3_labels, nr) -> dict:
        """Computes the metrics for a given classifier."""
        # Define metrics
        metric_fn = load(metric, config_name="multilabel")

        x_test = df_val["text"]
        

        y_pred = classifier.predict(x_test)
        
        weF1 = metric_fn.compute(predictions=y_pred, references=y_test, average='macro')
        print('macro F1: ')
        print(weF1['f1'])
        microF1 = metric_fn.compute(predictions=y_pred, references=y_test, average='micro')
        print('microF1: ')
        print(microF1['f1'])
        f1_ind = metric_fn.compute(predictions=y_pred, references=y_test, average=None)
        
        dict_label={label:score for label, score in zip(task3_labels,f1_ind['f1'])}
        print(dict_label)
        results = str(nr)+ '_results2.txt'
        results_path = os.path.join(self.output_path, results)
        with open(results_path, "w") as f:
            f.write('macro f1: ' + str(weF1['f1']) + '\n')
            f.write('micro f1: ' + str(microF1['f1']) + '\n')
            for item in dict_label.items():
                f.write(str(item) + '\n')
        return dict_label


    def full_train_sem3(self)  -> None:
        """full training on sem task 3 dataset"""
        # load data
        #path_folder= 'C:\\Users\\au458226\\OneDrive - Aarhus Universitet\\Dokumenter\\projekter\\semEval\\semeval_taskPRE\\data\\'
        path_folder=os.getcwd()
        
        
        seed=42
        
        
        # prepare data training 
        if self.args.lang=='mul':
            lang=['en','en','it','ge','ru','fr','po']
        else: lang = [self.args.lang]
        
        #load data
        df3 = creat_task3_multi(lang)
        
        #num_labels = len(task3_labels_)
        if self.args.lang=='en':
            task3_labels=TASK3_LABELS_EN
            under_sample_list = [30] #[15,25,35,45,55,65,75]
           
        else:
            task3_labels=TASK3_LABELS
            under_sample_list = [145,155,165,175,185,195,205]

        if self.args.do_eval=='True':
            df3_train_split = df3.sample(frac=0.85, random_state=SEED)
            df3_val_split = df3.drop(df3_train_split.index)
            df_val, val_labels_matrix = make_data(df3_val_split,task3_labels)
            
        else: df3_train_split = df3
        
        

        for i in under_sample_list:
            # Train the model on the current train split
            df_train, labels_matrix = make_data(df3_train_split, task3_labels, sample=self.args.sample, sample_size=i)
    
              
            classifier = self.train(df_train, labels_matrix, i)

            
            if self.args.do_eval=='True':  
                metrics = self.eval(classifier,df_val, val_labels_matrix, 'f1', task3_labels, i)
        #clf_path = os.path.join(self.output_path, 'classifier')
        #classifier.save(clf_path)
        #with open(results_path, "w") as f_out:
        #    json.dump({"score": metrics[metric] * 100, "measure": metric}, f_out, sort_keys=True)
        
    def full_predict(self):
        # load trained classifier and predict on eval set 
        print('hej')
                # load data
        #path_folder= 'C:\\Users\\au458226\\OneDrive - Aarhus Universitet\\Dokumenter\\projekter\\semEval\\semeval_taskPRE\\data\\'
        path_folder=os.getcwd()
        
        # prepare data training 
        if self.args.lang=='mul':
            lang=['en','en','it','ge','ru','fr','po']
        else: lang = [self.args.lang]
        
        #load data
        df3 = creat_task3_multi(lang)
        
        #num_labels = len(task3_labels_)
        if self.args.lang=='en':
            task3_labels=TASK3_LABELS_EN
           
        else:
            task3_labels=TASK3_LABELS

        if self.args.do_eval=='True':
            df3_train_split = df3.sample(frac=0.85, random_state=SEED)
            df3_val_split = df3.drop(df3_train_split.index)
        else: df3_train_split = df3
        
        
        df_val, val_labels_matrix = make_data(df3_val_split,task3_labels)
        
        clf_path = os.path.join(self.output_path, 'classifier')
        print(clf_path)
        classifier = SetFitModel.from_pretrained(clf_path)
        
        
        metrics = self.eval(classifier,df_val, val_labels_matrix, 'f1', task3_labels)
        
        
    def full_predict_task3(self):
        # load trained classifier and predict on eval set 
        print('hej')
                # load data
        #path_folder= 'C:\\Users\\au458226\\OneDrive - Aarhus Universitet\\Dokumenter\\projekter\\semEval\\semeval_taskPRE\\data\\'
        path_folder=os.getcwd()
        
        # prepare data training 
        if self.args.lang=='mul':
            lang=['en','en','it','ge','ru','fr','po']
        else: lang = [self.args.lang]
        
        #load data
        df3 = creat_task3_multi(lang)
        
        #num_labels = len(task3_labels_)
        if self.args.lang=='en':
            task3_labels=TASK3_LABELS_EN
           
        else:
            task3_labels=TASK3_LABELS
        id2label = {idx:label for idx, label in enumerate(task3_labels)}
        # load x_dev without labels 
        path_folder = os.getcwd()
        dev_folder = path_folder + '/data/{}/dev-articles-subtask-3/'.format(self.args.pred_lang)
        df_dev =  make_dataframe_task3(dev_folder)
        df_dev=df_dev.reset_index()
        df_all = df_dev.copy(deep=True)
        
        clfs = [clf for clf in os.listdir(self.output_path) if clf.startswith('classifier')]
        for clf in clfs:
            clf_path = os.path.join(self.output_path, clf)
            print(clf_path)
            classifier = SetFitModel.from_pretrained(clf_path)
            y_pred = classifier.predict(df_dev['text'])
            df_all[str(clf)] = pd.Series(list(y_pred)).apply(lambda x: [id2label[i] for i in np.where(x==1)[0]])
            df_all[str(clf)] = df_all[str(clf)].apply(lambda x: ','.join(x))
            
            df_dev[str(clf)] = pd.Series(list(y_pred))
            
        file_name = 'large_anchor_task3_all_pred_{}.csv'.format(self.args.pred_lang)
        df_dev_path = os.path.join(self.output_path, file_name)
        df_all.to_csv(df_dev_path, sep='\t')
        
        # majority vote
        df_dev['maj'] = df_dev[df_dev.columns[3:]].sum(axis=1)/len(df_dev.columns[3:])
        df_dev['maj'] = df_dev['maj'].apply(lambda x: np.round(x))
        
        series_pred=pd.Series(list(y_pred))  
        series_labels=series_pred.apply(lambda x: [id2label[i] for i in np.where(x==1)[0]])
            
        df_dev['labels_pred']=series_labels.apply(lambda x: ','.join(x))
        maj_file= 'majority_pred_{}.csv'.format(self.args.pred_lang)
        
        output_pred_file  = os.path.join(self.output_path, maj_file)
        
        df_dev[['id','line','labels_pred']].to_csv(output_pred_file, sep='\t', header=None, index=None)
        
def main():
    args = parse_args()
    global DEVICE
    DEVICE = args.device
    if args.lang=='mul':
        global IDX_TEMPLATES
        IDX_TEMPLATES=IDX_TEMPLATES_MUL
    run_fewshot = RunFewShot(args)
    print(args.do_train)
    if args.do_train=='True':
        print('into trian')
        run_fewshot.full_train_sem3()
        
    if args.do_inference=='True':
        print('into predict')
        run_fewshot.full_predict_task3()
    


if __name__ == "__main__":
    main()