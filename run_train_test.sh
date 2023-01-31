
 
for pred_lang in en it po ru fr ge
do
python run_multilabel_test.py \
--model_name_or_path xlm-roberta-large \
--model_type xlmroberta \
--lang mul \
--lang_ekstra ${pred_lang} \
--pred_lang ${pred_lang} \
--do_train True \
--do_eval True \
--do_predict True \
--output models3/Final_xlm_${pred_lang} \
--lr 1e-5 \
--sample over \
--q 0.40 

done

python run_multilabel_test.py --model_name_or_path roberta-large --model_type roberta --lang en --pred_lang en --do_train True --do_eval True --do_predict True --output models3/Final_rob_en --lr 1e-5 --sample over --q 0.40

# pred new lang
for pred_lang in es gr ka
do
python run_multilabel_test.py \
--model_name_or_path models3/Final_xlm_en \
--model_type xlmroberta \
--lang mul \
--pred_lang ${pred_lang} \
--do_train False \
--do_eval False \
--do_predict True \
--output models3/Test_xlm_out \
--lr 1e-5 \
--sample over \
--q 0.40 

done

# task2
python run_multilabel_task2_test.py \
--model_name_or_path xlm-roberta-large \
--model_type xlmroberta \
--lang mul \
--pred_lang en \
--do_train True \
--do_eval True \
--do_predict False \
--output models2/Test_xlm \
--lr 4e-5 \
--sample over \
--q 0.40 

# predict task 2 model
for pred_lang in en it po ru fr ge gr es ka
do
python run_multilabel_task2_test.py \
--model_name_or_path models2/Test_xlm \
--model_type xlmroberta \
--lang mul \
--pred_lang ${pred_lang} \
--do_train False \
--do_eval False \
--do_predict True \
--output models2/Test_xlm_out \
--lr 4e-5 \
--sample over \
--q 0.40 

done

python run_multilabel_test.py \
--model_name_or_path models3/Final_xlm_en \
--model_type xlmroberta \
--lang mul \
--pred_lang en \
--do_train False \
--do_eval False \
--do_predict True \
--output models3/Test_xlm_out \
--lr 1e-5 \
--sample over \
--q 0.40 

#python run_multilabel_test.py --model_name_or_path Final_rob_en --model_type roberta --lang en --pred_lang en --do_train False --do_eval #False --do_predict True --output models3/Test_rob_en --lr 1e-5 --sample over --q 0.40