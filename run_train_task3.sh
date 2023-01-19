# python run_multilabel.py --model_name_or_path xlm-roberta-large --model_type xlmroberta --lang mul  --lang_ekstra en2 --pred_lang en --do_train True --do_eval True --do_predict True --output models3/New_ekstra2en_xlm_task3_42_040_1e5 --lr 1e-5 --sample over --q 0.40
 
for pred_lang in fr ge
do
python run_multilabel.py \
--model_name_or_path xlm-roberta-large \
--model_type xlmroberta \
--lang mul \
--lang_ekstra ${pred_lang} \
--pred_lang ${pred_lang} \
--do_train True \
--do_eval True \
--do_predict True \
--output models3/New_xlm_task3_42_040_1e5_ekstra_${pred_lang} \
--lr 1e-5 \
--sample over \
--q 0.40 

done

python run_multilabel.py --model_name_or_path roberta-large --model_type roberta --lang en  --lang_ekstra None --pred_lang en --do_train True --do_eval True --do_predict True --output models3/New_en_rob_task3_42_040_1e5 --lr 1e-5 --sample over --q 0.40