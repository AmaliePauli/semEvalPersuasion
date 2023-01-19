for pred_lang in en it ru po fr ge
do
python run_multilabel_task2.py \
--model_name_or_path xlm-roberta-large \
--model_type xlmroberta \
--lang mul \
--lang_ekstra ${pred_lang} \
--pred_lang ${pred_lang} \
--do_train True \
--do_eval True \
--do_predict True \
--output models3/New_xlm_task2_42_040_1e5_ekstra_${pred_lang} \
--lr 1e-5 \
--sample over \
--q 0.40 

done