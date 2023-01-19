for pred_lang in en ru po it fr ge
do
for path in models2/xlm_q020_s40 models2/xlm_q030_s41 models2/xlm_q040_s42 models2/xlm_q050_s43 models2/xlm_q060_s44
do
    python run_multilabel_task2.py \
    --model_name_or_path ${path}\
    --model_type xlmroberta \
    --lang mul \
    --pred_lang  ${pred_lang} \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --output models2/pred \
    --sample over

done
done

for path in models2/rob_q020_s40 models2/rob_q030_s41 models2/rob_q040_s42 models2/rob_q050_s43 models2/rob_q060_s44
do
    python run_multilabel_task2.py \
    --model_name_or_path ${path}\
    --model_type roberta \
    --lang en \
    --pred_lang en \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --output models2/pred \
    --sample over

done