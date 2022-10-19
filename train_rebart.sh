#!/bin/bash


DATA_DIR="data/"
OUT_DIR="outputs/reorder_exp/macbert"

mkdir -p ${OUT_DIR}
cp $0 ${OUT_DIR}

python -m source.encoder_decoder \
    --train_file ${DATA_DIR}/data_100.jsonl \
    --eval_data_file ${DATA_DIR}/data_100.jsonl \
    --test_path ${DATA_DIR}/data_test.jsonl \
    --out_dir $OUT_DIR \
    --model_type hfl/chinese-macbert-large \
    --model_name_or_path hfl/chinese-macbert-large \
    --device 0 \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --num_train_epochs 1 \
    --logging_steps 3000 \
    --gradient_accumulation_steps 8 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --overwrite_out_dir \
    --max_input_length 1024 \
    --max_output_length 40 \
    --task index_with_sep \
    --overwrite_cache\
    $@
#--overwrite_cache \

