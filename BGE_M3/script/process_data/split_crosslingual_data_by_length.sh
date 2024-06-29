#!/bin/bash

python3 -m BGE_M3.src.split_data_crosslingual \
    --input_path data/cross_lingual/merged_train_data_ids.jsonl \
    --corpus_path data/cross_lingual/merged_corpus_vi.json \
    --output_dir data/cross_lingual/merged_train_data_splitted \
    --cache_dir ./cache \
    --log_name .split_log \
    --length_list 0 512 1024 2048 \
    --model_name_or_path BAAI/bge-m3 \
    --num_proc 16 \
    "$@"