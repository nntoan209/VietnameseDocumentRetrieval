#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

python3 -m BGE_M3.src.hn_mine \
    --model_name_or_path BAAI/bge-m3 \
    --input_file data/sft/zalolegal_train_data.jsonl \
    --corpus_file data/eval/zalo_legal/filtered_corpus.json \
    --output_file data/sft/zalolegal_train_data_minedHN.jsonl \
    --use_index \
    --range_for_sampling 10-30 \
    --negative_number 10 \
    "$@"