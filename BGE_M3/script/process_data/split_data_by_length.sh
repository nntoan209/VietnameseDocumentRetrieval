#!/bin/bash

python3 -m BGE_M3.src.split_data \
    --input_path data/vi/train.jsonl \
    --output_dir data/vi/synthetic \
    --cache_dir ./cache \
    --log_name .split_log \
    --length_list 0 512 1024 2048 \
    --model_name_or_path BAAI/bge-m3 \
    --num_proc 32 \
    "$@"