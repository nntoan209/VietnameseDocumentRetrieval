#!/bin/bash

python3 -m BGE_M3.src.split_data \
    --input_path data/sft/train_data_minedHN.jsonl \
    --output_dir data/sft/splitted \
    --cache_dir ./cache \
    --log_name .split_log \
    --length_list 0 512 1024 2048 3072 4096 5120 6144 7168 \
    --model_name_or_path /home/admin_mcn/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/babcf60cae0a1f438d7ade582983d4ba462303c2 \
    --num_proc 64 \
    "$@"