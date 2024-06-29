#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=8

python3 BGE_M3/eval/eval_msmarco.py \
    --model_savedir saved_models/bgem3_generic_pretraining_20240501/checkpoint-5784 saved_models/bgem3_generic_pretraining_20240503/checkpoint-2579 saved_models/bgem3_generic_pretraining_20240505/checkpoint-4678\
    --query_max_length 128 \
    --query_batch_size 4096 \
    --passage_max_length 256 \
    --passage_batch_size 2048 \
    --data_folder data/eval/msmarco \
    "$@"