#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="M3 Reranker"
export HF_DATASETS_CACHE="./.cache"
PORT_ID=$(expr $RANDOM + 1000)
SEED=$(expr $RANDOM)
# RUN_NAME="m3_reranker_bce_$(date '+%Y%m%d')"
RUN_NAME="m3_reranker_bce_20240718"

torchrun --nproc_per_node 1 --master_port $PORT_ID \
    -m BGE_M3.reranker.run \
    --seed $SEED \
    --output_dir saved_models/$RUN_NAME \
    --model_name_or_path BAAI/bge-reranker-v2-m3 \
    --use_bce_loss True \
    --train_data data/reranker_law \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --tf32 True \
    --bf16 \
    --num_train_epochs 5 \
    --dataloader_drop_last True \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --deepspeed BGE_M3/ds_config.json \
    --dataloader_drop_last True \
    --train_group_size 6 \
    --max_len 2048 \
    --weight_decay 0.01 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --logging_dir "saved_models/$RUN_NAME/log" \
    --logging_steps 10 \
    --save_strategy epoch \
    --resume_from_checkpoint saved_models/m3_reranker_bce_20240718/checkpoint-3497 \
    "$@"