#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=8
export WANDB_PROJECT="BGE-M3 Synthetic"
export HF_DATASETS_CACHE="./.cache"
export TORCH_USE_CUDA_DSA=1
PORT_ID=$(expr $RANDOM + 1000)
SEED=$(expr $RANDOM)
RUN_NAME="bgem3_synthetic_$(date '+%Y%m%d')"

torchrun --nproc_per_node 2 --master_port $PORT_ID \
    -m BGE_M3.src.main \
    --seed $SEED \
    --output_dir saved_models/$RUN_NAME \
    --model_name_or_path  BAAI/bge-m3 \
    --train_data data/vi/base data/vi/synthetic \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --bf16 \
    --tf32 True \
    --gradient_checkpointing \
    --gradient_accumulation_step 1 \
    --deepspeed BGE_M3/ds_config.json\
    --num_train_epochs 6 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --dataloader_num_workers 2 \
    --normalized True \
    --temperature 0.04 \
    --query_max_len 128 \
    --passage_max_len 2048 \
    --train_group_size 2 \
    --negatives_cross_device True \
    --logging_steps 20 \
    --save_strategy epoch \
    --same_task_within_batch True \
    --unified_finetuning True \
    --use_self_distill True \
    --self_distill_start_step 10000 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --logging_dir "saved_models/$RUN_NAME/log" \
    --resume_from_checkpoint False \
    "$@"