#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=8
export WANDB_PROJECT="BGE-M3 SFT MSMARCO SQuADv2 Zalo"
export HF_DATASETS_CACHE="./.cache"
export TORCH_USE_CUDA_DSA=1
PORT_ID=$(expr $RANDOM + 1000)
SEED=$(expr $RANDOM)
RUN_NAME="bgem3_crosslingual_$(date '+%Y%m%d')"

torchrun --nproc_per_node 1 --master_port $PORT_ID \
    -m BGE_M3.src.main_crosslingual \
    --seed $SEED \
    --output_dir saved_models/cross_lingual/$RUN_NAME \
    --model_name_or_path  BAAI/bge-m3 \
    --train_data data/cross_lingual/merged_train_data_splitted \
    --merged_queries_vi data/cross_lingual/merged_queries_vi.json \
    --merged_queries_en data/cross_lingual/merged_queries_en.json \
    --merged_corpus_vi data/cross_lingual/merged_corpus_vi.json \
    --merged_corpus_en data/cross_lingual/merged_corpus_en.json \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --bf16 \
    --tf32 True \
    --gradient_checkpointing \
    --gradient_accumulation_step 1 \
    --deepspeed BGE_M3/ds_config.json \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --dataloader_num_workers 2 \
    --normalized True \
    --temperature 0.04 \
    --query_max_len 128 \
    --passage_max_len 2048 \
    --train_group_size 2 \
    --negatives_cross_device False \
    --logging_steps 20 \
    --save_strategy epoch \
    --same_task_within_batch True \
    --unified_finetuning True \
    --use_self_distill True \
    --aux_loss True \
    --self_distill_start_step 6000 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --resume_from_checkpoint False \
    "$@"