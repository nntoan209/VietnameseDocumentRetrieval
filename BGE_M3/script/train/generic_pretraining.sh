#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
export OMP_NUM_THREADS=8
export WANDB_PROJECT="BGE-M3 Generic"
export HF_DATASETS_CACHE="./.cache"
export TORCH_USE_CUDA_DSA=1
SEED=$(expr $RANDOM)
RUN_NAME="bgem3_generic_pretraining_$(date '+%Y%m%d')"

torchrun --nproc_per_node 1 \
    -m BGE_M3.src.main \
    --seed $SEED \
    --output_dir saved_models/$RUN_NAME \
    --model_name_or_path  ../../../home/admin_mcn/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/babcf60cae0a1f438d7ade582983d4ba462303c2 \
    --train_data data/train_generic/structured_data_doc_splitted data/train_generic/newssapo_filtered_splitted \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --bf16 \
    --tf32 True \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --deepspeed BGE_M3/ds_config.json\
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --dataloader_num_workers 2 \
    --normalized True \
    --temperature 0.05 \
    --query_max_len 128 \
    --passage_max_len 4096 \
    --train_group_size 1 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_strategy epoch \
    --same_task_within_batch True \
    --unified_finetuning False \
    --use_self_distill False \
    --self_distill_start_step 200 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --logging_dir "saved_models/$RUN_NAME/log" \
    --resume_from_checkpoint False \
    "$@"