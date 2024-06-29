#!/bin/bash

python3 -m text_dedup.minhash \
  --path "parquet" \
  --data_files "data/newssapo/baomoi_combined.parquet" \
  --split "train" \
  --cache_dir "./cache" \
  --output "data/newssapo/baomoi_dedup" \
  --output_cluster "data/newssapo/baomoi_dedup/cluster" \
  --column "body_text" \
  --batch_size 10000 \