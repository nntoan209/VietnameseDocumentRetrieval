#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python3 BGE_M3/eval/eval_vietnamese_biencoder.py \
    --query_batch_size 32 \
    --passage_batch_size 32 \
    --corpus_file data/final/corpus/merged_dedup_chunked_corpus_indexed.json \
    --dev_queries_file data/final/test/merged_segmented_dev_queries.json \
    --dev_rel_docs_file data/final/test/merged_dev_rel_docs.json \
    --save_dir results.txt \