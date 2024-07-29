#!/bin/bash

# export CUDA_VISIBLE_DEVICES="6,7"

python3 BGE_M3/eval/eval_vnbiencoder_rerank_vncrossencoder.py \
    --biencoder_path bkai-foundation-models/vietnamese-bi-encoder\
    --crossencoder_path bkai-foundation-models/vietnamese-bi-encoder \
    --query_max_length 128 \
    --query_batch_size 512 \
    --passage_max_length 256 \
    --passage_batch_size 256 \
    --corpus_file data/final/corpus/merged_dedup_chunked_corpus_indexed.json \
    --dev_queries_file data/final/test/merged_segmented_dev_queries.json \
    --dev_rel_docs_file data/final/test/merged_dev_rel_docs.json \
    --save_dir results.txt \