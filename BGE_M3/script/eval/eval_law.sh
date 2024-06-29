#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM="true"

python3 BGE_M3/eval/eval_law.py \
    --model_savedir  saved_models/bgem3_sft_20240523/checkpoint-10227 \
    --corpus_embeddings saved_embeddings/zalo_legal/saved_models/bgem3_sft_20240523/checkpoint-10227/corpus_embedding.npy \
    --save_corpus_embeddings \
    --query_max_length 128 \
    --query_batch_size 512 \
    --passage_max_length 2048 \
    --passage_batch_size 2 \
    --corpus_file data/eval/zalo_legal/filtered_corpus.json \
    --dev_queries_file data/eval/zalo_legal/dev_queries.json \
    --dev_rel_docs_file data/eval/zalo_legal/dev_rel_docs.json \
    --bm25_k1 0.6 \
    --bm25_b 0.9 \
    --bm25_weight 0.2 \
    --rrf_k 10 \
    --colbert_rerank \
    --save_dir BGE_M3/results/zalo_legal.txt \
    "$@"