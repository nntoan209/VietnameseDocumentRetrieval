#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM="true"


python3 BGE_M3/eval/eval_law.py \
    --model_savedir  nntoan209/bgem3-synthetic-v2-e4 \
    --save_corpus_embeddings \
    --corpus_embeddings saved_embeddings/law/nntoan209/bgem3-synthetic-v2-e4/corpus_embedding.npy \
    --query_max_length 128 \
    --query_batch_size 512 \
    --passage_max_length 2096 \
    --passage_batch_size 2 \
    --corpus_file data/eval/law/filtered_corpus.json \
    --dev_queries_file data/eval/law/dev_queries.json \
    --dev_rel_docs_file data/eval/law/dev_rel_docs.json \
    --bm25_k1 0.6 \
    --bm25_b 0.9 \
    --bm25_weight 0.2 \
    --rrf_k 10 \
    --ce_rerank \
    --ce_rerank_model nntoan209/bgem3-reranker-law-bce-e2 \
    --save_dir BGE_M3/results/law_eval_ce.txt
