#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM="true"

# vi-vi
# python3 BGE_M3/eval/eval_law.py \
#     --model_savedir BAAI/bge-m3 \
#     --corpus_embeddings saved_embeddings/BAAI/bge-m3/corpus_embedding.npy \
#     --save_corpus_embeddings \
#     --query_max_length 128 \
#     --query_batch_size 512 \
#     --passage_max_length 2048 \
#     --passage_batch_size 4 \
#     --colbert_rerank \
#     --corpus_file data/eval/law/filtered_corpus.json \
#     --dev_queries_file data/eval/law/dev_queries.json \
#     --dev_rel_docs_file data/eval/law/dev_rel_docs.json \
#     --save_dir BGE_M3/results/law_eval.txt

# en-vi
# python3 BGE_M3/eval/eval_law.py \
#     --model_savedir saved_models/vi/bgem3_vi_20240618/checkpoint-25638 \
#     --corpus_embeddings saved_embeddings/saved_models/vi/bgem3_vi_20240618/checkpoint-25638/corpus_embedding.npy \
#     --save_corpus_embeddings \
#     --query_max_length 128 \
#     --query_batch_size 512 \
#     --passage_max_length 2048 \
#     --passage_batch_size 1 \
#     --colbert_rerank \
#     --corpus_file data/eval/law/filtered_corpus.json \
#     --dev_queries_file data/eval/law/dev_queries_en.json \
#     --dev_rel_docs_file data/eval/law/dev_rel_docs.json \
#     --save_dir BGE_M3/results/law_eval.txt

# vi-en
# python3 BGE_M3/eval/eval_law.py \
#     --model_savedir saved_models/vi/bgem3_vi_20240618/checkpoint-25638 \
#     --save_corpus_embeddings \
#     --query_max_length 128 \
#     --query_batch_size 512 \
#     --passage_max_length 2048 \
#     --passage_batch_size 1 \
#     --colbert_rerank \
#     --corpus_file data/eval/law/filtered_corpus_en.json \
#     --dev_queries_file data/eval/law/dev_queries.json \
#     --dev_rel_docs_file data/eval/law/dev_rel_docs.json \
#     --save_dir BGE_M3/results/law_eval.txt

# en-en
python3 BGE_M3/eval/eval_law.py \
    --model_savedir saved_models/vi/bgem3_vi_20240618/checkpoint-25638 \
    --corpus_embeddings saved_embeddings/saved_models/vi/bgem3_vi_20240618/checkpoint-25638/corpus_embedding_en.npy \
    --save_corpus_embeddings \
    --query_max_length 128 \
    --query_batch_size 512 \
    --passage_max_length 2048 \
    --passage_batch_size 2 \
    --colbert_rerank \
    --corpus_file data/eval/law/filtered_corpus_en.json \
    --dev_queries_file data/eval/law/dev_queries_en.json \
    --dev_rel_docs_file data/eval/law/dev_rel_docs.json \
    --save_dir BGE_M3/results/law_eval.txt