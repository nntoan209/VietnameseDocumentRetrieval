#!/bin/bash

export JAVA_HOME='/usr/lib/jvm/jre-11-openjdk'
export JVM_PATH='/usr/lib/jvm/jre-11-openjdk/lib/server/libjvm.so'

python3 BGE_M3/process_data/chunk_corpus.py \
    --corpus_path data/final/corpus/merged_dedup_corpus_indexed.json \
    --chunk_size 256 \
    --chunk_overlap 10 \
    --save_path data/final/corpus/merged_dedup_chunked_corpus_indexed.json \
