#!/bin/bash

# export JAVA_HOME='/usr/lib/jvm/jre-11-openjdk'
# export JVM_PATH='/usr/lib/jvm/jre-11-openjdk/lib/server/libjvm.so'

python3 BGE_M3/process_data/chunk_corpus.py \
    --corpus_path data/eval/zalo_legal/filtered_corpus.json \
    --chunk_size 256 \
    --chunk_overlap 32 \
    --save_path data/eval/zalo_legal/filtered_corpus_chunked.json \
