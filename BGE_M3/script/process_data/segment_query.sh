#!/bin/bash

export JAVA_HOME='/usr/lib/jvm/jre-11-openjdk'
export JVM_PATH='/usr/lib/jvm/jre-11-openjdk/lib/server/libjvm.so'

python3 BGE_M3/process_data/segment_query.py \
    --dev_queries_path data/final/test/merged_dev_queries.json \
    --save_path data/final/test/merged_segmented_dev_queries.json \
