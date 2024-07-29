import py_vncorenlp
import json
import os
from tqdm import tqdm

segment_model = py_vncorenlp.VnCoreNLP(annotators=['wseg'])

data_paths = ["data/eval/law", "data/eval/zalo_legal", "data/eval/zalo_qa"]

for data_path in data_paths:

    print(data_path)
    
    corpus_path = os.path.join(data_path, "filtered_corpus.json")
    queries_path = os.path.join(data_path, "dev_queries.json")

    # segment queries
    with open(queries_path, encoding='utf-8') as fIn:
        segmented_queries = {}

        queries = json.load(fIn)
        for index, query in tqdm(queries.items()):
            segmented_query = " ".join(segment_model.word_segment(query))
            segmented_queries[index] = segmented_query
            
        with open(os.path.join(data_path, "dev_queries_segmented.json"), "w", encoding='utf-8') as fOut:
            json.dump(segmented_queries, fOut, indent=2, ensure_ascii=False)
            

    # segment corpus
    with open(corpus_path, encoding='utf-8') as fIn:
        segmented_corpus = {}

        corpus = json.load(fIn)
        for index, doc in tqdm(corpus.items()):
            segmented_doc = " ".join(segment_model.word_segment(doc))
            segmented_corpus[index] = segmented_doc
            
        with open(os.path.join(data_path, "filtered_corpus_segmented.json"), "w", encoding='utf-8') as fOut:
            json.dump(segmented_corpus, fOut, indent=2, ensure_ascii=False)
