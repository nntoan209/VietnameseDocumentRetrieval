from argparse import ArgumentParser
import numpy as np
import pickle
import json
import os
from tqdm import tqdm
from BGE_M3.process_data.utils import bm25_tokenizer
from rank_bm25 import BM25Plus
from multiprocessing import Pool
from BGE_M3.eval.utils import calculate_metrics

parser = ArgumentParser(description='Eval law using BM25')
parser.add_argument("--corpus_file", type=str, default="data/eval/zalo_qa/filtered_corpus.json", help="path to the corpus file")
parser.add_argument("--dev_queries_file", type=str, default="data/eval/zalo_qa/dev_queries.json", help="path to the dev queries file")
parser.add_argument("--dev_rel_docs_file", type=str, default="data/eval/zalo_qa/dev_rel_docs.json", help="path to the dev relevant documents file")
parser.add_argument("--bm25_result_model", type=str, default=None, help="path to the bm25 result model")
parser.add_argument("--bm25_result_score", type=str, default=None, help="path to the bm25 result rank")
parser.add_argument("--bm25_k1", type=float, default=0.4, help="BM25 k1 parameter")
parser.add_argument("--bm25_b", type=float, default=0.6, help="BM25 b parameter")
parser.add_argument("--save_dir", type=str, default="saved_models/zalo_legal", help="path to save the model")

args = parser.parse_args()

if __name__ == "__main__":
    
    corpus = json.load(open(args.corpus_file, encoding='utf-8'))
    dev_queries = json.load(open(args.dev_queries_file, encoding='utf-8'))
    dev_rel_docs = json.load(open(args.dev_rel_docs_file, encoding='utf-8'))
    
    qids = list(dev_queries.keys())
    pids = list(corpus.keys())
    
    mrr_at_k = [5, 10, 100]
    accuracy_at_k = [1, 5, 10, 100]
    precision_recall_at_k = [1, 5, 10, 100]
    ndcg_at_k = [5, 10, 100]
    map_at_k = [5, 10, 100]
    
    if args.bm25_result_model is not None:
        print("Loading BM25 model ...")
        with open(args.bm25_result_model, 'rb') as bm25result_file:
            bm25 = pickle.load(bm25result_file)
    else:
        print("Fitting BM25 model ...")
        bm25 = BM25Plus(corpus=list(corpus.values()),
                        tokenizer=bm25_tokenizer,
                        k1=args.bm25_k1,
                        b=args.bm25_b)
        
        with open(os.path.join(args.save_dir, f'bm25_{args.bm25_k1}_{args.bm25_b}'), 'wb') as bm25result_file:
            pickle.dump(bm25, bm25result_file)
    
    # Convert from BM25 scores to ranks
    if args.bm25_result_score is not None:
        print("Loading BM25 scores ...")
        sparse_scores = np.load(args.bm25_result_score)
    else:
        print("Calculate bm25 scores ...")
        queries_split_size = 100
        split_queries = [list(dev_queries.values())[i:i + queries_split_size] for i in range(0, len(dev_queries), queries_split_size)]
        def process_split(split):
            return [bm25.get_scores(bm25_tokenizer(query)).astype(np.float32) for query in split]
        with Pool(64) as p:
            parallel_results = p.map(process_split, split_queries)
            
        sparse_scores = np.zeros((len(dev_queries), len(corpus)), dtype=np.float16)
        for i, split in enumerate(parallel_results):
            for j, query_scores in enumerate(split):
                sparse_scores[i * queries_split_size + j] = query_scores
        print("Saving BM25 scores ...")
        np.save(os.path.join(args.save_dir, f"bm25_score_{args.bm25_k1}_{args.bm25_b}.npy"), sparse_scores)
        
    only_pidqids_results = {}
    print("Get top k results ...")
    for idx, qid in tqdm(enumerate(qids), desc="BM25 retrieval"):
        top_k = np.argsort(-sparse_scores[idx])[:100]
        only_pidqids_results[qid] = [pids[i] for i in top_k]
        
    print("Calculating metrics ...")
    metrics = calculate_metrics(queries_result_list=only_pidqids_results,
                                relevant_docs=dev_rel_docs,
                                queries=dev_queries,
                                mrr_at_k=mrr_at_k,
                                accuracy_at_k=accuracy_at_k,
                                precision_recall_at_k=precision_recall_at_k,
                                ndcg_at_k=ndcg_at_k,
                                map_at_k=map_at_k)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")
