from BGE_M3.src.utils import BGEM3FlagModel
import numpy as np
import json
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
from BGE_M3.eval.utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = ArgumentParser(description='Eval law using BGE-M3')
parser.add_argument("--model_savedir", type=str, default=None, help='model(s) save dir')
parser.add_argument("--corpus_embeddings", type=str, required=True, help='saved corpus embeddings file')

parser.add_argument("--corpus_file", type=str, default="data/eval/law/filtered_corpus.json", help="path to the corpus file")
parser.add_argument("--dev_queries_file", type=str, default="data/eval/law/dev_queries.json", help="path to the dev queries file")
parser.add_argument("--dev_rel_docs_file", type=str, default="data/eval/law/dev_rel_docs.json", help="path to the dev relevant documents file")

parser.add_argument("--bm25_k1", type=float, default=1.25, required=True, help="BM25 k1 parameter")
parser.add_argument("--bm25_b", type=float, default=0.9, required=True, help="BM25 b parameter")
parser.add_argument("--bm25_weight", type=float, nargs="+", help="BM25 weight for hybrid retrieval")
parser.add_argument("--rrf_k", type=int, default=10, help="k smoothing parameter for RFF score")

args = parser.parse_args()

if __name__ == "__main__":
    
    # If corpus_embeddings is not provided, then model_savedir should be provided
    if args.corpus_embeddings is None:
        assert args.model_savedir is not None, "model_savedir should be provided if corpus_embeddings is not provided"

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
        
    max_k = max(max(mrr_at_k), max(accuracy_at_k), max(precision_recall_at_k), max(ndcg_at_k), max(map_at_k))

    model = BGEM3FlagModel(args.model_savedir,  
                        pooling_method='cls',
                        use_fp16=True, # Setting use_fp16 to True speeds up computation with a slight performance degradation
                        device=None) # Use device=None to use the default device (cuda if available, else cpu)

    print("Dev Queries Embedding ...")
    queries_dev_embedding = model.encode(sentences=list(dev_queries.values()),
                                        batch_size=512,
                                        max_length=128,
                                        return_sparse=False,
                                        return_colbert_vecs=False)
    
    # Load the saved corpus embeddings
    print("Loading documents dense embeddings ...")
    sentences_embedding = {"dense_vecs": np.load(args.corpus_embeddings)}
    
    # Compute dense scores
    print("Calculate dense scores ...")
    sentences_embedding_dense = sentences_embedding['dense_vecs'].astype(np.float32)
    queries_dev_embedding_dense = queries_dev_embedding['dense_vecs'].astype(np.float32)
    dense_scores = model.dense_score(q_reps=queries_dev_embedding_dense, p_reps=sentences_embedding_dense)
    
    # Compute dense ranks
    print("Calculate dense ranks ...")
    dense_ranks = ranks_from_scores_parallel(scores=dense_scores, num_workers=32)
                
    print("Loading BM25 model ...")
    with open(f'saved_models/bm25_{args.bm25_k1}_{args.bm25_b}', 'rb') as bm25result_file:
        bm25 = pickle.load(bm25result_file)
            
    # Calculate the BM25 scores
    print("Loading BM25 scores ...")
    sparse_scores = np.load(f'saved_models/bm25_score_{args.bm25_k1}_{args.bm25_b}.npy')
    
    # Calculate sparse ranks
    print("Calculate sparse ranks ...")
    sparse_ranks = ranks_from_scores_parallel(scores=sparse_scores, num_workers=32)
    
    for w in args.bm25_weight:
        # Compute rrf score
        print(f"Computing RRF scores for bm25_weight = {w} ...")
        hybrid_scores = rrf_from_ranks(dense_ranks=dense_ranks,
                                       sparse_ranks=sparse_ranks,
                                       k=args.rrf_k,
                                       bm25_weight=w
                                       )
        
        hybrid_only_pidqids_results = {}
        for idx, qid in tqdm(enumerate(qids), desc="Get top k results for hybrid retrieval"):
            hybrid_scores_for_qid = hybrid_scores[idx]
            top_k = np.argsort(-hybrid_scores_for_qid)[:max_k]
            hybrid_only_pidqids_results[qid] = [pids[i] for i in top_k]
        
        hybrid_metrics = calculate_metrics(queries_result_list=hybrid_only_pidqids_results,
                                            queries=dev_queries,
                                            relevant_docs=dev_rel_docs,
                                            mrr_at_k=mrr_at_k,
                                            accuracy_at_k=accuracy_at_k,
                                            precision_recall_at_k=precision_recall_at_k,
                                            ndcg_at_k=ndcg_at_k,
                                            map_at_k=map_at_k)
        
        for k, v in hybrid_metrics.items():
            print(f"{k}: {v}")
        with open("test.txt", "a") as fOut:
            fOut.write(f"bm25_weight = {w} rrf_k = {args.rrf_k}\n")
            for k, v in hybrid_metrics.items():
                if not k.startswith("Precision"):
                    v = str(v).strip('%')
                    fOut.write(f"{v}, ")
            fOut.write("\n\n\n")
