from BGE_M3.src.utils import BGEM3FlagModel
from BGE_M3.process_data.utils import bm25_tokenizer
from rank_bm25 import BM25Plus
import numpy as np
import os
import json
import torch
import gc
from tqdm import tqdm
from sentence_transformers import util
from argparse import ArgumentParser
from multiprocessing import Pool
import pickle
from BGE_M3.src.utils import *
from BGE_M3.eval.utils import *

parser = ArgumentParser(description='Eval law using BGE-M3')
parser.add_argument("--model_savedir", type=str, default=None, help='model(s) save dir')
parser.add_argument("--corpus_embeddings", type=str, default=None, help='saved corpus embeddings file')
parser.add_argument("--save_corpus_embeddings", action='store_true', help='save corpus embeddings')
parser.add_argument("--query_max_length", type=int, default=128, help='query max length')
parser.add_argument("--query_batch_size", type=int, default=128, help='query batch size')
parser.add_argument("--passage_max_length", type=int, default=8192, help='passage max length')
parser.add_argument("--passage_batch_size", type=int, default=2, help='passage batch size')

parser.add_argument("--corpus_file", type=str, help="path to the corpus file")
parser.add_argument("--dev_queries_file", type=str, help="path to the dev queries file")
parser.add_argument("--dev_rel_docs_file", type=str, help="path to the dev relevant documents file")

parser.add_argument("--bm25_hybrid", action='store_true', help="use a sparse model (BM25) for hybrid retrieval")
parser.add_argument("--bm25_k1", type=float, default=0.4, help="BM25 k1 parameter")
parser.add_argument("--bm25_b", type=float, default=0.6, help="BM25 b parameter")
parser.add_argument("--bm25_weight", type=float, default=0.1, help="BM25 weight for hybrid retrieval")
parser.add_argument("--rrf_k", type=int, default=10, help="k smoothing parameter for RRF score")

parser.add_argument("--colbert_rerank", action='store_true', help="rerank with colbert")
parser.add_argument("--top_k_rerank", type=int, default=100, help="number of passages to retrieve for reranking")

parser.add_argument("--save_dir", type=str)

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
    
    if not args.colbert_rerank:
        args.top_k_rerank = 0
        
    max_k_hybrid = max(max(mrr_at_k), max(accuracy_at_k), max(precision_recall_at_k), max(ndcg_at_k), max(map_at_k))
    max_k = max(args.top_k_rerank, max_k_hybrid)

    model = BGEM3FlagModel(args.model_savedir,  
                        pooling_method='cls',
                        use_fp16=True, # Setting use_fp16 to True speeds up computation with a slight performance degradation
                        device=None) # Use device=None to use the default device (cuda if available, else cpu)

    print("Dev Queries Embedding ...")
    queries_dev_embedding = model.encode(sentences=list(dev_queries.values()),
                                        batch_size=args.query_batch_size,
                                        max_length=args.query_max_length,
                                        return_sparse=False,
                                        return_colbert_vecs=False)
    
    if args.corpus_embeddings is None:
        print("Documents Embedding ...")
        sentences_embedding = model.encode(sentences=list(corpus.values()),
                                        batch_size=args.passage_batch_size,
                                        max_length=args.passage_max_length,
                                        return_sparse=False,
                                        return_colbert_vecs=False)
        if args.save_corpus_embeddings:
            print("Saving documents dense embeddings ...")
            os.makedirs(f"saved_embeddings/{args.model_savedir}", exist_ok=True)
            if args.corpus_file.split(".")[0].endswith("_en"):
                np.save(f"saved_embeddings/{args.model_savedir}/corpus_embedding_en.npy", sentences_embedding["dense_vecs"])
            else:
                np.save(f"saved_embeddings/{args.model_savedir}/corpus_embedding.npy", sentences_embedding["dense_vecs"])
    else:
        # Load the saved corpus embeddings
        print("Loading documents dense embeddings ...")
        sentences_embedding = {"dense_vecs": np.load(args.corpus_embeddings)}

    only_pidqids_results = {}
    hybrid_only_pidqids_results = {}
    rerank_only_pidqids_results = {}

    # Dense retrieval, using sentence_transformers semantic_search function.
    # This function requires float32 tensors as input
    sentences_embedding_dense = torch.from_numpy(sentences_embedding['dense_vecs']).to(torch.float32)
    queries_dev_embedding_dense = torch.from_numpy(queries_dev_embedding['dense_vecs']).to(torch.float32)

    results_dense_search = util.semantic_search(queries_dev_embedding_dense, sentences_embedding_dense, top_k=max_k)

    converted_results = {}
    for idx, result in enumerate(results_dense_search):
            for answer in result:
                answer['corpus_id'] = pids[answer['corpus_id']]
            converted_results[qids[idx]] = result
    for qid, result in converted_results.items():
        only_pidqids_results[qid] = [answer['corpus_id'].strip() for answer in result]
        

    if args.bm25_hybrid: # Hybrid retrieval with dense and sparse
        
        # Compute dense scores
        print("Calculating dense scores ...")
        sentences_embedding_dense = sentences_embedding['dense_vecs'].astype(np.float32)
        queries_dev_embedding_dense = queries_dev_embedding['dense_vecs'].astype(np.float32)
        dense_scores = model.dense_score(q_reps=queries_dev_embedding_dense, p_reps=sentences_embedding_dense)
        
        # Train the BM25 model
        if os.path.exists(f'saved_models/zalo_legal/bm25_{args.bm25_k1}_{args.bm25_b}'):
            print("Loading BM25 model ...")
            with open(f'saved_models/zalo_legal/bm25_{args.bm25_k1}_{args.bm25_b}', 'rb') as bm25result_file:
                bm25 = pickle.load(bm25result_file)
        else:
            print("Fitting BM25 model ...")
            bm25 = BM25Plus(corpus=list(corpus.values()),
                            tokenizer=bm25_tokenizer,
                            k1=args.bm25_k1,
                            b=args.bm25_b)
            print("Saving BM25 model ...")
            with open(f'saved_models/zalo_legal/bm25_{args.bm25_k1}_{args.bm25_b}', 'wb') as bm25result_file:
                pickle.dump(bm25, bm25result_file)
                
        # Calculate the BM25 scores
        if os.path.exists(f'saved_models/zalo_legal/bm25_score_{args.bm25_k1}_{args.bm25_b}.npy'):
            print("Loading BM25 scores ...")
            sparse_scores = np.load(f'saved_models/zalo_legal/bm25_score_{args.bm25_k1}_{args.bm25_b}.npy')
        else:
            print("Calculating BM25 scores ...")
            queries_split_size = 100
            split_queries = [list(dev_queries.values())[i:i + queries_split_size] for i in range(0, len(dev_queries), queries_split_size)]
            def process_split(split):
                return [bm25.get_scores(bm25_tokenizer(query)).astype(np.float32) for query in split]
            with Pool(64) as p:
                parallel_results = p.map(process_split, split_queries)
                
            sparse_scores = np.zeros((len(dev_queries), len(corpus)), dtype=np.float32)
            for i, split in enumerate(parallel_results):
                for j, query_scores in enumerate(split):
                    sparse_scores[i * queries_split_size + j] = query_scores
        
        # Compute rrf score
        print("Computing RRF scores ...")
        hybrid_scores = rrf_from_scores_parallel(scores1=dense_scores,
                                                scores2=sparse_scores,
                                                k=args.rrf_k,
                                                bm25_weight=args.bm25_weight,
                                                num_workers=32)

        for idx, qid in tqdm(enumerate(qids), desc="Get top k results for hybrid retrieval"):
            hybrid_scores_for_qid = hybrid_scores[idx]
            top_k = np.argsort(-hybrid_scores_for_qid)[:max_k_hybrid]
            hybrid_only_pidqids_results[qid] = [pids[i] for i in top_k]


    # Rerank with Colbert
    if args.colbert_rerank:
        
        if args.bm25_hybrid:
            only_pidqids_to_rerank = hybrid_only_pidqids_results
        else:
            only_pidqids_to_rerank = only_pidqids_results
        
        for qid, pids in tqdm(only_pidqids_to_rerank.items(), desc="Rerank with Colbert"):
            colbert_scores_for_qid = model.compute_batch_colbert_score(query=dev_queries[qid],
                                                                       passages=[corpus[pid] for pid in pids],
                                                                       batch_size=args.passage_batch_size,
                                                                       max_query_length=args.query_max_length,
                                                                       max_passage_length=args.passage_max_length)

            # Sort the pair (pid, colbert_score) in descending order based on the colbert_score
            pids_with_colbert_scores_for_qid = list(zip(only_pidqids_to_rerank[qid], colbert_scores_for_qid))
            rerank_pids_with_colbert_scores_for_qid = sorted(pids_with_colbert_scores_for_qid, key=lambda x: x[1], reverse=True)
            
            # Get the pids only
            rerank_only_pidqids_results[qid] = [answer[0] for answer in rerank_pids_with_colbert_scores_for_qid]
            
    ### Calculate and write metrics
    metrics = calculate_metrics(queries_result_list=only_pidqids_results,
                                queries=dev_queries,
                                relevant_docs=dev_rel_docs,
                                mrr_at_k=mrr_at_k,
                                accuracy_at_k=accuracy_at_k,
                                precision_recall_at_k=precision_recall_at_k,
                                ndcg_at_k=ndcg_at_k,
                                map_at_k=map_at_k)
    
    if args.bm25_hybrid:
        hybrid_metrics = calculate_metrics(queries_result_list=hybrid_only_pidqids_results,
                                            queries=dev_queries,
                                            relevant_docs=dev_rel_docs,
                                            mrr_at_k=mrr_at_k,
                                            accuracy_at_k=accuracy_at_k,
                                            precision_recall_at_k=precision_recall_at_k,
                                            ndcg_at_k=ndcg_at_k,
                                            map_at_k=map_at_k)
    
    if args.colbert_rerank:
        rerank_metrics = calculate_metrics(queries_result_list=rerank_only_pidqids_results,
                                            queries=dev_queries,
                                            relevant_docs=dev_rel_docs,
                                            mrr_at_k=mrr_at_k,
                                            accuracy_at_k=accuracy_at_k,
                                            precision_recall_at_k=precision_recall_at_k,
                                            ndcg_at_k=ndcg_at_k,
                                            map_at_k=map_at_k)
    
    with open(args.save_dir, "a") as fOut:
        fOut.write(f"{args.model_savedir}\n")
        fOut.write(f"{args.corpus_file} {args.dev_queries_file}\n")
        fOut.write(f"Dense search:\n")
        for key, value in metrics.items():
            fOut.write(f"\t{key}: {value}\n")
            
        if args.bm25_hybrid:
            fOut.write(f"Hybrid search:\n")
            for key, value in hybrid_metrics.items():
                fOut.write(f"\t{key}: {value}\n")
                
        if args.colbert_rerank:
            if args.bm25_hybrid:
                fOut.write(f"Hybrid search + Colbert rerank:\n")
            else:
                fOut.write(f"Colbert rerank:\n")
            for key, value in rerank_metrics.items():
                fOut.write(f"\t{key}: {value}\n")
                
        fOut.write(f"-----------------------------------------------------------------------------------------------------------------\n\n")
    
    # clean up
    del model
    del sentences_embedding
    del queries_dev_embedding
    del sentences_embedding_dense
    del queries_dev_embedding_dense
    if args.bm25_hybrid:
        del bm25
        del dense_scores
        del sparse_scores
        del hybrid_scores
    gc.collect()
    torch.cuda.empty_cache()
