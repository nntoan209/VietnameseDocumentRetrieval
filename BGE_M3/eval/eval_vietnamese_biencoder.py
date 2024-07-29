from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

import json

ds = 'zalo_legal'

corpus = json.load(open(f"data/eval/{ds}/filtered_corpus_segmented.json", encoding='utf-8'))
dev_queries = json.load(open(f"data/eval/{ds}/dev_queries_segmented.json", encoding='utf-8'))
dev_rel_docs = json.load(open(f"data/eval/{ds}/dev_rel_docs.json", encoding='utf-8'))

qids = list(dev_queries.keys())
pids = list(corpus.keys())

mrr_at_k = [5, 10, 100]
accuracy_at_k = [1, 5, 10, 100]
precision_recall_at_k = [1, 5, 10, 100]
ndcg_at_k = [5, 10, 100]
map_at_k = [5, 10, 100]

print("Dev Queries Embedding ...")
queries_dev_embedding = model.encode(list(dev_queries.values()),
                                     batch_size=256,
                                     convert_to_tensor=True,
                                     normalize_embeddings=True,
                                     device="cuda:0",
                                     show_progress_bar=True)

print("Documents Embedding ...")
sentences_embedding = model.encode(list(corpus.values()),
                                   batch_size=8,
                                   convert_to_tensor=True,
                                   normalize_embeddings=True,
                                   device="cuda:0",
                                   show_progress_bar=True)

only_pidqids_results = {}
results_dense_search = util.semantic_search(queries_dev_embedding, sentences_embedding, top_k=100)
converted_results = {}
for idx, result in enumerate(results_dense_search):
    for answer in result:
        answer['corpus_id'] = pids[answer['corpus_id']]
    converted_results[qids[idx]] = result
for qid, result in converted_results.items():
    only_pidqids_results[qid] = [answer['corpus_id'].strip() for answer in result]
    # if ds == 'zalo_legal':
        # final_list = []
        # for a in ["_".join(answer['corpus_id'].strip().split()[:-1]) for answer in result]:
        #     if a not in final_list:
        #         final_list.append(a)
        #     else:
        #         final_list.append("already_exist")
        # only_pidqids_results[qid] = final_list

    
from BGE_M3.eval.utils import *
results = calculate_metrics(queries_result_list=only_pidqids_results,
                        queries=dev_queries,
                        relevant_docs=dev_rel_docs,
                        mrr_at_k=mrr_at_k,
                        accuracy_at_k=accuracy_at_k,
                        precision_recall_at_k=precision_recall_at_k,
                        ndcg_at_k=ndcg_at_k,
                        map_at_k=map_at_k)

for key, value in results.items():
    print(key, value)