import argparse
import json
import random
import numpy as np
import faiss
from tqdm import tqdm
import os

from BGE_M3.src.utils import BGEM3FlagModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-base-en", type=str)
    parser.add_argument('--input_file', default=None, type=str)
    parser.add_argument('--corpus_file', default=None, type=str)
    parser.add_argument('--reindexed_corpus_file', default=None, type=str)
    parser.add_argument('--candidate_pool', default=None, type=str)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--range_for_sampling', default="10-210", type=str, help="range to sample negatives")
    parser.add_argument('--use_index', action='store_true', help='use index for pos and neg document')
    parser.add_argument('--use_gpu_for_searching', action='store_true', help='use faiss-gpu')
    parser.add_argument('--negative_number', default=15, type=int, help='the number of negatives')

    return parser.parse_args()


def create_index(embeddings, use_gpu):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(index,
                 query,
                 topk: int = 200,
                 batch_size: int = 64):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def get_corpus(candidate_pool):
    corpus = []
    for line in open(candidate_pool):
        line = json.loads(line.strip())
        corpus.append(line['text'])
    return corpus


def find_knn_neg(model: BGEM3FlagModel, input_file, corpus_file, reindexed_corpus_file, candidate_pool, output_file, sample_range, negative_number, use_gpu):
    corpus = []
    with open(corpus_file, encoding='utf-8') as f:
        filtered_corpus = json.load(f)
        invert_corpus_indexed = {v: k for k, v in filtered_corpus.items()}
        corpus = list(filtered_corpus.values())
        corpus_index = list(filtered_corpus.keys())
        
    if reindexed_corpus_file is not None:
        with open(reindexed_corpus_file, encoding='utf-8') as f:
            reindexed_corpus = json.load(f)
    
    queries = []
    train_data = []
    for line in open(input_file, encoding='utf-8'):
        line = json.loads(line.strip())
        train_data.append(line)
        # corpus.extend(line['pos'])
        # if 'neg' in line:
        #     corpus.extend(line['neg'])
        queries.append(line['query'])

    if candidate_pool is not None:
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))

    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode(queries, batch_size=512, max_length=128)['dense_vecs']
    if os.path.exists(f'saved_embeddings/zalo_legal/{args.model_name_or_path}/corpus_embedding.npy'):
        print('load corpus embedding from saved file')
        p_vecs = np.load(f'saved_embeddings/zalo_legal/{args.model_name_or_path}/corpus_embedding.npy')
    else:
        print(f'inferencing embedding for corpus (number={len(corpus)})--------------')
        p_vecs = model.encode(corpus, batch_size=4, max_length=4096)['dense_vecs']

    print('create index and search------------------')
    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1])
    assert len(all_inxs) == len(train_data)

    for i, data in enumerate(train_data):
        query = data['query']
        context_cluster = data.get('__context_cluster__', None)
        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        if context_cluster is not None:
            for inx in inxs:
                if inx == -1: break
                if not args.use_index:
                    # check:
                    # 1. invert_corpus_indexed[corpus[inx]] not in context_cluster
                    # 2. corpus[inx] not in data['pos']
                    if corpus[inx] not in data['pos'] and corpus[inx] != query and invert_corpus_indexed[corpus[inx]] not in context_cluster:
                        filtered_inx.append(inx)
                else:
                    # check:
                    # 1. invert_corpus_indexed[corpus[inx]] not in context_cluster
                    # 2. corpus[inx] not in pos_passages (derived from data['pos])
                    pos_passages = []
                    for pid in data['pos']:
                        if pid.startswith('oid'):
                            pos_passages.append(reindexed_corpus[pid.split('_')[1]])
                        elif pid.startswith('cluster'):
                            pos_passages.append(filtered_corpus[pid.split('_')[1]])
                            
                    if invert_corpus_indexed[corpus[inx]] not in context_cluster \
                    and corpus[inx] not in pos_passages:
                        filtered_inx.append(inx)
        else:
            for inx in inxs:
                if inx == -1: break
                if not args.use_index:
                    if corpus[inx] not in data['pos'] and corpus[inx] != query:
                        filtered_inx.append(inx)
                else:
                    if corpus_index[inx] not in data['pos'] and corpus[inx] != query:
                        filtered_inx.append(inx)

        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)
        data['neg'] = [corpus_index[inx] if args.use_index else corpus[inx] for inx in filtered_inx]

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in train_data:
            if data.get('__context_cluster__', None) is not None:
                data.pop('__context_cluster__')
            if len(data['neg']) < negative_number and not args.use_index:
                data['neg'].extend(random.sample(corpus, negative_number - len(data['neg'])))
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = get_args()
    sample_range = args.range_for_sampling.split('-')
    sample_range = [int(x) for x in sample_range]

    model = BGEM3FlagModel(args.model_name_or_path,
                           pooling_method='cls',
                           normalize_embeddings=True,
                           use_fp16=True,
                           device=None)

    find_knn_neg(model,
                 input_file=args.input_file,
                 corpus_file=args.corpus_file,
                 reindexed_corpus_file=args.reindexed_corpus_file,
                 candidate_pool=args.candidate_pool,
                 output_file=args.output_file,
                 sample_range=sample_range,
                 negative_number=args.negative_number,
                 use_gpu=args.use_gpu_for_searching)
