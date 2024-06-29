"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.
"""

from sentence_transformers import  LoggingHandler, util
from BGE_M3.src.utils import BGEM3FlagModel
from BGE_M3.eval.utils import RetrievalEvaluator
import logging
import os

import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

parser = argparse.ArgumentParser()
parser.add_argument("--model_savedir", '--list', nargs='+', required=True, help='model(s) save dir') 
parser.add_argument("--data_folder", default='../msmacro_translated', type=str)
parser.add_argument("--query_max_length", type=int, default=64, help='query max length')
parser.add_argument("--query_batch_size", type=int, default=256, help='query batch size')
parser.add_argument("--passage_max_length", type=int, default=4096, help='passage max length')
parser.add_argument("--passage_batch_size", type=int, default=8, help='passage batch size')
args = parser.parse_args()

# Data files
collection_filepath = os.path.join(args.data_folder, 'collection_translated.tsv')
dev_queries_file = os.path.join(args.data_folder, 'queries_translated.dev.small.tsv')
# If not exists -> raise error
if not os.path.exists(collection_filepath) or not os.path.exists(dev_queries_file):
    raise Exception("Missing collection and queries files")

qrels_filepath = os.path.join(args.data_folder, 'qrels.dev.tsv')
# load qrels file if not exists -> download
if not os.path.exists(qrels_filepath):
    util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv', qrels_filepath)

# Load data
corpus = {}             #Our corpus pid => passage
dev_queries = {}        #Our dev queries. qid => query
dev_rel_docs = {}       #Mapping qid => set with relevant pids
needed_pids = set()     #Passage IDs we need
needed_qids = set()     #Query IDs we need

# Load the 6980 dev queries
with open(dev_queries_file, encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        dev_queries[qid] = query.strip()

# Load which passages are relevant for which queries
with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split('\t')

        if qid not in dev_queries:
            continue

        if qid not in dev_rel_docs:
            dev_rel_docs[qid] = set()
        dev_rel_docs[qid].add(pid)

        needed_pids.add(pid)
        needed_qids.add(qid)

# Read passages
with open(collection_filepath, encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        passage = passage
        corpus[pid] = passage.strip()

# Run evaluator
logging.info("Queries: {}".format(len(dev_queries)))
logging.info("Corpus: {}".format(len(corpus)))

for model_path in args.model_savedir:
    # Load model
    model = BGEM3FlagModel(model_path,
                        pooling_method='cls',
                        use_fp16=True,
                        device=None)

    retrieval_evaluator = RetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                            query_max_length=args.query_max_length,
                                            query_batch_size=args.query_batch_size,
                                            corpus_max_length=args.passage_max_length,
                                            corpus_batch_size=args.passage_batch_size,
                                            show_progress_bar=True,
                                            corpus_chunk_size=100000,
                                            accuracy_at_k=[1, 5, 10, 100],
                                            precision_recall_at_k=[1, 5, 10, 100],
                                            mrr_at_k = [5, 10, 100],
                                            map_at_k=[5, 10, 100],
                                            ndcg_at_k=[5, 10, 100],
                                            name="MsmarcoDevSmall")

    output_path = os.path.join("BGE_M3/results", model_path.split("/")[-1])
    os.makedirs(output_path, exist_ok=True)
    retrieval_evaluator(model, output_path=output_path)
