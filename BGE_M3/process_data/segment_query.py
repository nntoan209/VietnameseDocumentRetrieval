import py_vncorenlp
import json
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser(description='Chunking corpus')
parser.add_argument("--dev_queries_path", type=str, help='path to queries file')
parser.add_argument("--save_path", type=str, help='path to save file')

args = parser.parse_args()

if __name__ == "__main__":
    segment_model = py_vncorenlp.VnCoreNLP(annotators=['wseg'])

    with open(args.dev_queries_path, encoding='utf-8') as fIn:
        segmented_queries = {}

        queries = json.load(fIn)
        for index, query in tqdm(queries.items(), desc="Segment queries"):
            segmented_query = " ".join(segment_model.word_segment(query))
            segmented_queries[index] = segmented_query
    
    with open(args.save_path, "w", encoding='utf-8') as fOut:
        json.dump(segmented_queries, fOut, indent=2)
