import json
from argparse import ArgumentParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool

parser = ArgumentParser(description='Chunking corpus')
parser.add_argument("--corpus_path", type=str, help='path to corpus file')
parser.add_argument("--chunk_size", type=int, default=256, help='maximum chunk size')
parser.add_argument("--chunk_overlap", type=int, default=10, help='overlap size')
parser.add_argument("--save_path", type=str, help='path to save file')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " ", ""],
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap,
    length_function=lambda x: len(tokenizer(x)["input_ids"]),
    is_separator_regex=False,
)

if __name__ == "__main__":
    with open(args.corpus_path, encoding='utf-8') as f:
        corpus = json.load(f)
    
    chunked_corpus = {}

    for original_index, original_law in tqdm(corpus.items(), desc="Processing"):
        chunked_law = text_splitter.create_documents([original_law])

        idx = 0
        for law in chunked_law:
            chunked_corpus[f"{original_index}_{idx}"] = law.page_content
            idx += 1
        
    with open(args.save_path, "w", encoding='utf-8') as f:
        json.dump(chunked_corpus, f, ensure_ascii=False, indent=2)
