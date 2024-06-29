INDEX_DATASET = 10
PART_DATASET = 112

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

NUM_CPU = os.cpu_count()
HF_TOKEN = "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"
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    MOUNT_PATH = "/content/drive"
else:
    MOUNT_PATH = "."
    
EXTEND_PATH = "data/law_translated"

CURRENT_FOLDER_PATH = os.path.join(MOUNT_PATH, EXTEND_PATH)
os.makedirs(CURRENT_FOLDER_PATH, exist_ok=True)

import json
import datasets

dataset = json.load(open("data/eval/law/filtered_corpus.json", encoding='utf-8'))
dataset = {"id": list(dataset.keys()), "text": list(dataset.values())}
dataset = datasets.Dataset.from_dict(dataset)

import re
import sys
import typing as tp
import unicodedata

from sacremoses import MosesPunctNormalizer
from sentence_splitter import SentenceSplitter

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def sentenize_with_fillers(text, splitter, fix_double_space=True, ignore_errors=False):
    """Apply a sentence splitter and return the sentences and all separators before and after them"""
    if fix_double_space:
        text = re.sub(" +", " ", text)
        text = re.sub(r"\.+", ".", text)
    sentences = splitter.split(text)
    fillers = []
    i = 0
    for sentence in sentences:
        start_idx = text.find(sentence, i)
        if ignore_errors and start_idx == -1:
            # print(f"sent not found after {i}: `{sentence}`")
            start_idx = i + 1
        assert start_idx != -1, f"sent not found after {i}: `{sentence}`"
        fillers.append(text[i:start_idx])
        i = start_idx + len(sentence)
    fillers.append(text[i:])
    return sentences, fillers

def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

class TextPreprocessor:
    """
    Mimic the text preprocessing made for the NLLB model.
    This code is adapted from the Stopes repo of the NLLB team:
    https://github.com/facebookresearch/stopes/blob/main/stopes/pipelines/monolingual/monolingual_line_processor.py#L214
    """

    def __init__(self, lang="en"):
        self.mpn = MosesPunctNormalizer(lang=lang)
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
        self.replace_nonprint = get_non_printing_char_replacer(" ")

    def __call__(self, text: str) -> str:
        clean = self.mpn.normalize(text)
        clean = self.replace_nonprint(clean)
        # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
        clean = unicodedata.normalize("NFKC", clean)
        return clean
    
from transformers.tokenization_utils_base import BatchEncoding

INITIAL_BATCH_SIZE = 8
MAX_LENGTH = 328

class Translator():
    def __init__(self, model_url):

        if torch.cuda.is_available():
            print("Running on GPU ...")
            self.device = torch.device("cuda")
        else:
            print("WARNING. Running on CPU ...")
            self.device = torch.device("cpu")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_url)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_url, src_lang="vi_VN")
        self.splitter = SentenceSplitter("en")
        self.preprocessor = TextPreprocessor()

    def translate(self, text, max_length_generation="auto",
                  by_sentence=True, preprocess=True,
                  translate_single=True, batch_size=None,
                  **kwargs):
        """Translate a text sentences by sentences, preserving the fillers around the sentences."""
        if by_sentence:
            sents, fillers = sentenize_with_fillers(text, splitter=self.splitter, ignore_errors=True)
        else:
            sents = [text]
            fillers = ["", ""]
        if preprocess:
            sents = [self.preprocessor(sent) for sent in sents]
        results = []
        if not translate_single and batch_size: # Batch inference
            current_translate_idxs = []
            current_translate_sents = []
            for idx, sent in enumerate(sents):
                if len(sent.strip()):
                    current_translate_idxs.append(idx)
                    current_translate_sents.append(sent)
                if (idx == len(sents) - 1 or len(current_translate_idxs) == batch_size) and len(current_translate_idxs):
                    # tranaslate
                    current_results = self.translate_vi2en(current_translate_sents, max_length_generation, **kwargs)
                    assert len(current_results) == len(current_translate_idxs), "Something wrong, not align len"
                    for idx_result, sent_result in zip(current_translate_idxs, current_results):
                         sents[idx_result] = sent_result

            for sent, sep in zip(sents, fillers):
                results.append(sep)
                results.append(sent.strip())

            results.append(fillers[-1])
            # torch.cuda.empty_cache()
            return "".join(results)

        else: # Single inference
            for sent, sep in zip(sents, fillers):
                results.append(sep)
                if len(sent.strip()):
                    results.append(self.translate_vi2en(sent, max_length_generation, **kwargs)[0])
                else:
                    results.append(sent)

            results.append(fillers[-1])
            # torch.cuda.empty_cache()
            return "".join(results)




    def translate_vi2en(self, text, max_length_generation, **kwargs):
        input_ids = self.tokenizer(text,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt",
                                    pad_to_multiple_of=8,
                                    max_length=MAX_LENGTH)

        maximum_token_current_batch = input_ids.input_ids.shape[1]
        num_input_ids = input_ids.input_ids.shape[0]
        if max_length_generation == "auto":
            max_length_generation = int(10 + 1.2* maximum_token_current_batch)

        out = []
        if maximum_token_current_batch <= 128:
          ideal_batch_size = 8
        elif maximum_token_current_batch <= 192:
          ideal_batch_size = 4
        else:
          ideal_batch_size = 1

        for i in range(0, num_input_ids, ideal_batch_size):
          # Create mini-batch
          mini_input_ids = {}
          for key, value in input_ids.items():
              mini_input_ids[key] = value[i:min(i+ideal_batch_size, num_input_ids)]

          mini_input_ids = BatchEncoding(data=mini_input_ids)

          generated_tokens = self.model.generate(
              **mini_input_ids.to(self.device),
              decoder_start_token_id=self.tokenizer.lang_code_to_id["en_XX"],
              max_length=max_length_generation,
              num_return_sequences=1,
              num_beams=5,
              early_stopping=True,
              **kwargs,
          )
          out.extend(self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
        return out
    
LEN_DATASET = len(dataset)
NUM_PER_PART_DATASET = (LEN_DATASET + PART_DATASET - 1 ) //  PART_DATASET
START_IDX = (INDEX_DATASET - 1) * NUM_PER_PART_DATASET
END_IDX = min(START_IDX + NUM_PER_PART_DATASET, LEN_DATASET)

print("******** RUNNING **********")
print(f"Part {INDEX_DATASET} of {PART_DATASET}")
print(f"From {START_IDX} to {END_IDX}")
print("***************************")

dataset = dataset.select(range(START_IDX, END_IDX))

MODEL_URL = "vinai/vinai-translate-vi2en-v2"
FILE_NAME = f"law_translated-{INDEX_DATASET}-of-{PART_DATASET}.json"
OUTPUT_FOLDER = os.path.join(CURRENT_FOLDER_PATH, "data")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

FILE_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, FILE_NAME)
SAVE_PER_SAMPLES = 20

# PUSH TO HF
PUSH_TO_HF_PER_SAMPLES = 300
HF_REPO_PATH = "nntoan209/law_translated"

from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=HF_TOKEN)

api.create_repo(repo_id=HF_REPO_PATH,
                private=True,
                repo_type="dataset",
                exist_ok=True)

# Download
try:
    hf_hub_download(repo_id=HF_REPO_PATH,
                    token=HF_TOKEN,
                    repo_type="dataset",
                    local_dir=OUTPUT_FOLDER,
                    filename=FILE_NAME,
                    local_files_only=False)

except Exception as e:
    print("Warning, not exist files on huggingface")
    pass

translator = Translator(MODEL_URL)

# Continue translating ..
import json
from tqdm import tqdm

IS_CONTINUE = True
all_the_data = []

if os.path.exists(FILE_OUTPUT_PATH) and IS_CONTINUE:
    print(f"\nAlready exist file {os.path.basename(FILE_OUTPUT_PATH)}, continue translating ...")
    with open(FILE_OUTPUT_PATH) as fIn:
        all_the_data = json.load(fIn)

count = len(all_the_data)
for data in tqdm(dataset.select(range(len(all_the_data), len(dataset)))):

    text_en = translator.translate(data["text"], translate_single=False, batch_size=INITIAL_BATCH_SIZE)


    result_translated = {"id": data["id"],
                         "text": text_en
                         }

    all_the_data.append(result_translated)

    count += 1
    if not count % SAVE_PER_SAMPLES:
        print(f"\nWriting output to {os.path.basename(FILE_OUTPUT_PATH)} with {len(all_the_data)} samples ... ")
        with open(FILE_OUTPUT_PATH, "w") as fOut:
            json.dump(all_the_data, fOut, ensure_ascii=False, indent=2)
        print("Writing done")

    if not count % PUSH_TO_HF_PER_SAMPLES:
        api.upload_file(
            path_or_fileobj=FILE_OUTPUT_PATH,
            path_in_repo=FILE_NAME,
            repo_id=HF_REPO_PATH,
            repo_type="dataset")
        print(f"In Push to {os.path.basename(FILE_OUTPUT_PATH)} to HF ..")
        
print(f"\nWriting output to {os.path.basename(FILE_OUTPUT_PATH)} with {len(all_the_data)} samples ... ")
with open(FILE_OUTPUT_PATH, "w") as fOut:
    json.dump(all_the_data, fOut, ensure_ascii=False, indent=4)
print("Writing done")
