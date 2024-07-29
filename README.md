# Legal Retrieval

## 1. Install requirements
```
pip install -e .
```

## 2. Training data preparation

Prepare a **.jsonl** file, each line is in the form 
```{"query": str, "pos": List[str], "__context_cluster__": List[str]}``` where ```"pos"``` is the list of positive passages and ```"__context_cluster__"``` is the list of their relative clusters in the deduplicated corpus. 

### Hard negative mining

```
#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

python3 BGE_M3/src/hn_mine.py \
    --model_name_or_path BAAI/bge-m3 \
    --input_file data/sft/train_data.jsonl \
    --corpus_file data/eval/law/filtered_corpus.json \
    --output_file data/sft/train_data_minedHN.jsonl \
    --range_for_sampling 5-30 \
    --negative_number 10
```

Change the arguments and run
```
bash BGE_M3/scripts/process_data/hardneg_mining.sh
```

### Split data by length

```
#!/bin/bash

python3 BGE_M3/process_data/split_by_length.py \
    --input_path data/sft/train_data_minedHN.jsonl \
    --output_dir data/sft/train_data_minedHN_splitted \
    --cache_dir ./cache \
    --log_name .split_log \
    --length_list 0 512 1024 2048 \
    --model_name_or_path BAAI/bge-m3 \
    --num_proc 32 \
    "$@"
```

Change the arguments and run
```
bash BGE_M3/script/process_data/split_by_length.sh
```

Reference datasets:
- [Train Dataset](https://huggingface.co/datasets/nntoan209/TrainData-CrossLingual/tree/main) (All 4 datasets combined)
- [Law Eval Dataset](https://huggingface.co/datasets/nntoan209/LawEval-CrossLingual)
- [Zalo Legal 2021 Eval Dataset](nntoan209/ZaloLegal-CrossLingual)
- [Zalo QA Eval Dataset](https://huggingface.co/datasets/nntoan209/ZaloQA-CrossLingual)

Script to download eval dataset:
```
mkdir data/eval
cd data/eval
huggingface-cli download nntoan209/LawEval --repo-type dataset --local-dir .
```
## 3. Finetune

### Vietnamese only:

```
bash BGE_M3/script/train/finetune_unified.sh
```
Some important arguments:
- ```train_data```: paths to the folder containing the splitted dataset, seperated by a space.
- ```mix_dataset```: whether to mix all the datasets for training when having multiple datasets. If set to False, each batch will contain the data from one dataset only.
- ```deepspeed```: path to the deepspeed config file, use to reduce the GPU memory when training. Refer to [Deepspeed examples](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) for some example configurations.
- ```per_device_train_batch_size```: this arguments is the default batch size to use when the batch size for each length group is not set in the [dataset.py](dataset.py) file.
- ```temperature```: the temperature used to calculate the loss function. When this arguments is set to lower, it increases the weight of the most similar pairs and the penalty for dissimilar pairs, and vice versa.
- ```negative_cross_device```: Whether to use cross-batch negative between devices (gather the embeddings from multiple devices). This will increase the number of negatives for each query.
- ```unified_finetuning``` and ```use_self_distill```: set to ```True``` to enable unified finetuning (finetune all 3 modes at the same time).
- ```self_distill_start_step```: the number of steps before starting to unified finetune.
- ```resume_from_checkpoint```: set to True to return training from the latest checkpoint (in case of GPU interrupted while training).

For more arguments, refer to [TrainingArguments](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments) of Huggingface

### Cross-lingual:
```
bash BGE_M3/script/train/train_crosslingual.sh
```
Most of the arguments are the same when training for Vietnamese only. The differences are in the dataset file. For cross-lingual training, the ```--train_data``` file contains the ids of the queries and documents, and there are 4 additional files: ```merged_queries_vi```, ```merged_queries_en```, ```merged_corpus_vi``` and ```merged_queries_en``` which are json files, the keys are the ids, and the values are the texts.


### Cross-encoder Reranker:
```
bash BGE_M3/script/train/finetune_reranker.sh
```
Additional argument: ```--use_bce_loss```: whether to use BCE loss instead of Cross Entropy loss as the original model.

## 4. Evaluation

### Rerank using Multi-vector (ColBERT) mode of BGE-M3

```
bash BGE_M3/script/eval/eval_law.sh
```

Some important arguments: 
- ```model_savedir```: path to the Huggingface model or the local saved model when training.
- ```save_corpus_embeddings```: whether to save the dense embeddings of the corpus (for later uses if any).
- ```corpus_embeddings```: path to the saved corpus embeddings.
- ```bm25_hybrid```: use BM25 with dense mode for hybrid retrieval
- ```bm25_k1```, ```bm25_b```, ```bm25_weight``` and ```rrf_k```: parameters for hybrid retrieval.
- ```colbert_rerank```: use Colbert mode for rerank.

We also provided the script to run the evaluation for cross-lingual retrieval, which has the same arguments as aboved.

```
bash BGE_M3/script/eval/run_eval.sh
```

### Rerank using a Cross-encoder

```
bash BGE_M3/script/eval/eval_law_ce.sh
```

Additional arguments:
- ```ce_rerank```: whether to use a cross-encoder for reranking
- ```ce_rerank_model```: path to the Huggingface model or the local saved model when training.