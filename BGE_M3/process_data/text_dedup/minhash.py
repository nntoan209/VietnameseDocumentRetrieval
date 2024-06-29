#!/usr/bin/env python
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import pickle
import random
from collections import defaultdict
from typing import Any
from typing import Callable
from utils.constant import HASH_CONFIG
import datasets
import numpy as np
from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm
from utils.custom import *
from utils import UnionFind
from utils import ngrams
from utils.add_args import add_io_args
from utils.add_args import add_meta_args
from utils.add_args import add_minhash_args
from utils.analysis import optimal_param
from utils.hashfunc import sha1_hash
from utils.hashfunc import xxh3_16hash
from utils.hashfunc import xxh3_32hash
from utils.timer import Timer
import pyarrow.parquet as pq
import pyarrow as pa
import json
SEED = 42
RNG = np.random.RandomState(SEED)

datasets.logging.set_verbosity_error()


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: list[tuple[int, int]],
    permutations: np.ndarray,
    hash_func: Callable,
    dtype: type,
    max_hash: np.uint,
    modulo_prime: np.uint,
) -> dict[str, Any]:
    """
    Calculate hash values for the content.

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    min_length : int
        The minimum length of the document in terms of tokens.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.
    hash_func : Callable
        The hash function to use.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.

    Examples
    --------
    >>> content = "hello world"
    >>> idx = 0
    >>> num_perm = 250
    >>> ngram_size = 1
    >>> hashranges = [(i, i + 25) for i in range(0, 250, 25)]
    >>> max_hash = np.uint32((1 << 32) - 1)
    >>> modulo_prime = np.uint32((1 << 32) - 5)
    >>> PERMUTATIONS = (RNG.randint(1, modulo_prime, size=num_perm),RNG.randint(0, modulo_prime, size=num_perm))
    >>> res = embed_func(content, idx, num_perm=num_perm, ngram_size=ngram_size, min_length=0, hashranges=hashranges,
    ... permutations=PERMUTATIONS, hash_func=xxh3_32hash,dtype=np.uint32, max_hash=max_hash, modulo_prime=modulo_prime)
    >>> len(res["__signatures__"])
    10
    >>> res["__id__"]
    0
    """
    # a, b are each np.ndarray arrays containing {num_perm} pairs of random numbers used for building new hashes
    # the formula is a * x(base hash of each shingle) + b
    a, b = permutations
    # split content on whitespace (NON_ALPHA regex), tokenize with ngrams(), and join these n-grams into a single space separated string.
    # we then convert to lower case and then bytestrings which is then hashed. Only unique hashed n-grams are left.
    tokens: set[bytes] = {
        bytes(" ".join(t).lower(), "utf-8")
        for t in ngrams(list(filter(None,NON_ALPHA.split(content.lower()))), ngram_size, min_length)
    }

    hashvalues: np.ndarray = np.array(
        [hash_func(token) for token in tokens], dtype=dtype
    ).reshape(len(tokens), 1)
    # Permute the hash values to produce new universal hashes
    # Element-wise multiplication with 'hashvalues' and a (non 0 random value) and then adding b
    # Then, take modulo 'MODULO_PRIME' and bitwise_and with 'MAX_HASH' to keep only the necessary bits.
    hashvalues = (hashvalues * a + b) % modulo_prime & max_hash
    # this part is where the name "min" of minhash comes from
    # this stacks all the hashes and then takes the minimum from each column
    masks: np.ndarray = np.full(shape=num_perm, dtype=dtype, fill_value=max_hash)
    hashvalues = np.vstack([hashvalues, masks]).min(axis=0)
    # Originally, byteswap was done for speed. Testing show it has a negligible impact
    # keeping  for backward compatibility, even though theoretically and empirically
    # it doesnt matter if it is there or not. github.com/ekzhu/datasketch/issues/114
    Hs: list[bytes] = [
        bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges
    ]
    return {"__signatures__": Hs, "__id__": idx}


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="minhash",
        description="Deduplicate text using minhash",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_minhash_args(parser)
    parser.add_argument(
        "--jaccard_thresh", type=float, default=None, help="Jaccard similarity threshold to use AFTER MinHashLSH"
    )
    parser.add_argument(
        "--filter_at_jaccard", type=float, default=None, help="Filter at jaccard thresh & then save to a file"
    )
    parser.add_argument(
        "--lsh_false_positive_thresh_weight", type=float, default=0.5, help="Weight for calculating optimal LSH params"
    )
    parser.add_argument(
        "--cluster_column", type=str, default="__cluster__", help="Column used to index clusters"
    )
    parser.add_argument(
        "--index_column", type=str, default="oid", help="Column used to index items"
    )
    # parser.add_argument(
    #     "--safe_throw", type=float, default=None, help="Throw all cluster at range safe_throw<x<jaccard_thresh. Only supported in PREVIOUS mode."
    # )
    parser.add_argument('--jaccard_mode', type=str,required=True,default="centroid")
    args = parser.parse_args()
    args.jaccard_mode=args.jaccard_mode.lower()
    # assert (args.safe_throw==None) or (args.safe_throw!=None and args.jaccard_thresh!=None and args.jaccard_mode=="previous")
    assert 0<=args.lsh_false_positive_thresh_weight<=1
    assert args.jaccard_mode in ["centroid","previous","full"]
    DTYPE, MAX_HASH, MODULO_PRIME = HASH_CONFIG.get(args.hash_bits, HASH_CONFIG[64])
    match args.hash_func:
        case "sha1":

            def hash_func(byte_data):
                return sha1_hash(byte_data, d=min(args.hash_bits, 32))
        case "xxh3":
            if args.hash_bits == 16:
                hash_func = xxh3_16hash
            else:
                hash_func = xxh3_32hash

    # for is originally used to reduce memory usage in MacOS but also ensures that the Union Find data structure
    # is not copied to child processes as long as it is not modified.
    mp.set_start_method("fork", force=True)

    uf = UnionFind()
    timer = Timer()

    if args.b is not None and args.r is not None:
        B, R = args.b, args.r
    else:
        # Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
        # of probabilities of false positive and false negative, taken from datasketch.
        # You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.
        # The following assumes a "perfect hash". using 16 bit hashes might challenge this assumption
        # lower precision dtype will cause more collisions, so higher false_positives and less false negatives.
        # Both effects move the result towards more documents being considered duplicates.
        B, R = optimal_param(
            args.threshold,
            args.num_perm,
            false_positive_weight=args.lsh_false_positive_thresh_weight,
            false_negative_weight=1-args.lsh_false_positive_thresh_weight,
        )

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    HASH_TABLES: list[dict[int, set]] = [defaultdict(set) for _ in range(B)]

    with timer("Total"):
        with timer("Loading"):
            if args.local:
                ds = load_from_disk(args.path)
            else:
                ds = load_dataset(
                    path=args.path,
                    name=args.name,
                    data_dir=args.data_dir,
                    data_files=args.data_files,
                    split=args.split,
                    revision=args.revision,
                    cache_dir=args.cache_dir,
                    num_proc=args.num_proc,
                    token=args.use_auth_token,
                )
            ds = ds.filter(
                lambda x: len(NON_ALPHA.split(x[args.column].lower()))
                >= args.min_length,
                num_proc=args.num_proc,
            )
            if args.index_column not in list(ds[0].keys()):
                print(f"ds indexing column {args.index_column} not found",list(ds[0].keys()))
                ds=ds.map(lambda _,id:{args.index_column:id},with_indices=True,num_proc=args.num_proc,desc=f"Adding unique index column {args.index_column}")
            else:
                print("confirmed that indexing column is found",args.index_column)
            i=0
            print("trying cluster_col",args.cluster_column)
            while args.cluster_column in list(ds[0].keys()):
                print("trying cluster_col",args.cluster_column)
                args.cluster_column=f"__cluster__{i}"
                i+=1
            print("setting new cluster indexing column to", args.cluster_column)
            del i
            if args.jaccard_mode!="previous":
                print("sorting oids...",end=" ")
                ds=ds.sort(column_names=args.index_column)
                print("done")
        LEN_DATASET = len(ds)
        # for minhash, we need to make a lot of hashes(=num_perms).
        # In many previous implementations, this is achieved through a method described in
        # `Universal classes of hash functions` https://doi.org/10.1016/0022-0000(79)90044-8
        # There we start with a know good hash x (=hash_func) and permutate it as the following:
        # `new_hash = (a * x + b) mod prime mod max_hash` we need one a (!=0), b pair per new hash
        # the following produces these a, b pairs
        PERMUTATIONS: tuple[np.ndarray, np.ndarray] = (
            RNG.randint(
                1, MODULO_PRIME, size=(args.num_perm,), dtype=DTYPE
            ),  # a is a multiplier so should not be 0
            RNG.randint(0, MODULO_PRIME, size=(args.num_perm,), dtype=DTYPE),  # b
        )
        with timer("MinHashing"):
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={
                    "num_perm": args.num_perm,
                    "hashranges": HASH_RANGES,
                    "ngram_size": args.ngram,
                    "min_length": args.min_length,
                    "permutations": PERMUTATIONS,
                    "hash_func": hash_func,
                    "dtype": DTYPE,
                    "max_hash": MAX_HASH,
                    "modulo_prime": MODULO_PRIME,
                },
                input_columns=[args.column],
                remove_columns=ds.column_names,
                num_proc=args.num_proc,
                with_indices=True,
                desc="Fingerprinting...",
            )
            LEN_EMBEDDED = len(embedded)
            if(args.batch_size==0):
                args.batch_size=LEN_EMBEDDED
            NUM_SHARDS = np.ceil(LEN_EMBEDDED / args.batch_size).astype(int)

        with timer("Clustering"):
            for i in tqdm(
                range(0, NUM_SHARDS),
                dynamic_ncols=True,
                desc="Iterating MinHashes...", 
            ):
                embedded_shard = embedded.shard(
                    num_shards=NUM_SHARDS,
                    index=i,
                    contiguous=True,
                    writer_batch_size=args.batch_size,
                )
                for key, Hs in zip(
                    embedded_shard["__id__"], embedded_shard["__signatures__"]
                ):
                    for i, H in enumerate(Hs):
                        HASH_TABLES[i][H].add(key)

            for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
                # cluster: Set[int]
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        uf.union(x, idx)
        with timer("Finding clusters"):
            # gc manipulations to ensure that uf object is not unneccessarily copied across processes
            gc.freeze()
            gc.disable()
            ds = ds.map(
                function=lambda _, idx: {args.cluster_column: uf.find(idx)},
                with_indices=True,
                num_proc=args.num_proc,
                new_fingerprint=str(random.getrandbits(128)),
                desc="Finding clusters...",
            )
            gc.enable()
            gc.collect()
        import copy
        if args.jaccard_thresh or (args.jaccard_thresh==0):
            jaccard_thresh_arr=[args.jaccard_thresh]
        else:
            jaccard_thresh_arr=[0.5,0.7,0.8]
        if args.filter_at_jaccard:
            jaccard_thresh_arr.append(args.filter_at_jaccard)
        jaccard_thresh_arr=set(jaccard_thresh_arr)
        if args.jaccard_mode=="previous":
            for jaccard_thresh in jaccard_thresh_arr:
                jaccard_ds=copy.deepcopy(ds).sort(args.cluster_column)
                num_cluster_cur=0
                cluster_arr=[]
                cluster_pre=[]
                with timer("Filtering"):
                    for item in (pbar:=tqdm(jaccard_ds,desc=f"Comparing candidates at {jaccard_thresh}....")):
                        item["textset"]=custom_tokenize_words_no_sep_with_num(item[args.column])
                        if item[args.cluster_column]==num_cluster_cur:
                            cluster_arr.append(item)
                        else:
                            if cluster_arr:
                                for idx in range(0,len(cluster_arr)):
                                    pbar.set_description(f"Comparing candidates at {jaccard_thresh}: {idx}/{len(cluster_arr)}....")
                                    cluster_arr[idx][args.cluster_column]=cluster_arr[idx][args.index_column]
                                    for preid in range(0,idx):
                                        _score=custom_jaccard(cluster_arr[idx]["textset"],cluster_arr[preid]["textset"])
                                        if _score>jaccard_thresh:
                                            cluster_arr[idx][args.cluster_column]=cluster_arr[preid][args.cluster_column]
                                            break
                            pbar.set_description(f"Comparing candidates at {jaccard_thresh}....")
                            cluster_pre.extend(cluster_arr)
                            num_cluster_cur=item[args.cluster_column]
                            cluster_arr=[item]
                    # Last cluster is left behind after the for loop, so we append it here
                    cluster_pre.extend(cluster_arr)
                    # Remove textset
                    for idx,_ in tqdm(enumerate(cluster_pre),desc="Removing temporary fields"):
                        try:
                            del cluster_pre[idx]["textset"]
                        except:
                            print("Exception at textset")
                with timer("Saving"):
                    pq.write_table(pa.Table.from_pylist(cluster_pre),args.output+f"_reindexed({args.threshold}|{jaccard_thresh}|PREVIOUS)")
                    if jaccard_thresh==args.filter_at_jaccard:
                        raise NotImplemented
                        final_data = harshreindexed_ds.filter(
                            function=lambda record, idx: record[args.cluster_column] == idx,
                            with_indices=True,
                            num_proc=args.num_proc,
                            desc="Filtering clusters...",
                        )
                        pq.write_table(final_data.data.table,args.output+f"_filtered({args.threshold}|{jaccard_thresh}|PREVIOUS)")
                    # final_data = final_data.remove_columns([args.cluster_column])
                    # final_data.to_json(args.output)
                    # jaccard_ds.to_json(args.output+f"_reindexed({args.threshold}|{jaccard_thresh})")
                    if args.debug:
                        with open(os.path.join(args.output, "uf.pkl"), "wb") as f:
                            pickle.dump(uf, f, protocol=pickle.HIGHEST_PROTOCOL)   
        elif args.jaccard_mode=="full":
            for jaccard_thresh in jaccard_thresh_arr:
                jaccard_ds=copy.deepcopy(ds).sort(args.cluster_column)
                num_cluster_cur=0
                cluster_arr=[]
                cluster_pre=[]
                with timer("Filtering"):
                    with open(args.output+f"_full_jaccard_scores({args.threshold}|{jaccard_thresh}|FULL)","w+") as of:
                        for item in (pbar:=tqdm(jaccard_ds,desc=f"Comparing candidates at {jaccard_thresh}....")):
                            item["textset"]=custom_tokenize_words_no_sep_with_num(item[args.column])
                            if item[args.cluster_column]==num_cluster_cur:
                                cluster_arr.append(item)
                            else:
                                if cluster_arr:
                                    uf=UnionFind()
                                    candidate_score=[]
                                    # cluster_arr[0][args.cluster_column]=cluster_arr[0][args.index_column]
                                    for idx in range(0,len(cluster_arr)):
                                        pbar.set_description(f"Comparing candidates at {jaccard_thresh}: {idx}/{len(cluster_arr)}....")
                                        flag=True
                                        for idy in range(idx+1,len(cluster_arr)):
                                            j_score=custom_jaccard(cluster_arr[idx]["textset"],cluster_arr[idy]["textset"])
                                            if cluster_arr[idx][args.index_column]<cluster_arr[idy][args.index_column]:
                                                candidate_score.append((cluster_arr[idx][args.index_column],cluster_arr[idy][args.index_column],j_score))
                                            else:
                                                candidate_score.append((cluster_arr[idy][args.index_column],cluster_arr[idx][args.index_column],j_score))

                                            if cluster_arr[idx][args.cluster_column]==cluster_arr[idy][args.cluster_column]:
                                                flag=False
                                            elif j_score>jaccard_thresh:
                                                flag=False
                                                uf.union(cluster_arr[idy][args.index_column],cluster_arr[idy][args.index_column])
                                                # ncluster=min(cluster_arr[idy][args.cluster_column],cluster_arr[idx][args.cluster_column])
                                                # cluster_arr[idx][args.cluster_column]=ncluster
                                                # cluster_arr[idy][args.cluster_column]=ncluster
                                        if flag:
                                            cluster_arr[idx][args.cluster_column]=cluster_arr[idx][args.index_column]
                                    for idx in range(0,len(cluster_arr)):
                                        cluster_arr[idx][args.cluster_column]=uf.find(cluster_arr[idx][args.index_column])
                                    pbar.set_description(f"Comparing candidates at {jaccard_thresh}....")
                                    of.write(json.dumps({num_cluster_cur:candidate_score},ensure_ascii=False))
                                    of.write("\n")
                                cluster_pre.extend(cluster_arr)
                                num_cluster_cur=item[args.cluster_column]
                                cluster_arr=[item]
                    # Last cluster is left behind after the for loop, so we append it here
                    cluster_pre.extend(cluster_arr)
                    # Remove textset
                    for idx,_ in tqdm(enumerate(cluster_pre),desc="Removing temporary fields"):
                        try:
                            del cluster_pre[idx]["textset"]
                        except:
                            print("Exception at textset")
                with timer("Saving"):
                    pq.write_table(pa.Table.from_pylist(cluster_pre),args.output+f"_reindexed({args.threshold}|{jaccard_thresh}|FULL)")
                    if jaccard_thresh==args.filter_at_jaccard:
                        raise NotImplemented
                        final_data = harshreindexed_ds.filter(
                            function=lambda record, idx: record[args.cluster_column] == idx,
                            with_indices=True,
                            num_proc=args.num_proc,
                            desc="Filtering clusters...",
                        )
                        pq.write_table(final_data.data.table,args.output+f"_filtered({args.threshold}|{jaccard_thresh}|FULL)")
                    # final_data = final_data.remove_columns([args.cluster_column])
                    # final_data.to_json(args.output)
                    # jaccard_ds.to_json(args.output+f"_reindexed({args.threshold}|{jaccard_thresh})")
                    if args.debug:
                        with open(os.path.join(args.output, "uf.pkl"), "wb") as f:
                            pickle.dump(uf, f, protocol=pickle.HIGHEST_PROTOCOL)  
        elif args.jaccard_mode=="centroid":
            ds = ds.map(
                function=lambda ex,*,ds,col: {"__score__": custom_distance(ds[ex[args.cluster_column]][col],ex[col]),"oparent":ds[ex[args.cluster_column]][args.index_column]},
                fn_kwargs={
                    "ds":ds,
                    "col":args.column
                },
                num_proc=args.num_proc,
                # new_fingerprint=str(random.getrandbits(128)),
                desc="Reevaluating with distance func",
            )
            for jaccard_thresh in jaccard_thresh_arr:
                with timer("Filtering"):
                    jaccard_ds=copy.deepcopy(ds)

                    # This is where the deduplication happens
                    # Since there is no easy groupby in datasets
                    # I will use this simple filter for now
                    
                    jaccard_ds=jaccard_ds.map(
                        function=lambda ex, idx,*,thresh: {args.cluster_column: ex[args.cluster_column] if ex["__score__"]>thresh else idx},
                        fn_kwargs={
                            "thresh":jaccard_thresh
                        },
                        with_indices=True,
                        num_proc=args.num_proc,
                        # new_fingerprint=str(random.getrandbits(128)),
                        desc=f"Reevaluating with jaccard thresh {jaccard_thresh}",
                    )
                with timer("Saving"):
                    pq.write_table(jaccard_ds.data.table,args.output+f"_reindexed({args.threshold}|{jaccard_thresh}|CENTROID)")
                    if jaccard_thresh==args.filter_at_jaccard:
                        final_data = jaccard_ds.filter(
                            function=lambda record, idx: record[args.cluster_column] == idx,
                            with_indices=True,
                            num_proc=args.num_proc,
                            desc="Filtering clusters...",
                        )
                        pq.write_table(final_data.data.table,args.output+f"_filtered({args.threshold}|{jaccard_thresh}|CENTROID)")
                    # final_data = final_data.remove_columns([args.cluster_column])
                    # final_data.to_json(args.output)
                    # jaccard_ds.to_json(args.output+f"_reindexed({args.threshold}|{jaccard_thresh})")
                    if args.debug:
                        with open(os.path.join(args.output, "uf.pkl"), "wb") as f:
                            pickle.dump(uf, f, protocol=pickle.HIGHEST_PROTOCOL)

    PAD = 32
    for k, v in timer.elapsed_times.items():
        print(f"{k:<{PAD}}: {v:.2f}s")

    print(f"{'Before':<{PAD}}: {len(ds)}")
    # print(f"{'After':<{PAD}}: {len(final_data)}")
