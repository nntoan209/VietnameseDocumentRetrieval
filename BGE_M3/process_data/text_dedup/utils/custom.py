import re
from utils import UnionFind
import datasets
NON_ALPHA = re.compile(r"[^\w\d]", re.UNICODE)
def custom_tokenize_by_word(text:str)->set:
    return set(list(filter(None,re.split(NON_ALPHA,text.lower()))))

def custom_tokenize_words_no_sep_with_num(text:str)->set:
    return set(re.findall(r"\w+ *\d*",text,re.UNICODE))
def custom_jaccard(a:set,b:set):
    inter= len(a.intersection(b))
    if inter==0:
        return 0
    return  float(inter)/float(len(a.union(b)))
def custom_containment(a:set,b:set):
    inter= len(a.intersection(b))
    if inter==0:
        return 0
    return  float(inter)/float(min(len(a),len(b)))
def custom_distance(a:str,b:str,tokenize:callable=custom_tokenize_words_no_sep_with_num)->float:
    return custom_jaccard(tokenize(a),tokenize(b))
def custom_cluster(example,id:int,*,uf:UnionFind,ds:datasets.Dataset,column:str,jaccard_thresh:float):
    parentID=uf.find(id)
    if parentID==id:
        example["__cluster__"]=id
    else:
        parent=ds[parentID]
        if custom_distance(parent[column],example[column]) >jaccard_thresh:
            example["__cluster__"]=parentID
        else:
            example["__cluster__"]=id
    return example