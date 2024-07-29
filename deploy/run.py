import json
from typing import Any
import torch

import gradio as gr
import numpy as np
from sentence_transformers import util
from BGE_M3.src.utils import BGEM3FlagModel

MODEL_PATH = "saved_models/bgem3_synthetic_20240703/checkpoint-35665"

model = BGEM3FlagModel(
    model_name_or_path=MODEL_PATH,
    pooling_method='cls',
    normalize_embeddings=True,
    use_fp16=True,
    device="cuda:0",
)

sample_queries = {
    "law": [
        "Thực phẩm không bảo đảm an toàn bị thu hồi theo những hình thức nào?",
        "Ban quản lý các dự án Lâm nghiệp đặt trụ sở ở đâu ?",
        "Việc xét duyệt cấp tín dụng của tổ chức tín dụng được thực hiện như thế nào?",
        "Người đang học nghề , người tập nghề có được hưởng ngày nghỉ hằng năm không?",
        "Giảng viên đại học có yêu cầu khả năng nghiên cứu khoa học không ?"
    ],
    "zalo_legal": [
        "Khu vực ô nhiễm bom mìn vật nổ là gì?",
        "Nhiệm vụ khảo sát xây dựng do ai lập?",
        "Gói thầu mua sắm hàng hóa 14 tỷ có được xem là gói thầu có quy mô nhỏ hay không?",
        "Mức thời gian được giảm thi hành án đối với người bị phạt cải tạo không giam giữ được quy định như thế nào?",
        "Quay đầu xe trên cầu thì bị phạt bao nhiêu?"
    ],
    "zalo_qa": [
        "Quang Hải giành được chức vô địch U21 quốc gia năm bao nhiêu tuổi?",
        "Bác ra đi tìm đường cứu nước năm bao nhiêu?",
        "Trụ sở chính của Liên Hiệp Quốc đặt ở đâu?",
        "Dòng sông nào dài nhất thế giới?",
        "Thủ đô của Canada hiện nay là thành phố nào?"
    ]
}

def update_sample_queries(dataset):
    return "\n".join(sample_queries[dataset])

# Load corpus
with open("data/eval/law/filtered_corpus.json", encoding='utf-8') as f:
    law_eval_corpus = json.load(f)
    law_eval_pids = list(law_eval_corpus.keys())
    
with open("data/eval/zalo_legal/filtered_corpus_with_title.json", encoding='utf-8') as f:
    zalo_legal_corpus = json.load(f)
    zalo_legal_pids = list(zalo_legal_corpus.keys())

with open("data/eval/zalo_qa/filtered_corpus.json", encoding='utf-8') as f:
    zalo_qa_corpus = json.load(f)
    zalo_qa_pids = list(zalo_qa_corpus.keys())

# Load corpus embedding
law_eval_corpus_embeddings = np.load(f"saved_embeddings/law/{MODEL_PATH}/corpus_embedding.npy")
law_eval_corpus_embeddings_dense = torch.from_numpy(law_eval_corpus_embeddings).to(torch.float32)

zalo_legal_corpus_embeddings = np.load(f"saved_embeddings/zalo_legal/{MODEL_PATH}/corpus_embedding.npy")
zalo_legal_corpus_embeddings_dense = torch.from_numpy(zalo_legal_corpus_embeddings).to(torch.float32)

zalo_qa_corpus_embeddings = np.load(f"saved_embeddings/zalo_qa/{MODEL_PATH}/corpus_embedding.npy")
zalo_qa_corpus_embeddings_dense = torch.from_numpy(zalo_qa_corpus_embeddings).to(torch.float32)

def search(query: str, dataset: str, reranking: bool = False, top_k: int = 100):
    
    if dataset == "law":
        corpus = law_eval_corpus
        pids = law_eval_pids
        corpus_embeddings_dense = law_eval_corpus_embeddings_dense
    elif dataset == "zalo_legal":
        corpus = zalo_legal_corpus
        pids = zalo_legal_pids
        corpus_embeddings_dense = zalo_legal_corpus_embeddings_dense
    elif dataset == "zalo_qa":
        corpus = zalo_qa_corpus
        pids = zalo_qa_pids
        corpus_embeddings_dense = zalo_qa_corpus_embeddings_dense    
    
    print()
    ans: list[str] = []
    
    ##### Sematic Search #####
    question_embedding = model.encode(query,
                                      batch_size=1,
                                      max_length=128,
                                      show_progress_bar=False)
    question_embedding_dense = torch.from_numpy(question_embedding['dense_vecs']).to(torch.float32)
    
    results_dense_search = util.semantic_search(question_embedding_dense, corpus_embeddings_dense, top_k=top_k)[0]
    only_pids_results = []
    for answer in results_dense_search:
        ans.append(pids[answer['corpus_id']])

    ##### Re-Ranking #####
    if reranking:
        colbert_scores_for_qid = model.compute_batch_colbert_score(query=query,
                                                                passages=[corpus[pid] for pid in ans],
                                                                batch_size=2,
                                                                max_query_length=128,
                                                                max_passage_length=2048)
        
        pids_with_colbert_scores = list(zip(ans, colbert_scores_for_qid))
        rerank_pids_with_colbert_scores = sorted(pids_with_colbert_scores, key=lambda x: x[1], reverse=True)
        ans = [answer[0] for answer in rerank_pids_with_colbert_scores]

    return corpus[ans[0]], corpus[ans[1]], corpus[ans[2]], corpus[ans[3]], corpus[ans[4]]


with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(lines=1, placeholder=None, label="Enter your query here")
            reranking_checkbox = gr.Checkbox(label="Enable reranking")
            dataset = gr.Dropdown(choices=["law", "zalo_legal", "zalo_qa"], value="law", label="Choose dataset")
            sample_queries_text = gr.Textbox(
                value=update_sample_queries("law"),
                label="Example Queries",
                lines=5,
                interactive=False
            )
            search_btn = gr.Button("Search") 
            dataset.change(update_sample_queries, inputs=[dataset], outputs=[sample_queries_text])
            
        with gr.Column():
            out1 = gr.Textbox(type="text", label="Search result 1")
            out2 = gr.Textbox(type="text", label="Search result 2")
            out3 = gr.Textbox(type="text", label="Search result 3")
            out4 = gr.Textbox(type="text", label="Search result 4")
            out5 = gr.Textbox(type="text", label="Search result 5")

    search_btn.click(fn=search, inputs=[inp, dataset, reranking_checkbox], outputs=[out1, out2, out3, out4, out5])

iface.launch()
