import sys
import json
import torch
import argparse

from typing import List, Dict, Any
from transformers import AutoModel, AutoTokenizer

sys.path.append('.')

def tokenize_with_specb(texts, is_query):
    # Tokenize without padding
    batch_tokens = tokenizer(texts, padding=False, truncation=True)   
    # Add special brackets & pay attention to them
    for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
        if is_query:
            seq.insert(0, SPECB_QUE_BOS)
            seq.append(SPECB_QUE_EOS)
        else:
            seq.insert(0, SPECB_DOC_BOS)
            seq.append(SPECB_DOC_EOS)
        att.insert(0, 1)
        att.append(1)
    # Add padding
    batch_tokens = tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
    return batch_tokens

def get_weightedmean_embedding(batch_tokens, model):
    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state
        # 新版的transformers可能不支持las_hidden_state
        # 如果29行代码报错，可以用下面的代码代替
        # last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).hidden_states[-1]


    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )

    # Get attn mask of shape [bs, seq_len, hid_dim]
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask

    return embeddings

def compute_recall(pred_list: List, gold_list: List):
    correct_count = 0
    for p in pred_list:
        if p in gold_list:
            correct_count += 1
    rec = 1.0*correct_count/len(gold_list)
    return rec




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--docs_file", type=str, help="data path")
    parser.add_argument("--doc_embeddings_file", type=str, help="dump path")
    parser.add_argument("--unit", type=str, default="table",help="dump path")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")

    args = parser.parse_args()

    # model_path = f"/share/home/chewanxiang/xlzhang/retrival/SGPT/model/SGPT-5.8B-weightedmean-msmarco-specb-bitfit"
    # docs_file = "./data/Spider/schema_166.txt"
    # doc_embeddings_file = "./data/Spider/schema166_embeddings_58.json"

    assert args.unit in ["DB", "table"]

    model_path = args.model_path
    docs_file = args.docs_file
    doc_embeddings_file = args.doc_embeddings_file
    unit = args.unit
    # tables_file = "./data/Spider/tables.json"

    sim = []
    sort_sim = []
    doc_embeddings_json = []
    docs = []

    # Get our models - The package will take care of downloading the models automatically
    # For best performance: Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    # Deactivate Dropout (There is no dropout in the above models so it makes no difference here but other SGPT models may have dropout)
    model.eval()
    docs_type = docs_file.split("/")[-1].split(".")[-1]
    # assert docs_type in ["json", "txt"]

    # if docs_type == "json":
    with open(docs_file, "r", encoding='utf-8') as f:
        orginal_tables = json.load(f)
    if docs_type == "json":
        for d in orginal_tables:
            tables_schema = d["schema"]
            docs += tables_schema
    elif docs_type == "txt":
        with open(docs_file, "r", encoding='utf-8') as f:
            schema_string = f.read()
        docs = schema_string.split("\n")

    SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
    SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]

    SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
    SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]

    doc_embeddings = get_weightedmean_embedding(tokenize_with_specb(docs, is_query=False), model)

    print(doc_embeddings[:5])

    for di, d in enumerate(doc_embeddings):
        doc_data = {}
        doc_data.setdefault("pred_schema", docs[di])
        doc_data.setdefault("embedding", d.tolist())
        doc_embeddings_json.append(doc_data)

    with open(doc_embeddings_file, 'w', encoding='utf-8') as f:
        json.dump(doc_embeddings_json, f, indent=4, ensure_ascii=False)

    print("Done!")

   