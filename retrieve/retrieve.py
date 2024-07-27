import sys
import math
import json
import torch
import argparse

from copy import deepcopy
from typing import List, Dict
from scipy.spatial.distance import cosine
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
    batch_tokens = tokenizer.pad(
        batch_tokens, padding=True, return_tensors="pt")
    return batch_tokens


def get_weightedmean_embedding(batch_tokens, model):
    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(
            **batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state
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
    sum_embeddings = torch.sum(
        last_hidden_state * input_mask_expanded * weights, dim=1)
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

def remove_dbname(schema: str):
    if '.' in schema:
        return "".join([t0.split(".")[1] if "." in t0 else t0 for t0 in schema.split("\\n")])
    else:
        return schema
    
def preprocess(utte: str):
    if "There is no" in utte or "None of the given tables" in utte or "No additional tables" in utte or "No completion needed" in utte:
        return ""
    return remove_dbname(utte)

def remove_dbname_list(schemas: List):
    for i, s in enumerate(schemas):
        schemas[i] = remove_dbname(s)
    
    return schemas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="llm path")
    parser.add_argument("--top_k", type=int, nargs="+")
    parser.add_argument("--queries_file", type=str, help="data path")
    parser.add_argument("--embedding_file", type=str, help="data size")
    parser.add_argument("--last_retrieved_file", type=str, default="", help="type of agent")
    parser.add_argument("--retrieved_file", type=str, help="type of agent")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    args = parser.parse_args()

    top_k = args.top_k
    model_path = args.model_path
    queries_file = args.queries_file
    # docs_file = args.schema_file
    doc_embeddings_file = args.embedding_file
    retrieved_docs_file = args.retrieved_file

    sim = []
    sort_sim = []
    doc_embeddings_json = []
    queries: List[List[str]] = []
    # docs = []
    retrieved_data = []

    # print(method)

    # Get our models - The package will take care of downloading the models automatically
    # For best performance: Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    # Deactivate Dropout (There is no dropout in the above models so it makes no difference here but other SGPT models may have dropout)
    model.eval()

    with open(queries_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = [d["utterance"].split('\n') for d in data]

    
    assert len(queries) == len(data)
    with open(doc_embeddings_file, "r", encoding='utf-8') as f:
        original_docs = json.load(f)

    SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
    SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]

    SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
    SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]

    # TODO

    idx_map = []
    for qi, q in enumerate(queries):
        idx_map.extend([(qi, di) for di in q])
    
    batch_size = 256
    batch_num = math.ceil(len(idx_map)/batch_size)
    embeddings_list = []
    for b in range(batch_num):
        query_embeddings0 = get_weightedmean_embedding(
            tokenize_with_specb([x[1] for x in idx_map[b*batch_size: min((b+1)*batch_size, len(idx_map))]], is_query=True), model)
        print(len(query_embeddings0))
        embeddings_list.append(query_embeddings0)
    
    query_embeddings = torch.cat(embeddings_list, dim=0)
    print(len(query_embeddings))

    # query_embeddings = get_weightedmean_embedding(
    #     tokenize_with_specb([x[1] for x in idx_map], is_query=True), model)
    # doc_embeddings = get_weightedmean_embedding(tokenize_with_specb(docs, is_query=False), model)

    print(query_embeddings[-2:])
    # print(doc_embeddings[:5])

    query_embeddings = [[x for i, x in enumerate(
        query_embeddings) if idx_map[i][0] == qi] for qi in range(len(queries))]
    print(len(query_embeddings))

    doc_embeddings = []
    docs = [x["pred_schema"] for x in original_docs]

    for ed in original_docs:
        tensor_data = ed.get("embedding")
        if tensor_data is not None:
            tensor = torch.tensor(tensor_data)
            doc_embeddings.append(tensor)

    if args.last_retrieved_file:
        schema_map = {}
        with open(args.last_retrieved_file, 'r', encoding='utf-8') as f:
            last_retrieved_data = json.load(f)
        for di, d in enumerate(original_docs):
            schema_map[d["pred_schema"]] = int(di)

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    def calculate_single_similarity(qe, de):
        sim.append([])
        # print(len(qe))
        for qi, q in enumerate(qe):
            # sort_sim.append([])
            for di, d in enumerate(de):
                cosine_sim = 1 - cosine(q, d)
                sim[-1].append((di, cosine_sim))
            # assert len(doc_embeddings) == len(sim[qi])
        sort_qi = sorted(sim[-1], key=lambda x: (x[1]), reverse=True)
        qi_sort_sim = []
        record_doc = []
        for sqi in sort_qi:
            if len(qi_sort_sim) == max(args.top_k):
                break
            if sqi[0] not in record_doc:
                qi_sort_sim.append(sqi)
                record_doc.append(sqi[0])
            # sort_sim[-1].append(sqi)
        sort_sim.append(qi_sort_sim)


    if not args.last_retrieved_file:
        for qi, q in enumerate(query_embeddings):
            calculate_single_similarity(q, doc_embeddings)
    else:
        for qi, q in enumerate(query_embeddings):
            calculate_single_similarity(q, [doc_embeddings[schema_map[x]] for x in [ret["schema"] for ret in last_retrieved_data[qi]["retrieved"]]])
            assert len([doc_embeddings[schema_map[x]] for x in [ret["schema"] for ret in last_retrieved_data[qi]["retrieved"]]]) == max(args.top_k)
            # print([doc_embeddings[schema_map[x]] for x in [ret["schema"] for ret in last_retrieved_data[qi]["retrieved"]]][:1])
            # print([original_docs[schema_map[x]]["pred_schema"] for x in [ret["schema"] for ret in last_retrieved_data[qi]["retrieved"]]][:5])
            # print(qi)
    recall = 0.0

    total_recall = {k: 0.0 for k in top_k}
    for qi, q in enumerate(queries):
        user_question = data[qi]["utterance"]
        inputt = q
        llm_pred_schema = q
        retrieved_docs = []
        pred_docs = []
        if args.last_retrieved_file:
            org_docs = [x["pred_schema"] for x in original_docs]
            docs = [org_docs[schema_map[x]] for x in [ret["schema"] for ret in last_retrieved_data[qi]["retrieved"]]]
        if data[qi].get("utterance_org") is not None:
            if isinstance(data[qi]["utterance_org"], str):
                utterance_org = [data[qi]["utterance_org"]]
            elif isinstance(data[qi]["utterance_org"], list):
                utterance_org = data[qi]["utterance_org"]
        else:
            utterance_org = []
        # utterance_org = [data[qi].get("utterance_org", "")]
        # utterance_org = [x for x in utterance_org if len(x) > 0]
        # print(len(sort_sim[qi]))
        for rank, rd in enumerate(sort_sim[qi]):
            qi_ret = {}
            qi_ret.setdefault("rank", rank)
            qi_ret.setdefault("schema", docs[rd[0]])
            pred_docs.append(docs[rd[0]])
            qi_ret.setdefault("similarity", rd[1])
            retrieved_docs.append(qi_ret)
        assert len(retrieved_docs) > 0
        gold_docs = data[qi]["rel_schema"]
        selected_database = data[qi].get("selected_database", [])
        temp_total_recall = {}
        for k in top_k:
            qi_recall = compute_recall(pred_docs[:k], gold_docs)
            total_recall[k] += qi_recall
            temp_total_recall[k] = qi_recall
        retrieved_data.append({"utterance": user_question, "input": inputt, "utterance_org": utterance_org,
                              "retrieved": retrieved_docs, "selected_database": selected_database, "gold": gold_docs, "recall": temp_total_recall})

    for k in top_k:
        print(f"recall@{k}: {total_recall[k]/len(queries)}")

    with open(retrieved_docs_file, 'w', encoding='utf-8') as f:
        json.dump(retrieved_data, f, indent=4, ensure_ascii=False)

    print("retrieval done")
