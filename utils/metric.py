from typing import List, Dict, Any


def compute_recall(pred_list: List[str], gold_list: List[str]) -> float:
    correct_count = 0
    for p in pred_list:
        if p in gold_list:
            correct_count += 1
    rec = 1.0*correct_count/len(gold_list)
    return rec


def compute_recall_multiple(top_k: List[int], pred_list: List[str], gold_list: List[str]) -> Dict[int, float]:
    # return {k: compute_recall(pred_list[:k], gold_list) for k in top_k if k < len(pred_list)}
    return {k: compute_recall(pred_list[:k], gold_list) for k in top_k if k <= len(pred_list)}


def compute_em_multiple(top_k: List[int], pred_list: List[str], gold_list: List[str]) -> Dict[int, float]:
    em: Dict[int, float] = {k:0.0 for k in top_k}
    for k in top_k: 
        if set(gold_list).issubset(set(pred_list[:k])):
            em[k] += 1
    return em

def compute_res(top_k: List[int], data: List[Dict[str, Any]]) -> Dict[str,Dict[int, float]]:
    recall: Dict[int, float] = {k:0.0 for k in top_k}
    em: Dict[int, float] = {k:0.0 for k in top_k}
    for di, d in enumerate(data):
        rec = compute_recall_multiple(top_k, [x["schema"] for x in d["retrieved"]], d["gold"])
        em0 = compute_em_multiple(top_k, [x["schema"] for x in d["retrieved"]], d["gold"])
        for k in top_k:
            recall[k] += rec[k]
            em[k] += em0[k]
    em = {k: v/len(data) for k, v in em.items()}
    recall = {k: v/len(data) for k, v in recall.items()}

    return {"recall": recall, "em": em}