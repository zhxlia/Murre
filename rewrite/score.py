import os
import sys
import math
import json
import argparse

from tqdm import tqdm
from typing import List, Dict, Any, Tuple

sys.path.append('.')


def find_json_files(path: str) -> List[str]:
    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    # print(sorted(json_files))
    return sorted(json_files)


def pos(x: float) -> float:
    return (x+2)/2

def judge_stop(utte: str) -> bool:
    if "There is no" in utte or "None of the given tables" in utte or "No additional tables" in utte or "No completion needed" in utte or ("None" in utte):
        return True
    return False

if __name__ == '__main__':
    from utils.metric import compute_recall, compute_recall_multiple, compute_em_multiple, compute_res
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved_paths", type=str, nargs="+")
    parser.add_argument("--dump_path", type=str)
    parser.add_argument("--top_k", type=int, nargs="+")
    # parser.add_argument("--score_file", type=str)
    args = parser.parse_args()

    # args.retrieved_paths = ["./retrieve/dataset/spider/125m/turn0", "./retrieve/dataset/spider/125m/main/2/turn1", "./retrieve/dataset/spider/125m/main/2/turn2", "./retrieve/dataset/spider/125m/main/2/turn3"]
    args.retrieved_paths = ["./retrieve/dataset/spider/125m/turn0", "./retrieve/dataset/spider/125m/main/2/turn1", "./retrieve/dataset/spider/125m/main/2/turn2"]
    # args.retrieved_paths = ["./retrieve/dataset/spider/125m/turn0", "./retrieve/dataset/spider/125m/main/2/turn1"]
    args.dump_path = "./retrieve/dataset/spider/125m/main/2/result/turn2"
    args.top_k = [3, 5, 10, 20, 30, 50, 100]
    # args.score_file = "./rewrite/dataset/spider/125m/input/turn1/score.json"
    
    num_turns = len(args.retrieved_paths)
    print(num_turns)
    retrieved_data: List[List[List[Dict[str, Any]]]] = [[]
                                                        for _ in range(num_turns)]

    for ti in range(num_turns):
        retrieved_files = find_json_files(args.retrieved_paths[ti])
        for fi, file in enumerate(retrieved_files):
            with open(file, 'r', encoding='utf-8') as f:
                retrieved_data[ti].append(json.load(f))

    # the similarity of each schema under the query
    schema_score: List[Dict[str, float]] = []
    schema_score_t0: List[Dict[str, float]] = []

    # num_turns = len(retrieved_data[0][0]["selected_database"])

    for di, data in enumerate(retrieved_data[0][0]):
        schema_score.append({})
        for x in data['retrieved']:
            schema_score[di].setdefault(x['schema'], 0)
    
    end_turn = [num_turns-1 for _ in retrieved_data[0][0]]

    for ti in range(1, num_turns):
        for di in range(len(retrieved_data[1][0])):
            for fi, fdata in enumerate(retrieved_data[-1]):
                # if judge_stop(retrieved_data[ti][fi][di]["utterance"]) and end_turn[di] > ti-1:
                if judge_stop(retrieved_data[ti][fi][di]["input"][0]) and end_turn[di] > ti-1:
                    end_turn[di] = ti-1

    count_end_turn: Dict[int, Any] = {}
    for ei, e in enumerate(end_turn):
        if e not in count_end_turn.keys():
            count_end_turn[e] = 0
        count_end_turn[e] += 1
    for c in count_end_turn.keys():
        count_end_turn[c] /= len(retrieved_data[0][0])
    print(count_end_turn)

    for di, d0 in enumerate(retrieved_data[0][0]):
        i_turn = end_turn[di]
        for fi, fdata in enumerate(retrieved_data[i_turn]):
            d = fdata[di]
            if "question" in d and not "utterance" in d:
                d["utterance"] = d["question"]
            for x in d['retrieved']:
                if d.get("selected_database"):
                    summ = math.log(pos(
                        x['similarity'])) + sum(math.log(pos(d["selected_database"][si][1])) for si, _ in enumerate(d["selected_database"]))
                else:
                    summ = pos(x['similarity'])
                schema_score[di][x['schema']] = max(summ, schema_score[di][x['schema']])
                if d.get("selected_database"):
                    for si, _ in enumerate(d["selected_database"]):
                        schema_score[di][d["selected_database"]
                                            [si][0]] = max(summ, schema_score[di][d["selected_database"][si][0]])
    # print(end_turn)
    print([i for i, et in enumerate(end_turn) if et==2])
    
    results: List[List[Tuple[str, float]]] = []
    for si, s in enumerate(schema_score):
        results.append([])
        results[-1] = sorted(s.items(), key=lambda x: x[1],
                             reverse=True)[:max(args.top_k)]
        retrieved = [{} for _ in range(max(args.top_k))]
        for k in range(max(args.top_k)):
            retrieved[k]["rank"] = k
            retrieved[k]["schema"] = results[si][k][0]
            retrieved[k]["similarity"] = results[si][k][1]
        utterance_org = retrieved_data[0][0][si].pop("question", None)
        retrieved_data[0][0][si].pop("pred_schema", None)
        retrieved_data[0][0][si]["utterance_org"] = utterance_org
        retrieved_data[0][0][si]["utterance"] = [[retrieved_data[i][fi][si]["utterance"] for fi in range(len(retrieved_data[i]))] for i in range(1, len(retrieved_data))]
        retrieved_data[0][0][si]["turn0_selected_database"] = [t["schema"] for t in retrieved_data[0][0][si]["retrieved"][:5]]
        retrieved_data[0][0][si]["retrieved"] = retrieved
        retrieved_data[0][0][si]["recall"] = compute_recall_multiple(args.top_k, [x['schema'] for x in retrieved], retrieved_data[0][0][si]['gold'])
 
    with open(os.path.join(args.dump_path, f"dev.json"), 'w', encoding='utf-8') as f:
        json.dump(retrieved_data[0][0], f, ensure_ascii=False, indent=4)
    
    recall = 0
    score = compute_res(args.top_k, retrieved_data[0][0])
    # print(recall / len(retrieved_data[0][0]))
    with open(os.path.join(args.dump_path, f"score.json"), 'w', encoding='utf-8') as f:
        json.dump(score, f, ensure_ascii=False, indent=4)
    print(score)


