import os
import math
import json
import argparse

from copy import deepcopy
from typing import List, Dict, Any, Tuple


def find_json_files(path: str) -> List[str]:
    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved_path", type=str)
    parser.add_argument("--dump_path", type=str)
    parser.add_argument("--top_k", type=int)

    args = parser.parse_args()

    # args.dump_path = "./rewrite/dataset/spider/125m/input/turn1_won"

    retrieved_data: List[List[Dict[str, Any]]] = []
    retrieved_files = find_json_files(args.retrieved_path)
    for file in retrieved_files:
        with open(file, 'r', encoding='utf-8') as f:
            retrieved_data.append(json.load(f))
    results: List[List[Dict[str, Any]]] = [[] for _ in range(args.top_k)]
    # print(len(retrieved_data[0]))

    for i in range(len(retrieved_data[0])):
        # len(retrieved_data) * top_k
        # print(len(retrieved_data[0]))
        retrieved_package: List[Tuple[str, List[str], List[float]]] = []
        for data in retrieved_data:
            if 'selected_database' not in data[i]:
                data[i]['selected_database'] = []
            for x in data[i]['retrieved'][:3*args.top_k]:
                if x['schema'] in [t[0] for t in data[i]['selected_database']]:
                    continue
                utterance = data[i].get("utterance", "") if data[i].get(
                    "utterance", "") else data[i].get("question", "")
                assert len(utterance) > 0
                retrieved_package.append((
                    utterance,
                    [t[0] for t in data[i]['selected_database']] + [x['schema']],
                    [t[1] for t in data[i]['selected_database']] + [x['similarity']],
                    data[i].get("utterance_org")
                ))

        retrieved_package = sorted(retrieved_package, key=(
            lambda x: sum(math.log(t) for t in x[2])), reverse=True)[:args.top_k]
        for j, r in enumerate(retrieved_package):
            results[j].append({
                'utterance': r[0],
                'utterance_org': r[3],
                'selected_database': list(zip(r[1], r[2])),
                "rel_schema": retrieved_data[0][i]['gold']
            })
            # count+=1
    # print(len(results))
    # print(len(results[0]))

    # if args.need_distribute:
    for i, r in enumerate(results):
        print(i)
        fpath = os.path.join(args.dump_path, f"dev.{i}.json")
        print(fpath)
        # print(r[:2])
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(r, f, ensure_ascii=False, indent=4)


