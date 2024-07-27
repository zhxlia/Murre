import json
import random

from typing import List, Dict, Any

random.seed(42)
def schema_to_list(schema:str):
    # print(schema)
    sset = set()
    if "." in schema:
        sset.add(schema.split(".")[0])
        sset.add(schema.split(".")[1].split("(")[0])
    else:
        sset.add(schema.split("(")[0])
    # print(schema.split(", "))
    for wi, w in enumerate(schema.split(", ")):
        if wi == 0:
            sset.add(w.split("(")[1])
        elif wi == len(schema.split(", "))-1:
            sset.add(w[:-2])
        else:
            sset.add(w)
    return list(sset)

def judge_error(d: Dict[str, Any], top_k: int):
    # print(d)
    if d.get("question"):
        q_list = d.get("question")
    elif d.get("utterance"):
        q_list = d.get("utterance")
    error = [False, False]
    for g in d["gold"]:
        g_list = schema_to_list(g)
        if g not in[r["schema"] for r in d["retrieved"]][:top_k] and len([gw for gw in g_list if gw not in q_list]) == len(g_list):
            # relevant_dissimilar += 1
            error[0] = True
            break
    # if True:
    if not error[0]:
        for r in [x["schema"] for x in d["retrieved"][:top_k]]:
            r_list = schema_to_list(r)
            if r not in d["gold"] and len([rw for rw in r_list if rw in q_list])>0:
                # irrelevant_similar += 1
                error[1] = True
                break
    return error

if __name__=="__main__":
    inp_path0 = "./retrieve/dataset/spider/125m/result/turn0/dev.json"
    inp_path1 = "./retrieve/dataset/spider/125m/crush/dev.json"
    inp_path2 = "./retrieve/dataset/spider/125m/beam/3/result/turn2/dev.json"
    oup_path = f"./retrieve/dataset/spider/125m/beam/3/result/turn2/dev.comp.json"
    top_k = 3
    # num_sample = 64

    with open(inp_path0, "r", encoding="utf-8") as f:
        data0 = json.load(f)
    with open(inp_path1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(inp_path2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    
    # data = [data2[di] for di, d in enumerate(data2) if d["recall"][str(top_k)] < data0[di]["recall"][str(top_k)]] 
    data_comp = [data2[di] for di, d in enumerate(data2) if d["recall"][str(top_k)] > data1[di]["recall"][str(top_k)] and len(d["gold"])==2 and d["recall"][str(top_k)] > data0[di]["recall"][str(top_k)]]
    data_comp_crush = [data1[di] for di, d in enumerate(data2) if d["recall"][str(top_k)] > data1[di]["recall"][str(top_k)] and len(d["gold"])==2 and d["recall"][str(top_k)] > data0[di]["recall"][str(top_k)]]
    # data_si = [d for di, d in enumerate(data2) if d["recall"][str(top_k)] > data1[di]["recall"][str(top_k)] and judge_error(data1[di], top_k)[1]]
    # data_rd = [d for di, d in enumerate(data2) if d["recall"][str(top_k)] > data1[di]["recall"][str(top_k)] and judge_error(data1[di], top_k)[0]]
    # data_ind = [di for di, d in enumerate(data2) if d["recall"][top_k] > data1[di]["recall"][top_k]]

    print(len(data_comp))

    # print(len(data_rd))
    # print(len(data_rd)/len(data))
    # print(len(data_si))
    # print(len(data_si)/len(data))

    # print(data_ind)
    # data = random.sample(data, num_sample)

    for di, d in enumerate(data_comp):
        d["retrieved"] = d["retrieved"][:10]
        d["crush"] = data_comp_crush [di]["utterance"]
        # d["error"] = [""]

    with open(oup_path, 'w', encoding='utf-8') as f:
        json.dump(data_comp, f, indent=4, ensure_ascii=False)

    # da = [d for d in data if len(d["gold"])>1]
    # print(len(da)/len(data))

    # da1 = [d for d in data if d["retrieved"][0]["schema"] in d["gold"]]
    # print(len(da1)/len(data))

    
    # count_e = 0
    # for d in data:
    #     flag = True
    #     for d0 in d["turn0_selected_database"]:
    #         if d0 in d["gold"]:
    #             flag = False
    #     if flag:
    #         count_e += 1
    # print(count_e/len(data))
