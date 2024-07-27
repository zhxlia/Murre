import os
import sys
import json
import sqlite3
import argparse

from typing import List, Dict, Any

sys.path.append('.')

ZERO_SHOT_PROMPT = """
-- Given the following SQL tables, your job is to write queries given a user’s request .
{table}

Question: {question}
select
""".strip()

ONE_SHOT_PROMPT = """
Using valid SQLite , answer the following questions for the tables provided .

CREATE TABLE information (
description_losses text ,
column_1939_40 text ,
column_1940_41 text ,
column_1941_42 text ,
column_1942_43 text ,
column_1943_44 text ,
column_1944_45 text ,
total text
);
/*
Columns and examples in each column :
description_losses: "Direct War Losses", "Murdered", "Deaths In Prisons & Camps", "Deaths Outside of Prisons & Camps", "Murdered in Eastern Regions", "Deaths other countries", "Total" ;
column_1939_40: "360,000", "75,000", "69,000", "-", "-", "-", "504,000" ;
column_1940_41: "-", "100,000", "210,000", "42,000", "-", "-", "352,000" ;
column_1941_42: "-", "116,000", "220,000", "71,000", "-", "-", "407,000" ;
column_1942_43: "-", "133,000", "266,000", "142,000", "-", "-", "541,000" ;
column_1943_44: "-", "82,000", "381,000", "218,000", "-", "-", "681,000" ;
column_1944_45: "183,000", "-", "-", "-", "100,000", "-", "270,000" ;
total: "543,000", "506,000", "1,146,000", "473,000", "100,000", "2,000", "2,770,000" ;
*/
Question: how many people were murdered in 1940/41?
SELECT column_1940_41 FROM information where description_losses = "Murdered"

{table}
Question: {question}
select
""".strip()

def check_sql_equivalence(db_path: str, p_str: str, g_str: str):
    p_str = p_str.lower()
    g_str = g_str.lower()
    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 执行预测的SQL并获取结果
        cursor.execute(p_str)
        p_results = cursor.fetchall()
        p_columns = [desc[0] for desc in cursor.description]
        # 执行标准答案的SQL并获取结果
        cursor.execute(g_str)
        g_results = cursor.fetchall()
        g_columns = [desc[0] for desc in cursor.description]
        # 关闭连接
        cursor.close()
        conn.close()
    except:
        return False, [], [], [], []

    # 如果列数或行数不同，则返回False
    if len(p_results) != len(g_results) or len(p_columns) != len(g_columns):
        return False, p_columns, p_results, g_columns, g_results

    # 检查列名称是否相同（忽略顺序）
    if set(p_columns) != set(g_columns):
        return False, p_columns, p_results, g_columns, g_results

    # 对于每一行，检查两个查询的结果是否一致（忽略列的顺序）
    for p_row, g_row in zip(p_results, g_results):
        p_dict = dict(zip(p_columns, p_row))
        g_dict = dict(zip(g_columns, g_row))
        if p_dict != g_dict:
            return False, p_columns, p_results, g_columns, g_results

    return True, p_columns, p_results, g_columns, g_results

def locate_idx(tables: List[str], pref: str):
    # print(tables)
    # print(pref)
    for ti, t in enumerate(tables):
        if pref == t.split("(")[0]:
            return ti
    return 100

def construct_tables_input(tables: List[str], dbs_dict: Dict[str, Any]):
    db_tables: Dict[str, List[str]] = {}
    tables_input: List[str] = ["" for i in range(len(tables))]
    for si, schema in enumerate(tables):
        db_id = schema.split(".")[0]
        if db_tables.get(db_id):
            db_tables[db_id].append(schema)
        else:
            db_tables.setdefault(db_id, [schema])
    for db, dtables in db_tables.items():
        # print("\n")
        # print(db)
        db_dict = dbs_dict[db]
        # print([dt.split("(")[0].split(".")[1] for dt in dtables])
        filt_db_dict = filter_ret_tables_from_db(db_dict, db, [dt.split("(")[0].split(".")[1] for dt in dtables])
        # print(filt_db_dict)
        table_names = filt_db_dict["table_names"]
        packed_tables = pack_table(filt_db_dict, True)
        # print(packed_tables)
        for pi, p in enumerate(packed_tables.split("\n\n")):
            idx = locate_idx(tables, f"{db}.{table_names[pi]}")
            tables_input[idx] = p
            # print(f"{idx}: {tables_input}")
    # print(tables_input)
    for t in tables_input:
        if len(t) == 0:
            print(t)
        assert len(t) > 0
    return "\n".join(tables_input)


if __name__ == '__main__':
    from utils.format_db import format_db
    from utils.generate import generate_with_llm
    from retrieve.utils import filter_ret_tables_from_db, pack_table
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--query_file", type=str)
    parser.add_argument("--tables_file", type=str)
    # parser.add_argument("--databases_path", type=str)
    parser.add_argument("--dump_file", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    
    args = parser.parse_args()
    config_file = args.config_file
    # db_dir = args.databases_path
    prompt = ZERO_SHOT_PROMPT
    query_path = args.query_file
    oup_path = args.dump_file
    top_k = args.top_k

    inputs = []
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open(query_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.tables_file, 'r', encoding='utf-8') as f:
        org_dbs_dict = json.load(f)
    dbs_dict = {d["db_id"]: d for d in org_dbs_dict}
    for d in data:
        table = construct_tables_input([r["schema"] for r in d["retrieved"][:top_k]], dbs_dict)
        if d.get("utterance") and isinstance(d["utterance"], str):
            question = d["utterance"]
        elif d.get("utterance_org"):
            question = d["utterance_org"]
        elif d.get("input") and isinstance(d["input"], list):
            question = d["input"][0]
        else:
            question = d.get("question")
        assert question is not None
        # question = d["utterance_org"] if d.get("utterance_org") else d["question"]
        d_prompt = prompt.format(table=table, question=question)
        inputs.append(d_prompt)
    
    prediction = generate_with_llm('35turbo', inputs, config)

    predicted_sqls = ""
    
    for i, d in enumerate(data):
        predicted_sqls += "select " + prediction[i][0][0].replace("\n", " ") + "\n\n"
        # print(inputs[i])
    in_file = os.path.join("/".join(oup_path.split("/")[:-1]), f"inp.{top_k}.txt")
    with open(in_file,'w') as f:
        f.write("\n\n".join(inputs))
    with open(oup_path,'w') as f:
        f.write(predicted_sqls) 
        # f.write("\n".join([f"{input[pi]} \n {p}" for pi, p in enumerate(predicted_sqls.split("\n")[:-1])])) 
    # for i, d in enumerate(data):
    #     d["perd_sql"] = "select " + prediction[i][0][0]
    #     db_path = os.path.join(db_dir, d["db_id"], d["db_id"]+".sqlite")
    #     d["correct"] = check_sql_equivalence(db_path, d["perd_sql"], d["query"])[0]

    # print(f"EM : {len([x for x in data if x['correct']]) / len(data)}")
    # with open(oup_path, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)
    