import sys
import json
import random
import argparse

from copy import deepcopy
random.seed(42)
# import transformers

sys.path.append('.')


if __name__ == '__main__':
    from utils.generate import generate_with_llm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--retrieved_file", type=str)
    parser.add_argument("--dump_file", type=str)
    parser.add_argument("--prompt_file", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    args = parser.parse_args()
    # transformers.set_seed(args.random_seed)

    with open(args.retrieved_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data = random.sample(data, 64)
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = '\n'.join(line.strip('\n') for line in f)
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    prompts = []
    for d in data:
        sel_dbs = " \n ".join([x[0] for x in d['selected_database']]) if len(d['selected_database']) > 1 else  d['selected_database'][0][0]
        prompt = prompt_template.format(
            question=d['utterance'],
            database=sel_dbs
        )
        prompts.append(prompt)
    

    predictions = generate_with_llm(args.model_name_or_path, prompts, config)
    for d, p in zip(data, predictions):
        rewrited_query = p[0][0].strip()
        utte = deepcopy(d["utterance"])
        if d.get("utterance_org") and isinstance(d["utterance_org"], list):
            d["utterance_org"].append(utte)
        elif d.get("utterance_org") and isinstance(d["utterance_org"], str):
            d["utterance_org"] = [d["utterance_org"]].append(utte)
        elif not d.get("utterance_org"):
            d["utterance_org"] = [utte]
        # if isinstance(d["utterance_org"], str):
        #     d["utterance_org"] = [d["utterance_org"]].append(deepcopy(d["utterance"]))
        # elif isinstance(d["utterance_org"], list):
        #     d["utterance_org"] = d["utterance_org"].append(deepcopy(d["utterance"]))
        d['utterance'] = rewrited_query

    with open(args.dump_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
