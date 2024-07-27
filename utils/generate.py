import os
import time
import torch

from tqdm import tqdm
from copy import deepcopy
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor


class GPTGenerator(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.error_types = {
            "continue_error": [
                "timed out",
                "Connection error",
                "Connection reset by peer",
                "Remote end closed connection without response",
                "occurred in violation of protocol",
                "Failed to resolve",
                "TLSV1_ALERT_INTERNAL_ERROR",
                "Error communicating"
            ],
            "sleep_error": [
                "call rate limit",
                "token rate limit"
            ],
            "ignore_error": [
                "content",
                "reduce the length"
            ]
        }

    def generate_single(self, packed_data: List[tuple]) -> List[Tuple[str, float]]:
        from openai import AzureOpenAI
        from openai.types.chat import ChatCompletion

        sentence, engine, config = packed_data
        client = AzureOpenAI(
            api_version="your api version",
            azure_endpoint="your azure endpoint",
            api_key="your api key"
        )

        while True:
            try:
                if 'turbo' in engine:
                    completion: ChatCompletion = client.chat.completions.create(
                        model=engine,
                        messages=[{"role": "user", "content": sentence}],
                        **config)
                    return [(x.message.content, 1.0) for x in completion.choices]
                else:
                    raise Exception(f"unsupported engine: {engine}")
            except Exception as e:
                continue_flag = False
                sleep_flag = False
                ignore_flag = False
                for x in self.error_types['continue_error']:
                    if x in str(e):
                        continue_flag = True
                for x in self.error_types['sleep_error']:
                    if x in str(e):
                        sleep_flag = True
                        continue_flag = True
                for x in self.error_types['ignore_error']:
                    if x in str(e):
                        ignore_flag = True
                if sleep_flag:
                    time.sleep(5)
                if continue_flag:
                    continue
                if not ignore_flag:
                    print(e)
                return [""]

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        config = deepcopy(config)
        if config['parallel']:
            config.pop('parallel')
            packed_data = [(x, self.model_name, config) for x in source]
            with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as _:
                result: List[List[str]] = list(process_map(
                    self.generate_single, packed_data, max_workers=os.cpu_count() // 2, chunksize=1))
        else:
            config.pop('parallel')
            result: List[List[str]] = [self.generate_single(
                (x, self.model_name, config)) for x in tqdm(source)]
        return result


class LlamaGenerator(object):
    def __init__(self, model_name_or_path: str):
        self.llm = LLM(model=model_name_or_path,
                       tensor_parallel_size=torch.cuda.device_count())
        self.tokenizer = self.llm.get_tokenizer()
        self.stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:",
                            "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]

    def batch_data(self, data_list: List[str], batch_size: int) -> List[List[str]]:
        n = len(data_list) // batch_size
        batch_data = []
        for i in range(n-1):
            start = i * batch_size
            end = (i+1) * batch_size
            batch_data.append(data_list[start:end])
        last_start = (n-1) * batch_size
        batch_data.append(data_list[last_start:])
        return batch_data

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        too_long_data_count = 0
        source_filtered = []
        for i, x in tqdm(enumerate(source), total=len(source), desc="filtering too long input"):
            if len(self.tokenizer(x)['input_ids']) > self.llm.llm_engine.model_config.max_model_len:
                source[i] = "TL;NR"
                too_long_data_count += 1
            else:
                source_filtered.append(x)
        print(f"too long input count: {too_long_data_count}")
        if config['ignore_too_long']:
            source = source_filtered

        sampling_params = SamplingParams(
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=config['max_tokens'],
            n=config['n'],
            logprobs=1,
            stop=self.stop_tokens
        )
        batch_instances = self.batch_data(
            source, batch_size=config['batch_size'])
        res_completions: List[str] = []
        for _, prompt in tqdm(enumerate(batch_instances), total=len(batch_instances), desc="generating"):
            if not isinstance(prompt, list):
                prompt = [prompt]
            completions = self.llm.generate(
                prompt, sampling_params, use_tqdm=False)
            for output in completions:
                generated_text = []
                for x in output.outputs:
                    total_logprob = 0
                    for t in x.logprobs:
                        max_logprob_token = max(t.items(), key=lambda x: x[1])
                        if max_logprob_token[0] == 13:
                            break
                        total_logprob += max_logprob_token[1]
                    generated_text.append(
                        (x.text.lstrip('\n').split('\n\n')[0], total_logprob))
                res_completions.append(generated_text)
        return res_completions


MODEL_MAP: Dict[str, object] = {
    "llama": LlamaGenerator,
    "code": GPTGenerator,
    "turbo": GPTGenerator
}


def generate_with_llm(model_name_or_path: str, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
    for token in MODEL_MAP:
        if token in model_name_or_path.lower():
            generator = MODEL_MAP[token](model_name_or_path)
            break
    return generator.generate(source, config)


def consistency(answers: List[Tuple[str, Any, float]]) -> Tuple[str, Any]:
    count: Dict[str, float] = {}
    record: Dict[str, Tuple[str, str]] = {}
    for a, b, c in answers:
        x = str(b)
        if "error" in x.lower():
            continue
        if x not in count:
            count[x] = 0
            record[x] = (a, b)
        count[x] += c
    if not count:
        return "", ""
    return record[max(count, key=lambda x: count[x])]
