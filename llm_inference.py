from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List
import re
from tqdm import tqdm
import os
import magic


MAX_TOKENS = 8192


def split_on_chunks(input_file:str, chunk_size: int):
    st_idx = 0
    end_idx = st_idx + chunk_size
    chunks = []

    with open(input_file, 'r', encoding='utf-8') as f:
        data = str(f.read())

    while st_idx < len(data):
        chunk = data[st_idx: end_idx]
        if end_idx < len(data) - 1 and re.search(r'\s$', chunk) is None:
            last_space = re.search(r'\s\S*$', chunk)
            if last_space is None:
                raise Exception('bad text: no spaces detected')
            chunk_end_idx = last_space.span()[0]
            chunk = chunk[:chunk_end_idx]
            end_idx = st_idx + chunk_end_idx
        chunks.append(chunk)
        st_idx = end_idx
        end_idx = min(st_idx + chunk_size, len(data))

    return chunks


def infer_chat(model, tokenizer, chat_template: List[dict], user_query: str, max_tokens: int = MAX_TOKENS):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=max_tokens)
    messages = chat_template + [{'role': 'user', 'content': user_query}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = model.generate([text], sampling_params)
    assert len(outputs) == 1

    return outputs[0].outputs[0].text


def refactor_doc(file_path: str, output_dir: str, few_shot_prompt: List[dict], model, tokenizer, chunk_size: int = MAX_TOKENS):
    if not os.path.isfile(file_path):
        raise Exception(f'not existing source file - {file_path}')
        
    assert magic.from_file(file_path, mime=True) == 'text/plain'
    
    def add_chunk(result, chunk):
        if re.search(r'\s$', chunk) is None:
            return result + " " + chunk
        return result + chunk

    # chunk size is equal to max tokens
    print(f'refactoring file - {file_path}')
    data_chunks = split_on_chunks(file_path, chunk_size)
    result = ""
    name, ext = os.path.splitext(os.path.basename(file_path))

    for chunk in data_chunks:
        filtered_chunk = infer_chat(model, tokenizer, few_shot_prompt, 'refactor this text: ' + chunk, max_tokens=chunk_size)
        result = add_chunk(result, filtered_chunk)

    with open(os.path.join(output_dir, name + '.txt'), 'w', encoding='utf-8') as o:
        o.write(result)


def load_model(model_name: str):
    llm = LLM(model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return llm, tokenizer
