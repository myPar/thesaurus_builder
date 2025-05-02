from transformers import AutoTokenizer, T5EncoderModel
import torch
import torch.nn.functional as F
import os
import re
import magic
import sys
import numpy as np
import argparse


# return model and tokenizer:
def load_model(device):
    assert device in ['cpu', 'cuda']
    model = T5EncoderModel.from_pretrained("ai-forever/FRIDA").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA", local_files_only=True)

    return model, tokenizer

def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]

# returns embeddings:
def infer(model, tokenizer, inputs, device, batch_size=10):
    model.to(device)
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[start:start + batch_size]
            
            tokenized_inputs = tokenizer(
                batch_inputs,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            outputs = model(**tokenized_inputs)
            
            embeddings = pool(
                outputs.last_hidden_state,
                tokenized_inputs["attention_mask"],
                pooling_method="cls"  # or "mean"
            )
            
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)


def extract_definitions(md_file_path:str):
    with open(md_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.strip().splitlines()
    definitions = []
    current_definition = []
    definition_start_re = re.compile(r"^\*\s+(?:__|\*\*)([^*_]+)(?:__|\*\*)[^—-]*[—-]\s+(.*)$")

    for line in lines:
        match = definition_start_re.match(line)
        if match:
            # new definition appears - add previous to result:
            if len(current_definition) > 0:
                definitions.append('\n'.join(current_definition).strip())
            current_definition = [line]
        else:
            # add line to definition
            current_definition.append(line)
    # add remain definition
    definitions.append('\n'.join(current_definition).strip())

    return definitions

# definitions = extract_definitions('/home/cossmo/thesaurus/prompts/prompt.md')
# with open('definitions.md', 'w', encoding='utf-8') as f:
#     f.write('\n\n'.join(definitions))

def remove_markdown_artifacts(data: str) -> str:
    data = re.sub(r"#+[ \t]*([^\n\r]+)[\r\n]+", r"\1\n", data)   # remove markdown header artifact and redundant new lines
    data = re.sub(r"\*+([а-яА-Я0-9A-Za-z\. \t]+[:\?\.;]?)\*+", r"\1", data)
    data = re.sub(r"\*+([^*]+:?)\*+", r"\1", data)
    data = re.sub(r"_+([а-яА-Я0-9A-Za-z\. \t]+[:\?\.;]?)_+", r"\1", data)

    return data

# returns indices to drop
def search_duplicates(embeddings, similarity_threshold:float=0.9) -> set[int]:
    assert len(embeddings.size()) == 2
    embeddings = embeddings.to('cpu')
    scores_matrix = torch.matmul(embeddings, embeddings.T)
    indices_to_drop = set()
    indices = torch.arange(0, len(embeddings))

    for i in range(len(scores_matrix)):
        if i in indices_to_drop:
            continue
        score_line = scores_matrix[i]
        similar_indices = set(indices[score_line > similarity_threshold].to('cpu').numpy())
        similar_indices.discard(i)
        indices_to_drop.update(similar_indices)

    return indices_to_drop


# building thesaurus based on 
def build_thesaurus(src_dir:str, output_dir:str, model, tokenizer, device:str='cuda'):
    if not os.path.isdir(src_dir):
        raise Exception(f'no specified directory - {src_dir}')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    files = os.listdir(src_dir)
    files = [f for f in files if magic.from_file(os.path.join(src_dir, f), mime=True) == 'text/plain']
    definitions = []

    for f in files:
        f_path = os.path.join(src_dir, f)
        definitions += extract_definitions(f_path)
    definitions_text = ["paraphrase: " + remove_markdown_artifacts(definition) for definition in definitions]
    print(*definitions_text)
    definition_embeddings = infer(model, tokenizer, definitions_text, device)

    all_ids = np.arange(0, len(definitions))
    duplicates_ids = search_duplicates(definition_embeddings)
    remain_ids = list(set(all_ids) - duplicates_ids)
    remain_ids.sort()
    remain_definitions = np.array(definitions)[remain_ids]

    with open(os.path.join(output_dir, 'thesaurus.md'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(remain_definitions))

    return remain_definitions


def validate_args(args):
    if not os.path.isdir(args.dir_path):
        raise Exception(f"input dir - {args.dir_path} doesn't exists")
    if not os.path.isdir(args.output):
        os.mkdir(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, required=True,
                        help='input dir path with .md files')
    parser.add_argument('--output', type=str, required=True,
                        help='path to generation result directory')
    try:
        args = parser.parse_args()
        validate_args(args)
    except Exception as e:
        print('Parse args exception: ' + repr(e), file=sys.stderr)
        parser.print_help()
        return    
    model, tokenizer = load_model('cuda')
    print('building thesaurus...')
    build_thesaurus(args.dir_path, args.output, model, tokenizer)
    print('thesaurus is build.')

if __name__ == '__main__':
    main()