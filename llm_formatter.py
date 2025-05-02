import os
import argparse
import json
import sys
import magic
from pdf_extractor import DocContentExtractor
from srt_extractor import save_srt_extracted
from llm_inference import load_model, refactor_doc
from tqdm import tqdm


def validate_args(args):
    if args.subtitles_parser_prompt.strip() == "":
        raise Exception(f'no subtitles parser prompt file')
    if args.pdf_parser_prompt.strip() == "":
        raise Exception(f'no pdf parser prompt file')
    if not os.path.isdir(args.dir_path) and args.file_path.strip() == "":
        raise Exception(f"input dir - {args.dir_path} doesn't exists and no input file is specified")
    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    if not os.path.isdir(args.temp_dir):
        os.mkdir(args.temp_dir)


def read_json(file_path:str):
    if not os.path.isfile(file_path):
        raise Exception(f"can't read json: no such file - {file_path}")
    with open(file_path, encoding='utf-8') as f:
        data = f.read()
    return json.loads(data)


def extract_text(file_path: str, pdf_dir: str, srt_dir: str, pdf_extractor: DocContentExtractor):
    name, ext = os.path.splitext(os.path.basename(file_path))
    if magic.from_file(file_path, mime=True) == 'application/pdf':
        out_dir = pdf_dir
        pdf_extractor.extract_content(file_path, out_dir)
    elif magic.from_file(file_path, mime=True) == 'text/plain' and ext == '.srt':
        out_dir = srt_dir
        save_srt_extracted(file_path, os.path.join(out_dir, name + '.txt'))
    else:
        raise Exception(f'not supported file format for text extraction: {ext}')


def refactor_docs(dir_path:str, output_dir:str, model, tokenizer, few_shot_prompt):
    files = os.listdir(dir_path)
    pbar = tqdm(total=len(files), desc=f'parse docs in {dir_path}...')

    for f in files:
        file_path = os.path.join(dir_path, f)
        refactor_doc(file_path, output_dir, few_shot_prompt, model, tokenizer)
        pbar.update(1)
    pbar.close()


def get_path_and_prompt(file_path:str, pdf_dir: str, srt_dir: str, few_shot_pdf, few_shot_srt):
    file_type = magic.from_file(file_path, mime=True)
    name, ext = os.path.splitext(os.path.basename(file_path))

    if file_type == 'application/pdf':
        dst_path = os.path.join(pdf_dir, name + ".txt")
        few_shot_prompt = few_shot_pdf
    elif file_type == 'text/plain':
        dst_path = os.path.join(srt_dir, name + ".txt")
        few_shot_prompt = few_shot_srt

    return dst_path, few_shot_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='model path on disk')
    parser.add_argument('--file_path', type=str, required=True,
                        help='file to refactor, if not specified, --dir_path will be used')
    parser.add_argument('--output', type=str, required=True,
                        help='path to generation result directory')
    parser.add_argument('--dir_path', type=str, required=True,
                        help='directory to get files on refactoring from. this argument is ignored if '
                             '--file_path is specified')
    parser.add_argument('--temp_dir', type=str, required=False, default='temp_dir',
                        help="temporary directory where files with extracted text from srt and pdf's will stored")
    parser.add_argument('--pdf_parser_prompt', type=str, required=False, default='prompts/pdf_parser_prompt.json',
                        help='json file with few-shot prompt for pdf extracted content parsing')
    parser.add_argument('--subtitles_parser_prompt', type=str, required=False, default='prompts/subtitles_parser_prompt.json',
                        help='json file with few-shot prompt for subtitle extracted content parsing')
    try:
        args = parser.parse_args()
        validate_args(args)
    except Exception as e:
        print('Parse args exception: ' + repr(e), file=sys.stderr)
        parser.print_help()
        return
    model_path = args.model_path
    output = args.output
    file_path = args.file_path
    dir_path = args.dir_path
    temp_dir = args.temp_dir
    pdf_parser_prompt = args.pdf_parser_prompt
    subtitles_parser_prompt = args.subtitles_parser_prompt

    # create separate dirs for pdf's extracted text and for srt extracted texts
    pdf_dir = os.path.join(temp_dir, 'pdf')
    srt_dir = os.path.join(temp_dir, 'srt')

    if not os.path.isdir(pdf_dir):
        os.mkdir(pdf_dir)
    if not os.path.isdir(srt_dir):
        os.mkdir(srt_dir)
    few_shot_pdf = read_json(pdf_parser_prompt)
    few_shot_srt = read_json(subtitles_parser_prompt)

    pdf_extractor = DocContentExtractor()
    model, tokenizer = load_model(model_path)

    if file_path != "":
        extract_text(file_path, pdf_dir, srt_dir, pdf_extractor)
        dst_path, few_shot_prompt = get_path_and_prompt(file_path, pdf_dir, srt_dir, few_shot_pdf, few_shot_srt)
        refactor_doc(file_path, dst_path, few_shot_prompt, model, tokenizer)
    else:
        files = os.listdir(dir_path)
        # select only text files:
        text_files = [f for f in files if magic.from_file(os.path.join(dir_path, f), mime=True) == 'text/plain']

        if len(files) == 0:
            print(f'WARNING: no text files exists here - {dir_path}, nothing to parse')
            return
        # extract texts from pdf's and from srt with creating separate pdf and srt directories for it:
        for file in text_files:
            file_path = os.path.join(dir_path, file)
            extract_text(file, pdf_dir, srt_dir, pdf_extractor)
        # parse pdf's:
        refactor_docs(pdf_dir, output, model, tokenizer, few_shot_pdf)
        # parse srt's
        refactor_docs(srt_dir, output, model, tokenizer, few_shot_srt)
