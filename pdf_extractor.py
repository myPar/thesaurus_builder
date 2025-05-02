from pypdf import PdfReader
import os
import magic
from pathlib import Path
import argparse
import sys


class DocContentExtractor():
    def __init__(self, min_content_size: int=50):
        self.min_content_size = min_content_size

    def _save_pdf_content(self, reader: PdfReader, f_path: str):
        content = ""

        for page in reader.pages:
            try:
                content += page.extract_text() + "\n"
            except Exception as e:
                continue    # skip unparsable page

        if len(content.strip()) >= self.min_content_size:
            with open(f_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())

    def extract_content(self, pdf_file_path:str, out_dir:str):
        if not os.path.isfile(pdf_file_path):
            raise Exception(f'not pdf file with such path - {pdf_file_path}')
        # check file is pdf:
        file_type = magic.from_file(pdf_file_path, mime=True)
        if file_type != 'application/pdf':
            raise Exception(f"can't parse not pdf file - {pdf_file_path}")
        # extract content and save:
        pdf_reader = PdfReader(Path(pdf_file_path))
        f_name = os.path.basename(pdf_file_path).split('.')[0] + '.txt'
        out_file_path = os.path.join(out_dir, f_name)

        self._save_pdf_content(pdf_reader, out_file_path)


def validate_args(args):
    if args.file_path.strip() != "" and not os.path.isfile(args.file_path):  # file is specified but doesn't exist
        raise Exception(f"file - {args.file_path} doesn't exists")
    if not os.path.isdir(args.dir_path) and args.file_path.strip() == "":
        raise Exception(f"input dir - {args.dir_path} doesn't exists and no input file is specified")
    if not os.path.isdir(args.output):
        os.mkdir(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=False, default="",
                        help='file to postprocess, if not specified, --dir_path will be used')
    parser.add_argument('--dir_path', type=str, required=False, default=".",
                        help='directory to get files on postprocessing from. this argument is ignored if '
                             '--file_path is specified')
    parser.add_argument('--output', type=str, required=False, default="postprocess_dir",
                        help='path to postprocessing result directory')
    try:
        args = parser.parse_args()
        validate_args(args)
    except Exception as e:
        print('Parse args exception: ' + repr(e), file=sys.stderr)
        parser.print_help()
        return
    output = args.output.strip()
    file_path = args.file_path.strip()
    dir_path = args.dir_path.strip()
    pdf_extractor = DocContentExtractor()

    if file_path != "":
        f_name = os.path.basename(file_path)
        if not magic.from_file(file_path, mime=True) == 'application/pdf':
            print(f"WARNING: {file_path} - is not a text file, so can't be parsed")
            return
        pdf_extractor.extract_content(file_path, output)
        print(f'file {f_name} is processed')
    else:
        files = os.listdir(dir_path)

        for f in files:
            file_path = os.path.join(dir_path, f)
            f_name = os.path.basename(file_path)

            if not magic.from_file(file_path, mime=True) == 'application/pdf':
                print(f"WARNING: {file_path} - is not a pdf file, so can't be parsed")            
                continue

            pdf_extractor.extract_content(file_path, output)
            print(f'file {f_name} is processed')

if __name__ == '__main__':
    main()
        