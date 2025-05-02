import argparse
import os
import sys
import magic


def extract_text(srt_file_path: str):
    if not os.path.isfile(srt_file_path):
        raise Exception(f'no srt file with such name: {srt_file_path}')
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    delimiter = '\n\n'
    srt_blocks = data.split(delimiter)
    texts = [srt_block.split('\n')[-1] for srt_block in srt_blocks]
    text = '\n'.join(texts)

    return text


def save_srt_extracted(srt_file_path: str, dst_file_path:str):
    text = extract_text(srt_file_path)

    with open(dst_file_path, 'w', encoding='utf-8') as f:
        f.write(text)


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

    if file_path != "":
        f_name = os.path.basename(file_path)
        if not magic.from_file(file_path, mime=True) == 'text/plain':
            print(f"WARNING: {file_path} - is not a text file, so can't be filtered")
            return
        save_srt_extracted(file_path, os.path.join(output, f_name))
        print(f'file {f_name} is processed')
    else:
        files = os.listdir(dir_path)

        for f in files:
            file_path = os.path.join(dir_path, f)
            f_name = os.path.basename(file_path)

            if not magic.from_file(file_path, mime=True) == 'text/plain':
                print(f"WARNING: {file_path} - is not a text file, so can't be filtered")            
                continue

            save_srt_extracted(file_path, os.path.join(output, f_name))
            print(f'file {f_name} is processed')

if __name__ == '__main__':
    main()