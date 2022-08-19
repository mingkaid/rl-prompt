import argparse
import os
from download import maybe_download

MODEL_URLS = {'yelp': 'https://drive.google.com/file/d/112k87qGwprmnwZk3lZ1F9F7zfOka7Fm3/view?usp=sharing',
              'shakespeare': 'https://drive.google.com/file/d/1TV88GbOcrpj5QIQkJ48bG4fPgSIFIvpQ/view?usp=sharing'}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, choices=list(MODEL_URLS.keys()))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    target_path = './evaluation/ppl'
    os.makedirs(target_path, exist_ok=True)
    maybe_download(urls=MODEL_URLS[args.model_name],
                   path=target_path,
                   extract=True,
                   filenames=args.model_name + '.tar.gz')


