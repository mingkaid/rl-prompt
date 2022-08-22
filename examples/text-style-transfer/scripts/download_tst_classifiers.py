import argparse
import os
from download import maybe_download

MODEL_URLS = {'yelp-train': 'https://drive.google.com/file/d/1AUBbpFcfBkKh5WUGwdXFhHxzZspPRn7W/view?usp=sharing',
              'yelp-test': 'https://drive.google.com/file/d/1VOhHZiYzZy8fzKpFDEqsbwvxO6iJ8dSr/view?usp=sharing',
              'shakespeare-train-100-0': 'https://drive.google.com/file/d/1A-yKYvXovOwumB99UygzC40NrjXgoNeZ/view?usp=sharing',
              'shakespeare-train-100-1': 'https://drive.google.com/file/d/1iW_SVoxHwORTX8aK5DWm3y_AZkEmg7Ny/view?usp=sharing',
              'shakespeare-train-100-2': 'https://drive.google.com/file/d/1PzJN3nXHeBT8-d3iR7vpDOJZPIiW0cfX/view?usp=sharing',
              'shakespeare-test-all': 'https://drive.google.com/file/d/17UMjwFjn2us7EIKr1Pnzx-LHGojlu-zB/view?usp=sharing'}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, choices=list(MODEL_URLS.keys()))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    target_path = './style_classifiers'
    os.makedirs(target_path, exist_ok=True)
    maybe_download(urls=MODEL_URLS[args.model_name],
                   path=target_path,
                   extract=True,
                   filenames=args.model_name + '.tar.gz')


