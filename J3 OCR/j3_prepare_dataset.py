"""
audiolens — j3 prepare_dataset.py

fetches images from 2 huggingface datasets and saves them
locally for offline ocr evaluation.

output structure:
    dataset/
        images/
            cord_001.jpg
            medocr_001.jpg
            ...

usage:
    python j3_prepare_dataset.py
    python j3_prepare_dataset.py --images-per-dataset 10
"""

# -- config --

import argparse

parser = argparse.ArgumentParser(description='prepare j3 ocr dataset')
parser.add_argument('--images-per-dataset', type=int, default=3,
                    help='images to fetch from each hf dataset (default: 20)')
args = parser.parse_args()

IMAGES_PER_DATASET = args.images_per_dataset

DATASET_DIR = './dataset'
IMAGES_DIR  = f'{DATASET_DIR}/images'

# datasets to pull from
DATASETS = {
    'cord':   {'hf_id': 'naver-clova-ix/cord-v2',             'category': 'restaurant receipt'},
    'medocr': {'hf_id': 'naazimsnh02/medocr-vision-dataset',  'category': 'medical document'},
}

SEED = 42


# -- imports --

import os
import random

from PIL import Image
from datasets import load_dataset

os.makedirs(IMAGES_DIR, exist_ok=True)
random.seed(SEED)


def extract_image(sample):
    """returns pil image from sample."""
    img = sample.get('image')
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        try:
            img = Image.fromarray(img)
        except Exception:
            return None
    return img.convert('RGB')


# -- fetch and save --

print('=' * 60)
print('  audiolens — j3 dataset preparation')
print('=' * 60)
print(f'  images per dataset : {IMAGES_PER_DATASET}')
print(f'  total (hf only)    : ~{IMAGES_PER_DATASET * len(DATASETS)}')
print(f'  save location      : {os.path.abspath(DATASET_DIR)}')
print('=' * 60)

total_saved = 0

for dataset_key, info in DATASETS.items():
    hf_id    = info['hf_id']
    category = info['category']

    print(f'\nloading {dataset_key}  ({hf_id})...')
    print(f'category: {category}')

    try:
        ds = load_dataset(hf_id, split='train', streaming=True, trust_remote_code=False)

        saved   = 0
        scanned = 0

        for sample in ds:
            scanned += 1
            img = extract_image(sample)
            if img is None:
                continue

            filename = f'{dataset_key}_{saved+1:03d}.jpg'
            filepath = os.path.join(IMAGES_DIR, filename)
            img.save(filepath, 'JPEG', quality=95)

            saved      += 1
            total_saved += 1

            if saved >= IMAGES_PER_DATASET:
                break

        print(f'  saved {saved} images  (scanned {scanned} samples)')

    except Exception as e:
        print(f'  [warn] failed to load {dataset_key}: {e}')
        print(f'  skipping.')


# -- summary --

print('\n' + '=' * 60)
print('  done.')
print(f'  images saved : {total_saved}')
print(f'  location     : {IMAGES_DIR}')
print('=' * 60)
print('\nnext: run python j3_ocr.py')
