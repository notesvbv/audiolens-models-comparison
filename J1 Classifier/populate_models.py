"""
audiolens — populate_models.py

downloads required models from huggingface and saves them
locally into ./models/ so juncture scripts can load them offline.

run this once before running any juncture scripts.

usage:
    python populate_models.py
    python populate_models.py --juncture j1
    python populate_models.py --juncture j2
"""

import os
import argparse

parser = argparse.ArgumentParser(description='Download and save AudioLens models locally')
parser.add_argument('--juncture', type=str, default='all',
                    choices=['all', 'j1', 'j2', 'j3', 'j4'],
                    help='Which juncture models to download (default: all)')
args = parser.parse_args()

MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

# -- model registry --

J1_MODELS = [
    {
        'hf_id':      'microsoft/dit-base-finetuned-rvlcdip',
        'folder':     'dit-base-finetuned-rvlcdip',
        'type':       'image_classification',
        'size_mb':    343,
        'juncture':   'J1 — Document Classification',
    },
    {
        'hf_id':      'google/efficientnet-b0',
        'folder':     'efficientnet-b0',
        'type':       'image_classification',
        'size_mb':    21,
        'juncture':   'J1 — Document Classification',
    },
    {
        'hf_id':      'google/vit-base-patch16-224',
        'folder':     'vit-base-patch16-224',
        'type':       'image_classification',
        'size_mb':    330,
        'juncture':   'J1 — Document Classification',
    },
]

J2_MODELS = []
J3_MODELS = []
J4_MODELS = []

ALL_MODELS = J1_MODELS + J2_MODELS + J3_MODELS + J4_MODELS

JUNCTURE_MAP = {
    'j1':  J1_MODELS,
    'j2':  J2_MODELS,
    'j3':  J3_MODELS,
    'j4':  J4_MODELS,
    'all': ALL_MODELS,
}

# -- download logic --

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

def model_already_exists(folder):
    """returns true if the folder already has config + weights."""
    path = os.path.join(MODELS_DIR, folder)
    has_config   = os.path.exists(os.path.join(path, 'config.json'))
    has_weights  = (
        os.path.exists(os.path.join(path, 'model.safetensors')) or
        os.path.exists(os.path.join(path, 'pytorch_model.bin'))
    )
    return has_config and has_weights


def download_image_classification_model(hf_id, folder):
    save_path = os.path.join(MODELS_DIR, folder)
    os.makedirs(save_path, exist_ok=True)

    print(f'  Downloading processor...')
    processor = AutoImageProcessor.from_pretrained(hf_id)
    processor.save_pretrained(save_path)

    print(f'  Downloading model weights (safetensors)...')
    model = AutoModelForImageClassification.from_pretrained(
        hf_id,
        use_safetensors=True,
    )
    model.save_pretrained(save_path, safe_serialization=True)


# -- main --

target_models = JUNCTURE_MAP[args.juncture]

if not target_models:
    print(f'No models registered for --juncture {args.juncture} yet.')
    exit(0)

total_mb = sum(m['size_mb'] for m in target_models)
print('=' * 60)
print(f'  AudioLens — Model Downloader')
print('=' * 60)
print(f'  Juncture filter : {args.juncture.upper()}')
print(f'  Models to fetch : {len(target_models)}')
print(f'  Approx size     : ~{total_mb} MB total')
print(f'  Save location   : {os.path.abspath(MODELS_DIR)}')
print('=' * 60)

downloaded, skipped, failed = 0, 0, 0

for entry in target_models:
    hf_id    = entry['hf_id']
    folder   = entry['folder']
    size_mb  = entry['size_mb']
    junction = entry['juncture']

    print(f'\n[{junction}] {hf_id}  (~{size_mb} MB)')

    if model_already_exists(folder):
        print(f'  Already exists at ./models/{folder} — skipping.')
        skipped += 1
        continue

    try:
        if entry['type'] == 'image_classification':
            download_image_classification_model(hf_id, folder)

        print(f'  Saved -> ./models/{folder}')
        downloaded += 1

    except Exception as e:
        print(f'  ERROR: {e}')
        failed += 1

print('\n' + '=' * 60)
print(f'  Done.')
print(f'  Downloaded : {downloaded}')
print(f'  Skipped    : {skipped}  (already existed)')
print(f'  Failed     : {failed}')
print(f'\n  All models are in: {os.path.abspath(MODELS_DIR)}')
print('=' * 60)