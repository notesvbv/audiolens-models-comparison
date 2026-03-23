"""
audiolens — j3 ocr text extraction

runs tesseract and easyocr on cord and medocr images in ./dataset/images/
and saves the extracted text to a json file for manual review.

usage:
    python j3_ocr.py
"""

# -- config --

MODELS_DIR   = './models'
RESULTS_DIR  = './results'
IMAGES_DIR   = './dataset/images'
RESULTS_FILE = f'{RESULTS_DIR}/j3_ocr_extractions.json'

MODEL_PATHS = {
    'easyocr': f'{MODELS_DIR}/easyocr',
}


# -- imports --

import os
import sys
import json
import time
import subprocess
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from tqdm import tqdm

os.makedirs(RESULTS_DIR, exist_ok=True)


# -- preflight --

print('=' * 60)
print('  audiolens — j3 ocr text extraction')
print('=' * 60)

tess_check = subprocess.run(['which', 'tesseract'], capture_output=True)
if tess_check.returncode != 0:
    print('[error] tesseract not found.')
    print('install: conda install -c conda-forge tesseract pytesseract')
    sys.exit(1)

if not os.path.isdir(MODEL_PATHS['easyocr']) or len(os.listdir(MODEL_PATHS['easyocr'])) == 0:
    print(f'[error] easyocr models missing: {MODEL_PATHS["easyocr"]}')
    print('run: python populate_models.py')
    sys.exit(1)

if not os.path.isdir(IMAGES_DIR) or len(os.listdir(IMAGES_DIR)) == 0:
    print(f'[error] no images found in {IMAGES_DIR}')
    print('run: python j3_prepare_dataset.py')
    sys.exit(1)

print('preflight passed.')
print('=' * 60)


# -- load images --

image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f'\nfound {len(image_files)} images in {IMAGES_DIR}')


# -- run tesseract --

print('\n' + '=' * 60)
print('  MODEL 1 — Tesseract')
print('=' * 60)

import pytesseract

tess_results = {}

for fname in tqdm(image_files, desc='  Tesseract'):
    img = Image.open(os.path.join(IMAGES_DIR, fname)).convert('RGB')
    t0  = time.time()
    text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
    t1  = time.time()
    tess_results[fname] = {
        'text': text.strip(),
        'ms':   round((t1 - t0) * 1000, 2),
    }

tess_avg_ms = round(np.mean([r['ms'] for r in tess_results.values()]), 2)
print(f'  done. avg time: {tess_avg_ms} ms/image')


# -- run easyocr --

print('\n' + '=' * 60)
print('  MODEL 2 — EasyOCR')
print('=' * 60)

import easyocr

reader = easyocr.Reader(
    ['en'],
    model_storage_directory=MODEL_PATHS['easyocr'],
    download_enabled=False,
    verbose=False,
    gpu=False,
)

easy_results = {}

for fname in tqdm(image_files, desc='  EasyOCR'):
    img = Image.open(os.path.join(IMAGES_DIR, fname)).convert('RGB')
    img_array = np.array(img)
    t0   = time.time()
    texts = reader.readtext(img_array, detail=0)
    t1   = time.time()
    easy_results[fname] = {
        'text': ' '.join(texts).strip(),
        'ms':   round((t1 - t0) * 1000, 2),
    }

easy_avg_ms = round(np.mean([r['ms'] for r in easy_results.values()]), 2)
print(f'  done. avg time: {easy_avg_ms} ms/image')


# -- save results --

output = {}
for fname in image_files:
    dataset = fname.split('_')[0]
    output[fname] = {
        'dataset':   dataset,
        'tesseract': tess_results[fname]['text'],
        'easyocr':   easy_results[fname]['text'],
        'tesseract_ms': tess_results[fname]['ms'],
        'easyocr_ms':   easy_results[fname]['ms'],
    }

with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f'\n{"=" * 60}')
print(f'  results saved -> {RESULTS_FILE}')
print(f'  {len(image_files)} images processed')
print(f'  tesseract avg: {tess_avg_ms} ms/image')
print(f'  easyocr avg:   {easy_avg_ms} ms/image')
print(f'{"=" * 60}')
print('\nopen the json file to review and compare the extracted text.')
print('done.')
