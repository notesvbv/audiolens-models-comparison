"""
audiolens — j3 populate_models.py

pre-downloads ocr model weights so j3_ocr.py runs without network delays.

  tesseract  — system binary, installed via conda (not downloaded here)
  easyocr    — english models saved to ./models/easyocr/

run this once before running j3_ocr.py

usage:
    python populate_models.py

install requirements first:
    conda install -c conda-forge tesseract pytesseract
    pip install easyocr
"""

import os
import sys
import subprocess

MODELS_DIR  = './models'
EASYOCR_DIR = os.path.join(MODELS_DIR, 'easyocr')

os.makedirs(EASYOCR_DIR, exist_ok=True)

print('=' * 60)
print('  audiolens — j3 ocr model downloader')
print('=' * 60)


# -- check tesseract binary --

print('\nchecking tesseract...')
result = subprocess.run(['which', 'tesseract'], capture_output=True)
if result.returncode != 0:
    print('[error] tesseract binary not found.')
    print('install it with: conda install -c conda-forge tesseract pytesseract')
    sys.exit(1)

version = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
print(f'tesseract ok — {version.stdout.splitlines()[0]}')


# -- pre-download easyocr english models --

print('\ndownloading easyocr english models...')
print(f'save path: {os.path.abspath(EASYOCR_DIR)}')

try:
    import easyocr
    reader = easyocr.Reader(
        ['en'],
        model_storage_directory=EASYOCR_DIR,
        download_enabled=True,
        verbose=False,
        gpu=False,
    )
    print('easyocr models ready.')
except Exception as e:
    print(f'[error] easyocr download failed: {e}')
    print('make sure easyocr is installed: pip install easyocr')
    sys.exit(1)


print('\n' + '=' * 60)
print('  done. all models are ready.')
print(f'  easyocr  : {os.path.abspath(EASYOCR_DIR)}')
print('  tesseract: system binary')
print('=' * 60)
print('\nyou can now run: python j3_ocr.py')