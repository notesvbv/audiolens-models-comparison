"""
audiolens — j5 populate_models.py

downloads kokoro tts model from huggingface into ./models/
edge tts need no download — they are handled at runtime.

run this once before running j5_tts.py

usage:
    python populate_models.py
"""

import os
from huggingface_hub import snapshot_download

MODELS_DIR  = './models'
KOKORO_ID   = 'hexgrad/Kokoro-82M'
KOKORO_DIR  = os.path.join(MODELS_DIR, 'kokoro-82m')

os.makedirs(MODELS_DIR, exist_ok=True)

print('=' * 60)
print('  audiolens — j5 model downloader')
print('=' * 60)
print(f'  model      : {KOKORO_ID}')
print(f'  size       : ~326 MB')
print(f'  save path  : {os.path.abspath(KOKORO_DIR)}')
print('=' * 60)

# skip if already downloaded
if os.path.isdir(KOKORO_DIR) and len(os.listdir(KOKORO_DIR)) > 0:
    print('\nkokoro already exists — skipping download.')
    print(f'  path: {os.path.abspath(KOKORO_DIR)}')
else:
    print('\ndownloading kokoro...')
    try:
        snapshot_download(repo_id=KOKORO_ID, local_dir=KOKORO_DIR)
        print(f'saved -> {KOKORO_DIR}')
    except Exception as e:
        print(f'error: {e}')
        exit(1)

print('\ndone.')
print('you can now run: python j5_tts.py')