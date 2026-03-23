"""
audiolens — j1 document classification

benchmarks 3 models on 9 document categories from rvl-cdip.
results saved to results/j1_classification_results.json

models (loaded from ./models/ — run populate_models.py first):
  1. dit-base        (microsoft/dit-base-finetuned-rvlcdip)
  2. efficientnet-b0 (google/efficientnet-b0)
  3. vit-base        (google/vit-base-patch16-224)

usage:
    python j1_classifier.py
    python j1_classifier.py --images-per-class 50
"""

# -- config --

import argparse

parser = argparse.ArgumentParser(description='AudioLens J1 Document Classifier Evaluation')
parser.add_argument('--images-per-class', type=int, default=1000,
                    help='Number of images to sample per class (default: 1000)')
parser.add_argument('--epochs', type=int, default=10,
                    help='Fine-tuning epochs for EfficientNet and ViT (default: 10)')
args = parser.parse_args()

IMAGES_PER_CLASS = args.images_per_class
FINE_TUNE_EPOCHS = args.epochs
BATCH_SIZE       = 16
LEARNING_RATE    = 1e-3
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
SEED             = 42
MODELS_DIR       = './models'
RESULTS_DIR      = './results'
RESULTS_FILE     = f'{RESULTS_DIR}/j1_classification_results.json'

# local model paths (downloaded by populate_models.py)
MODEL_PATHS = {
    'dit':    f'{MODELS_DIR}/dit-base-finetuned-rvlcdip',
    'effnet': f'{MODELS_DIR}/efficientnet-b0',
    'vit':    f'{MODELS_DIR}/vit-base-patch16-224',
}


# -- imports --

import os
import sys
import json
import time
import random
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    EfficientNetForImageClassification,
    ViTForImageClassification,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs(RESULTS_DIR, exist_ok=True)

# tee logger — mirrors print() to both console and a log file
import datetime

class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

LOG_FILE   = f'{RESULTS_DIR}/j1_run_log.txt'
sys.stdout = Tee(LOG_FILE)

# device
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -- preflight check: make sure all model folders exist --

print('=' * 60)
print('  AudioLens — J1 Document Classification')
print('=' * 60)

missing = []
for name, path in MODEL_PATHS.items():
    config_exists  = os.path.exists(os.path.join(path, 'config.json'))
    weights_exist  = (
        os.path.exists(os.path.join(path, 'model.safetensors')) or
        os.path.exists(os.path.join(path, 'pytorch_model.bin'))
    )
    if not config_exists or not weights_exist:
        missing.append(path)

if missing:
    print('\n[ERROR] The following model folders are missing or incomplete:')
    for p in missing:
        print(f'  - {p}')
    print('\nRun this first:')
    print('  python populate_models.py --juncture j1')
    sys.exit(1)

print(f'Device      : {DEVICE}')
print(f'Models dir  : {os.path.abspath(MODELS_DIR)}')
print('All model folders verified.')
print('=' * 60)


# -- class mapping --

SELECTED_CLASSES = {
    0:  'letter',
    1:  'form',
    2:  'email',
    3:  'handwritten',
    4:  'advertisement',
    7:  'specification',
    9:  'news_article',
    10: 'budget',
    11: 'invoice',
}

RVLCDIP_TO_LOCAL = {rvl: local for local, rvl in enumerate(SELECTED_CLASSES.keys())}
LOCAL_TO_NAME    = {local: name for local, (_, name) in enumerate(SELECTED_CLASSES.items())}
CLASS_NAMES      = list(LOCAL_TO_NAME.values())
NUM_CLASSES      = len(CLASS_NAMES)
SELECTED_RVL_IDX = list(SELECTED_CLASSES.keys())

print(f'\n{NUM_CLASSES} document classes:')
for local, name in LOCAL_TO_NAME.items():
    print(f'  [{local}] {name}')


# -- load or stream dataset --
# first run streams rvl-cdip and caches locally; reruns load from cache

DATASET_CACHE = f'{RESULTS_DIR}/dataset_cache.pkl'

if os.path.exists(DATASET_CACHE):
    print(f'\nLoading dataset from local cache: {DATASET_CACHE}')
    with open(DATASET_CACHE, 'rb') as f:
        all_images, all_labels = pickle.load(f)
    print(f'Loaded {len(all_images)} images from cache.')

else:
    print(f'\nStreaming RVL-CDIP — collecting samples across {NUM_CLASSES} classes...')
    print('This only happens once. Will be cached after this run.\n')

    collected = {rvl: [] for rvl in SELECTED_CLASSES}
    counts    = {rvl: 0  for rvl in SELECTED_CLASSES}

    ds_stream = load_dataset('rvl_cdip', split='train', streaming=True, trust_remote_code=True)

    target_total = IMAGES_PER_CLASS * NUM_CLASSES
    pbar = tqdm(total=target_total, desc='  Collecting images', unit='img')
    samples_scanned = 0

    for sample in ds_stream:
        label = sample['label']
        samples_scanned += 1

        if label in collected and counts[label] < IMAGES_PER_CLASS:
            collected[label].append(sample['image'].convert('RGB'))
            counts[label] += 1
            pbar.update(1)
            # show per-class counts every 10 images
            if sum(counts.values()) % 10 == 0:
                status = '  |  '.join(
                    f'{SELECTED_CLASSES[k][:6]}:{counts[k]}'
                    for k in SELECTED_CLASSES
                )
                pbar.set_postfix_str(status)

        if all(c >= IMAGES_PER_CLASS for c in counts.values()):
            break

    pbar.close()

    all_images, all_labels = [], []
    for rvl_idx, images in collected.items():
        local_label = RVLCDIP_TO_LOCAL[rvl_idx]
        all_images.extend(images)
        all_labels.extend([local_label] * len(images))

    with open(DATASET_CACHE, 'wb') as f:
        pickle.dump((all_images, all_labels), f)
    print(f'Dataset cached -> {DATASET_CACHE}')


# -- shuffle and split --

combined = list(zip(all_images, all_labels))
random.shuffle(combined)
all_images, all_labels = map(list, zip(*combined))

total   = len(all_images)
n_train = int(total * TRAIN_RATIO)
n_val   = int(total * VAL_RATIO)
n_test  = total - n_train - n_val

train_images, train_labels = all_images[:n_train],               all_labels[:n_train]
val_images,   val_labels   = all_images[n_train:n_train+n_val],  all_labels[n_train:n_train+n_val]
test_images,  test_labels  = all_images[n_train+n_val:],         all_labels[n_train+n_val:]

print(f'\nDataset split — Train: {int(TRAIN_RATIO*100)}%  Val: {int(VAL_RATIO*100)}%  Test: 15%')


# -- dataset class and dataloaders --

class DocDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_loader = DataLoader(DocDataset(train_images, train_labels, standard_transform),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(DocDataset(val_images,   val_labels,   standard_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()

print(f'DataLoaders ready.')


# -- helpers --

def compute_metrics(true_labels, pred_labels, times_list, size_mb):
    return {
        'accuracy':  round(accuracy_score(true_labels, pred_labels), 4),
        'f1':        round(f1_score(true_labels, pred_labels, average='weighted', zero_division=0), 4),
        'precision': round(precision_score(true_labels, pred_labels, average='weighted', zero_division=0), 4),
        'recall':    round(recall_score(true_labels, pred_labels, average='weighted', zero_division=0), 4),
        'avg_ms':    round(np.mean(times_list) * 1000, 2),
        'size_mb':   size_mb,
    }


def print_metrics(name, m):
    print(f'\n--- {name} ---')
    print(f'  Accuracy          : {m["accuracy"]*100:.2f}%')
    print(f'  Weighted F1       : {m["f1"]*100:.2f}%')
    print(f'  Weighted Precision: {m["precision"]*100:.2f}%')
    print(f'  Weighted Recall   : {m["recall"]*100:.2f}%')
    print(f'  Avg Inference (ms): {m["avg_ms"]}')
    print(f'  Model Size (MB)   : {m["size_mb"]}')


def save_confusion_matrix(true_labels, pred_labels, title, filename, cmap):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap=cmap)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = f'{RESULTS_DIR}/{filename}'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Confusion matrix saved -> {path}')


def run_finetuning(model, model_name, optimizer):
    print(f'\nFine-tuning {model_name} for {FINE_TUNE_EPOCHS} epochs...')
    for epoch in range(FINE_TUNE_EPOCHS):
        model.train()
        train_correct, train_loss = 0, 0.0
        for imgs, labels in tqdm(train_loader, desc=f'  Epoch {epoch+1}/{FINE_TUNE_EPOCHS}', leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(pixel_values=imgs).logits
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                val_correct += (model(pixel_values=imgs).logits.argmax(1) == labels).sum().item()

        print(f'  Epoch {epoch+1}: '
              f'Train={train_correct/n_train:.4f}  '
              f'Val={val_correct/n_val:.4f}  '
              f'Loss={train_loss/len(train_loader):.4f}')


def run_inference(model, use_processor=None):
    preds, times = [], []
    model.eval()
    with torch.no_grad():
        for img in tqdm(test_images, desc='  Inference', leave=False):
            if use_processor:
                inputs = use_processor(images=img, return_tensors='pt').to(DEVICE)
                t0     = time.time()
                logits = model(**inputs).logits
                t1     = time.time()
                pred   = logits[0, SELECTED_RVL_IDX].argmax().item()
            else:
                tensor = standard_transform(img).unsqueeze(0).to(DEVICE)
                t0     = time.time()
                logits = model(pixel_values=tensor).logits
                t1     = time.time()
                pred   = logits.argmax(1).item()
            preds.append(pred)
            times.append(t1 - t0)
    return preds, times


def free_memory(model):
    del model
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()


# -- model 1: dit-base --
# already trained on full rvl-cdip, so inference only — no fine-tuning needed

print('\n' + '=' * 60)
print('  MODEL 1 — DiT-base')
print(f'  Path: {MODEL_PATHS["dit"]}')
print('=' * 60)

dit_processor = AutoImageProcessor.from_pretrained(MODEL_PATHS['dit'])
dit_model     = AutoModelForImageClassification.from_pretrained(MODEL_PATHS['dit'])
dit_model     = dit_model.to(DEVICE)
dit_model.eval()
print(f'Loaded. Params: {sum(p.numel() for p in dit_model.parameters()):,}')

dit_preds, dit_times = run_inference(dit_model, use_processor=dit_processor)
dit_metrics = compute_metrics(test_labels, dit_preds, dit_times, size_mb=343)
print_metrics('DiT-base', dit_metrics)
print('\nPer-class report:')
print(classification_report(test_labels, dit_preds, target_names=CLASS_NAMES, zero_division=0))
save_confusion_matrix(test_labels, dit_preds, 'DiT-base Confusion Matrix',
                      'j1_dit_confusion.png', 'Blues')
free_memory(dit_model)


# -- model 2: efficientnet-b0 --
# backbone frozen, only the classifier head is fine-tuned

print('\n' + '=' * 60)
print('  MODEL 2 — EfficientNet-B0')
print(f'  Path: {MODEL_PATHS["effnet"]}')
print('=' * 60)

effnet_model = EfficientNetForImageClassification.from_pretrained(
    MODEL_PATHS['effnet'],
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
for name, param in effnet_model.named_parameters():
    param.requires_grad = 'classifier' in name
effnet_model  = effnet_model.to(DEVICE)
trainable     = sum(p.numel() for p in effnet_model.parameters() if p.requires_grad)
print(f'Loaded. Trainable params (head only): {trainable:,}')

optimizer_eff = torch.optim.Adam(
    filter(lambda p: p.requires_grad, effnet_model.parameters()), lr=LEARNING_RATE
)
run_finetuning(effnet_model, 'EfficientNet-B0', optimizer_eff)

effnet_preds, effnet_times = run_inference(effnet_model)
effnet_metrics = compute_metrics(test_labels, effnet_preds, effnet_times, size_mb=21)
print_metrics('EfficientNet-B0', effnet_metrics)
print('\nPer-class report:')
print(classification_report(test_labels, effnet_preds, target_names=CLASS_NAMES, zero_division=0))
save_confusion_matrix(test_labels, effnet_preds, 'EfficientNet-B0 Confusion Matrix',
                      'j1_effnet_confusion.png', 'Greens')
free_memory(effnet_model)


# -- model 3: vit-base --
# same freeze-backbone-train-head strategy as efficientnet

print('\n' + '=' * 60)
print('  MODEL 3 — ViT-base')
print(f'  Path: {MODEL_PATHS["vit"]}')
print('=' * 60)

vit_model = ViTForImageClassification.from_pretrained(
    MODEL_PATHS['vit'],
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
for name, param in vit_model.named_parameters():
    param.requires_grad = 'classifier' in name
vit_model = vit_model.to(DEVICE)
trainable = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
print(f'Loaded. Trainable params (head only): {trainable:,}')

optimizer_vit = torch.optim.Adam(
    filter(lambda p: p.requires_grad, vit_model.parameters()), lr=LEARNING_RATE
)
run_finetuning(vit_model, 'ViT-base', optimizer_vit)

vit_preds, vit_times = run_inference(vit_model)
vit_metrics = compute_metrics(test_labels, vit_preds, vit_times, size_mb=330)
print_metrics('ViT-base', vit_metrics)
print('\nPer-class report:')
print(classification_report(test_labels, vit_preds, target_names=CLASS_NAMES, zero_division=0))
save_confusion_matrix(test_labels, vit_preds, 'ViT-base Confusion Matrix',
                      'j1_vit_confusion.png', 'Oranges')
free_memory(vit_model)


# -- final comparison table --

all_results = {
    'DiT-base':        {'metrics': dit_metrics,    'preds': dit_preds},
    'EfficientNet-B0': {'metrics': effnet_metrics, 'preds': effnet_preds},
    'ViT-base':        {'metrics': vit_metrics,    'preds': vit_preds},
}

print('\n' + '=' * 75)
print('  J1 DOCUMENT CLASSIFICATION — FINAL COMPARISON')
print('=' * 75)
print(f"{'Model':<20} {'Accuracy':>12} {'F1':>10} {'Precision':>12} {'Recall':>10} {'ms/img':>8} {'MB':>6}")
print('-' * 75)
for name, data in all_results.items():
    m = data['metrics']
    print(f"{name:<20} {m['accuracy']*100:>11.2f}% {m['f1']*100:>9.2f}% "
          f"{m['precision']*100:>11.2f}% {m['recall']*100:>9.2f}% "
          f"{m['avg_ms']:>8} {m['size_mb']:>6}")
print('=' * 75)


# -- comparison bar chart --

models = list(all_results.keys())
accs   = [d['metrics']['accuracy'] for d in all_results.values()]
f1s    = [d['metrics']['f1']       for d in all_results.values()]
times  = [d['metrics']['avg_ms']   for d in all_results.values()]
sizes  = [d['metrics']['size_mb']  for d in all_results.values()]
colors = ['steelblue', 'seagreen', 'darkorange']

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, values, title, ylabel, ylim in zip(
    axes,
    [accs,  f1s,   times,               sizes],
    ['Accuracy', 'Weighted F1', 'Avg Inference (ms)', 'Model Size (MB)'],
    ['Score',    'Score',       'Milliseconds',        'MB'],
    [(0, 1),     (0, 1),        None,                  None],
):
    bars = ax.bar(models, values, color=colors)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    if ylim:
        ax.set_ylim(*ylim)
    for bar, val in zip(bars, values):
        label = f'{val:.3f}' if isinstance(val, float) else str(val)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(values) * 0.02),
                label, ha='center', va='bottom', fontsize=9)

plt.suptitle('AudioLens J1 — Document Classification Model Comparison',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
chart_path = f'{RESULTS_DIR}/j1_comparison_chart.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nComparison chart saved -> {chart_path}')


# -- winner selection --
# composite score: 50% accuracy + 30% f1 + 20% speed

max_ms    = max(d['metrics']['avg_ms'] for d in all_results.values())
composite = {}
for name, data in all_results.items():
    m               = data['metrics']
    speed_score     = 1 - (m['avg_ms'] / max_ms)
    composite[name] = round((0.50 * m['accuracy']) + (0.30 * m['f1']) + (0.20 * speed_score), 4)

winner = max(composite, key=composite.get)

print('\n' + '=' * 55)
print('  COMPOSITE SCORE  (Acc 50% + F1 30% + Speed 20%)')
print('=' * 55)
for name, score in sorted(composite.items(), key=lambda x: -x[1]):
    marker = '  <-- WINNER' if name == winner else ''
    print(f'  {name:<20}: {score}{marker}')

print(f'\nJ1 WINNER       : {winner}')
print(f'Accuracy        : {all_results[winner]["metrics"]["accuracy"]*100:.2f}%')
print(f'Weighted F1     : {all_results[winner]["metrics"]["f1"]*100:.2f}%')
print(f'Avg Infer (ms)  : {all_results[winner]["metrics"]["avg_ms"]}')
print(f'Size (MB)       : {all_results[winner]["metrics"]["size_mb"]}')


# -- save results to json --

output = {
    'juncture': 'J1 — Document Classification',
    'config': {
        'images_per_class': IMAGES_PER_CLASS,
        'fine_tune_epochs': FINE_TUNE_EPOCHS,
        'train_val_test':   f'{int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/15',
        'device':           str(DEVICE),
        'seed':             SEED,
        'models_dir':       os.path.abspath(MODELS_DIR),
    },
    'classes': CLASS_NAMES,
    'models': {
        name: {
            'metrics':         data['metrics'],
            'composite_score': composite[name],
        }
        for name, data in all_results.items()
    },
    'winner': winner,
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nFull results saved -> {RESULTS_FILE}')
print('\nDone.')