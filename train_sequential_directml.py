#!/usr/bin/env python3
"""
FinAI Sequential Training Script [DirectML Version]
Trains datasets ONE AT A TIME using AMD GPUs on Windows, via torch-directml (no HF Accelerate)
Auto-selects CUDA (NVIDIA), DirectML (AMD/Windows), or CPU fallback.
"""
import os
import sys
import time
import csv
import subprocess
from datetime import datetime
import torch
try:
import torch_directml
DML_AVAILABLE = True
except ImportError:
DML_AVAILABLE = False
from datasets import load_dataset
from src.core.finai import FinAI
from src.config import Config

def get_device():
if torch.cuda.is_available():
print('[FinAI] Using CUDA (NVIDIA GPU) for training.')
return torch.device('cuda')
elif DML_AVAILABLE:
print('[FinAI] Using DirectML (AMD GPU/Windows) for training.')
# Create fresh DirectML device to avoid suspension issues
try:
device = torch_directml.device()
# Test the device with a small tensor
test_tensor = torch.tensor([1.0]).to(device)
del test_tensor
return device
except Exception as e:
print(f'[FinAI] DirectML device failed to initialize: {e}')
print('[FinAI] Falling back to CPU.')
return torch.device('cpu')
else:
print('[FinAI] No GPU found. Training on CPU.')
return torch.device('cpu')

DEVICE = get_device()

DATASET_HEADERS = ['name', 'config', 'split', 'date_trained', 'model_path', 'status']
def load_datasets_csv(file_path):
if not os.path.exists(file_path):
with open(file_path, 'w', newline='', encoding='utf-8') as f:
writer = csv.DictWriter(f, fieldnames=DATASET_HEADERS)
writer.writeheader()
return []
with open(file_path, 'r', newline='', encoding='utf-8') as f:
reader = csv.DictReader(f)
return list(reader)
def save_datasets_csv(file_path, datasets):
with open(file_path, 'w', newline='', encoding='utf-8') as f:
writer = csv.DictWriter(f, fieldnames=DATASET_HEADERS)
writer.writeheader()
writer.writerows(datasets)
def update_dataset_status(dataset_name, status, model_path=None):
datasets = load_datasets_csv('datasets.csv')
trained_datasets = load_datasets_csv('trained_datasets.csv')
for dataset in datasets[:]:
if dataset['name'] == dataset_name:
dataset['status'] = status
dataset['date_trained'] = time.strftime('%Y-%m-%d %H:%M:%S')
if model_path:
dataset['model_path'] = model_path
if status == 'completed':
trained_datasets.append(dataset)
datasets.remove(dataset)
break
save_datasets_csv('datasets.csv', datasets)
save_datasets_csv('trained_datasets.csv', trained_datasets)
def extract_text_from_dataset(dataset, split='train'):
texts = []
try:
if split and split in dataset:
data = dataset[split]
else:
data = dataset[list(dataset.keys())[0]]
text_fields = ['text', 'input', 'question', 'instruction', 'content', 'prompt', 'query', 'answer', 'response', 'output']
for item in data:
text = None
for field in text_fields:
if field in item and item[field]:
text = item[field]
if isinstance(text, str) and len(text.strip()) > 10:
break
if not text or not isinstance(text, str) or len(text.strip()) < 10:
for key, value in item.items():
if isinstance(value, str) and value.strip():
if not text or len(value) > len(text):
text = value
if not text or len(text.strip()) < 10:
text = " ".join([str(v) for k, v in item.items() if isinstance(v, (str, int, float)) and str(v).strip()])
if text and isinstance(text, str) and len(text.strip()) > 10:
texts.append(text.strip())
except Exception as e:
print(f"Error processing dataset: {e}")
return texts
def train_single_dataset(dataset_info, dataset_number, cfg):
dataset_name = dataset_info['name']
dataset_config = dataset_info.get('config') or None
dataset_split = dataset_info.get('split', 'train')
print(f"\n{'='*80}")
print(f"Training on dataset {dataset_number}: {dataset_name} [DirectML/No Accelerate]")
print("="*80)
try:
print("Loading dataset...")
dataset = None
try:
if dataset_config:
dataset = load_dataset(dataset_name, dataset_config)
else:
dataset = load_dataset(dataset_name)
except Exception as e:
print(f"Could not load dataset: {e}")
return False
print(f"Extracting text...")
texts = extract_text_from_dataset(dataset, dataset_split)
if not texts:
print(f"No text data found in {dataset_name}, skipping.")
return False
print(f"Extracted {len(texts):,} text samples")
os.makedirs("datasets", exist_ok=True)
temp_file = f"datasets/temp_dataset_{dataset_number}.txt"
print(f"Writing to temporary file...")
with open(temp_file, 'w', encoding='utf-8') as f:
f.write("\n\n".join(texts))
file_size_mb = os.path.getsize(temp_file) / 1024 / 1024
print(f"Written {file_size_mb:.2f} MB")

# Reset DirectML device before training
print("Resetting DirectML device...")
global DEVICE
if DML_AVAILABLE:
try:
# Force garbage collection and device reset
import gc
gc.collect()
DEVICE = get_device()
print(f"[FinAI] Fresh training device: {DEVICE}")
except Exception as e:
print(f"[FinAI] Device reset failed: {e}")

print(f"\n{'='*80}")
print(f"Training on {dataset_name} [DirectML/No Accelerate]...")
print(f"{'='*80}\n")
finai = FinAI()
print(f"[FinAI] Training device (torch): {DEVICE}")
finai.train_from_file(
temp_file,
steps=cfg.TRAIN_STEPS,
batch_size=cfg.BATCH_SIZE,
learning_rate=cfg.LEARNING_RATE,
grad_accum_steps=cfg.GRADIENT_ACCUM_STEPS,
mixed_precision='bf16' if DEVICE.type != 'cpu' else 'no',
weight_decay=cfg.WEIGHT_DECAY,
warmup_steps=cfg.WARMUP_STEPS,
max_grad_norm=cfg.MAX_GRAD_NORM,
use_accelerate=False,
use_gpu=True,
training_mode='single',
)
update_dataset_status(dataset_name, 'completed', cfg.LANGUAGE_MODEL_PATH)
print(f"\nDataset #{dataset_number} completed: {dataset_name}")
if os.path.exists(temp_file):
os.remove(temp_file)
print("Cleaned up temporary file\n")
return True
except Exception as e:
print(f"\nFailed to train on {dataset_name}: {str(e)}")
import traceback
traceback.print_exc()
return False
def main():
print("\n" + "="*80)
print("FinAI - Sequential Training (One Dataset at a Time) [DirectML]")
print("="*80 + "\n")
datasets = load_datasets_csv("datasets.csv")
trained_datasets = load_datasets_csv("trained_datasets.csv")
trained_names = {d['name'] for d in trained_datasets}
pending_datasets = [d for d in datasets if d['name'] not in trained_names]
if not pending_datasets:
print("No new datasets to train. All datasets have been processed.")
print(" The model has already been trained on all available data.\n")
return
print(f"Found {len(pending_datasets)} pending dataset(s):")
for i, d in enumerate(pending_datasets, 1):
print(f" {i}. {d['name']}")
print()
cfg = Config()
dataset_number = len(trained_datasets) + 1
for dataset_info in pending_datasets:
success = train_single_dataset(dataset_info, dataset_number, cfg)
if success:
dataset_number += 1
else:
print(f"\nWarning: Skipping {dataset_info['name']} due to errors")
print(" Continuing to next dataset...\n")
print("\n" + "="*80)
print("Sequential training completed! [DirectML]")
print(f" Total datasets trained in this session: {dataset_number - len(trained_datasets) - 1}")
print(f" Total datasets in history: {dataset_number - 1}")
print("="*80 + "\n")
if __name__ == "__main__":
try:
main()
except KeyboardInterrupt:
print("\n\nTraining interrupted by user")
sys.exit(1)
except Exception as e:
print(f"\nFatal error: {str(e)}")
import traceback
traceback.print_exc()
sys.exit(1)
