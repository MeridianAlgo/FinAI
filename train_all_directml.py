#!/usr/bin/env python3
"""
FinAI Unified Training Script [DirectML Version]
Train models using AMD GPUs on Windows (torch-directml backend, no HF Accelerate)

This script auto-selects CUDA (NVIDIA), DirectML (AMD/Windows), or CPU.
Docs: See README.md 'Training with AMD GPU (DirectML)' section.
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

# Device selection
if torch.cuda.is_available():
DEVICE = torch.device('cuda')
print('[FinAI] Using CUDA (NVIDIA GPU) for training.')
elif DML_AVAILABLE:
DEVICE = torch_directml.device()
print('[FinAI] Using DirectML (AMD GPU/Windows) for training.')
else:
DEVICE = torch.device('cpu')
print('[FinAI] No GPU found. Training on CPU.')

# Helper functions, CSV management, and dataset loaders below — direct copy from train_all.py

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

def main():
print("\n" + "="*80)
print("FinAI - Unified Training on Single Model [DirectML]")
print("="*80 + "\n")

# Load datasets
datasets = load_datasets_csv("datasets.csv")
trained_datasets = load_datasets_csv("trained_datasets.csv")

# Filter out already trained datasets
trained_names = {d['name'] for d in trained_datasets}
pending_datasets = [d for d in datasets if d['name'] not in trained_names]

if not pending_datasets:
print("No new datasets to train. All datasets have been processed.")
print("The model has already been trained on all available data.\n")
return

print(f"Found {len(pending_datasets)} new dataset(s) to add to training:")
for i, d in enumerate(pending_datasets, 1):
print(f" {i}. {d['name']}")
print()

# Combine all pending datasets
combined_text = []
successful_datasets = []

for dataset_info in pending_datasets:
dataset_name = dataset_info['name']
dataset_config = dataset_info.get('config') or None
dataset_split = dataset_info.get('split', 'train')
print(f"\nLoading dataset: {dataset_name}")
print("-" * 50)
try:
# Load the dataset
dataset = None
try:
if dataset_config:
dataset = load_dataset(dataset_name, dataset_config)
else:
dataset = load_dataset(dataset_name)
except Exception as e:
print(f"Could not load dataset: {e}")
continue
# Extract text
print(f"Extracting text from {dataset_name}...")
texts = extract_text_from_dataset(dataset, dataset_split)
if not texts:
print(f"Warning: No text data found in {dataset_name}, skipping.")
continue
print(f"Extracted {len(texts):,} text samples")
combined_text.extend(texts)
successful_datasets.append(dataset_info)
except Exception as e:
print(f"Error loading {dataset_name}: {str(e)}")
print(f"Skipping this dataset...")
continue

if not combined_text:
print("\nERROR: No valid training data found in any dataset")
return
print(f"\n{'='*80}")
print(f"Total text samples across all datasets: {len(combined_text):,}")
print(f"{'='*80}\n")

# Save combined text to temporary file in datasets folder
os.makedirs("datasets", exist_ok=True)
temp_file = "datasets/combined_training_data.txt"
print(f"Writing combined data to {temp_file}...")
with open(temp_file, 'w', encoding='utf-8') as f:
f.write("\n\n".join(combined_text))
print(f"Written {os.path.getsize(temp_file) / 1024 / 1024:.2f} MB\n")

# Train using FinAI (which handles single model persistence)
finai = FinAI()
cfg = Config()

try:
print(f"[FinAI] Training device (torch): {DEVICE}")
# Optionally, ensure model/data gets placed on DEVICE in FinAI implementation
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
use_accelerate=False, # Bypass accelerate logic in FinAI
)
# Update status for all successful datasets
model_path = cfg.LANGUAGE_MODEL_PATH
trained_datasets = load_datasets_csv("trained_datasets.csv")
dataset_number = len(trained_datasets) + 1
for i, dataset_info in enumerate(successful_datasets, start=dataset_number):
update_dataset_status(dataset_info['name'], 'completed', model_path)
print(f"\nDataset #{i} completed: {dataset_info['name']}")
print("\n" + "="*80)
print("Training completed successfully [DirectML]")
print(f"Model saved to: {model_path}")
print(f"Datasets trained: {len(successful_datasets)}")
print(f"Total datasets in history: {len(trained_datasets) + len(successful_datasets)}")
print("="*80 + "\n")
except Exception as e:
print(f"\nTraining failed: {str(e)}")
import traceback
traceback.print_exc()
return
finally:
# Clean up temporary file
if os.path.exists(temp_file):
os.remove(temp_file)
print(f"Cleaned up temporary file: {temp_file}\n")

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
