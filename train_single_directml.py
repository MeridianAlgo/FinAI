#!/usr/bin/env python3
"""
FinAI Single Dataset Training Script [DirectML Version]
Train a single dataset using AMD GPUs on Windows via torch-directml (no HF Accelerate)
Auto-selects CUDA (NVIDIA), DirectML (AMD/Windows), or CPU fallback.
"""
import sys
import os
import torch
try:
import torch_directml
DML_AVAILABLE = True
except ImportError:
DML_AVAILABLE = False
from datasets import load_dataset
from src.core.finai import FinAI
from src.config import Config
from scripts.manage_datasets import mark_dataset_as_trained, TRAINED_DATASETS_CSV, get_current_time
import csv

def get_device():
if torch.cuda.is_available():
print('[FinAI] Using CUDA (NVIDIA GPU) for training.')
return torch.device('cuda')
elif DML_AVAILABLE:
print('[FinAI] Using DirectML (AMD GPU/Windows) for training.')
return torch_directml.device()
else:
print('[FinAI] No GPU found. Training on CPU.')
return torch.device('cpu')

DEVICE = get_device()

def extract_text(dataset, split="train"):
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
print(f"Error extracting text: {e}")
return texts

def train_single_dataset(dataset_name):
print("="*80)
print(f"Training on: {dataset_name} [DirectML/AMD/No Accelerate]")
print("="*80)
try:
print(f"\nLoading dataset...")
dataset = load_dataset(dataset_name)
print(f"Extracting text...")
texts = extract_text(dataset)
if not texts:
print(f"Error: No text data found in {dataset_name}")
return False
print(f"Extracted {len(texts):,} text samples")
os.makedirs("datasets", exist_ok=True)
temp_file = f"datasets/temp_single_{dataset_name.replace('/', '_')}.txt"
with open(temp_file, 'w', encoding='utf-8') as f:
f.write("\n\n".join(texts))
print(f"Saved to: {temp_file}")
print(f"\nStarting training [DirectML]...")
finai = FinAI()
cfg = Config()
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
dataset_name=dataset_name,
training_mode='single',
)
if os.path.exists(temp_file):
os.remove(temp_file)
print(f"\n{'='*80}")
print(f"Training completed successfully! [DirectML]")
print(f"{'='*80}")
try:
moved = mark_dataset_as_trained(dataset_name, cfg.LANGUAGE_MODEL_PATH)
if not moved:
row = {
'name': dataset_name,
'config': '',
'split': 'train',
'date_trained': get_current_time(),
'model_path': cfg.LANGUAGE_MODEL_PATH,
'status': 'completed',
}
try:
with open(TRAINED_DATASETS_CSV, 'r', encoding='utf-8') as f:
reader = csv.DictReader(f)
fieldnames = reader.fieldnames or ['name','config','split','date_trained','model_path','status']
rows = list(reader)
except FileNotFoundError:
fieldnames = ['name','config','split','date_trained','model_path','status']
rows = []
rows.append(row)
with open(TRAINED_DATASETS_CSV, 'w', newline='', encoding='utf-8') as f:
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(rows)
print(f"Recorded training in {TRAINED_DATASETS_CSV}")
except Exception as _e:
print(f"Warning: could not update CSV status: {_e}")
return True
except Exception as e:
print(f"\nError: {e}")
import traceback
traceback.print_exc()
return False
def main():
if len(sys.argv) < 2:
print("Usage: python train_single_directml.py <dataset_name>")
print("\nExample:")
print(" python train_single_directml.py yukiarimo/english-vocabulary")
print(" python train_single_directml.py vumichien/financial-sentiment")
sys.exit(1)
dataset_name = sys.argv[1]
success = train_single_dataset(dataset_name)
sys.exit(0 if success else 1)
if __name__ == "__main__":
main()
