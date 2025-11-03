#!/usr/bin/env python3
"""
Train FinAI on datasets sequentially.
Reads datasets from datasets.csv, trains on each one individually,
and moves them to trained_datasets.csv after training completes.
"""
import os
import sys
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

import torch
from datasets import load_dataset
from src.core.finai import FinAI

# CSV file headers
DATASET_HEADERS = ['name', 'config', 'split', 'date_trained', 'model_path', 'status']

def detect_gpu():
    """Force CPU usage"""
    return False  # Always use CPU

def load_datasets_csv(file_path):
    """Load datasets from a CSV file"""
    if not os.path.exists(file_path):
        print(f"Creating new CSV file: {file_path}")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=DATASET_HEADERS)
            writer.writeheader()
        return []
    
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_datasets_csv(file_path, datasets):
    """Save datasets to a CSV file"""
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_HEADERS)
        writer.writeheader()
        writer.writerows(datasets)

def extract_text_from_dataset(dataset, split="train"):
    """Extract text from a dataset, handling different formats"""
    texts = []
    
    try:
        if split and split in dataset:
            data = dataset[split]
        else:
            data = dataset[list(dataset.keys())[0]]
        
        print(f"  Processing {len(data)} examples...")
        
        text_fields = ['text', 'input', 'question', 'instruction', 'content', 'prompt', 'query', 'answer', 'response']
        
        for item in data:
            text = None
            
            for field in text_fields:
                if field in item and item[field]:
                    text = item[field]
                    if isinstance(text, str):
                        break
            
            if not text or not isinstance(text, str):
                for key, value in item.items():
                    if isinstance(value, str) and value.strip():
                        if not text or len(value) > len(text):
                            text = value
                
                if not text or len(text) < 10:
                    text = " ".join([str(v) for k, v in item.items() if isinstance(v, (str, int, float)) and str(v).strip()])
            
            if text and isinstance(text, str) and len(text.strip()) > 10:
                texts.append(text.strip())
    
    except Exception as e:
        print(f"  WARNING: Error processing dataset: {e}")
    
    return texts

def train_on_dataset(dataset_info):
    """Train on a single dataset"""
    dataset_name = dataset_info['name']
    dataset_config = dataset_info.get('config') or None
    dataset_split = dataset_info.get('split', 'train')
    
    print(f"\n{'='*80}")
    print(f"Training on dataset: {dataset_name}")
    if dataset_config:
        print(f"Configuration: {dataset_config}")
    print(f"Split: {dataset_split}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)
        
        print("Extracting text from dataset...")
        texts = extract_text_from_dataset(dataset, dataset_split)
        
        if not texts:
            print(f"  WARNING: No text data found in {dataset_name}")
            return False, "No text data found"
            
        print(f"  Extracted {len(texts)} text samples")
        
        print("\nUsing CPU for training")
        device = torch.device("cpu")
        print(f"Using device: {device}")
        
        print("Initializing FinAI...")
        model = FinAI()
        
        # Ensure model is on CPU
        if hasattr(model, 'model') and model.model is not None:
            model.model = model.model.to(device)
        
        print("Starting training (CPU-only) via train_from_file...")
        import tempfile
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.txt') as tf:
            for line in texts:
                tf.write(line.replace('\n', ' ') + '\n')
            tmp_path = tf.name

        try:
            model.train_from_file(tmp_path, use_gpu=False)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        # Save artifacts in a dataset-specific folder
        base_models = 'models'
        src_model = os.path.join(base_models, 'finai_gpt.pt')
        src_token = os.path.join(base_models, 'tokenizer.pkl')
        out_dir = os.path.join(base_models, dataset_name.replace('/', '_'))
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(src_model):
            import shutil
            shutil.copy2(src_model, os.path.join(out_dir, 'finai_gpt.pt'))
        if os.path.exists(src_token):
            import shutil
            shutil.copy2(src_token, os.path.join(out_dir, 'tokenizer.pkl'))
        model_path = out_dir
        
        training_time = (time.time() - start_time) / 60
        print(f"\nTraining completed in {training_time:.2f} minutes")
        
        return True, model_path
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR training on {dataset_name}: {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg

def main():
    """Main training function"""
    print("FinAI Sequential Trainer")
    print("======================")
    
    # Create required directories
    os.makedirs("models", exist_ok=True)
    
    # Main training loop
    while True:
        # Load datasets
        datasets = load_datasets_csv("datasets.csv")
        trained_datasets = load_datasets_csv("trained_datasets.csv")
        
        # Filter out already trained datasets
        pending_datasets = [d for d in datasets if d['name'] not in {t['name'] for t in trained_datasets}]
        
        if not pending_datasets:
            print("\nNo more datasets to train. Exiting...")
            break
            
        # Get the next dataset to train on
        current_dataset = pending_datasets[0]
        
        # Train on the dataset
        success, result = train_on_dataset(current_dataset)

        if success:
            # Update the dataset info only on success
            current_dataset['date_trained'] = time.strftime("%Y-%m-%d %H:%M:%S")
            current_dataset['status'] = 'completed'
            current_dataset['model_path'] = result

            # Add to trained datasets
            trained_datasets.append(current_dataset)
            save_datasets_csv("trained_datasets.csv", trained_datasets)

            # Remove from pending datasets
            datasets = [d for d in datasets if d['name'] != current_dataset['name']]
            save_datasets_csv("datasets.csv", datasets)

            print(f"\nDataset '{current_dataset['name']}' completed successfully")
            print("\n" + "="*80 + "\n")
        else:
            # Failure: do not move to trained_datasets.csv, do not remove from datasets.csv
            print(f"\nDataset '{current_dataset['name']}' failed with error:\n{result}")
            print("Aborting training loop.")
            sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
