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
from pathlib import Path
from datasets import load_dataset
from src.core.finai import FinAI

# CSV file headers
DATASET_HEADERS = ['name', 'config', 'split', 'date_trained', 'model_path', 'status']

def detect_gpu():
    """Auto-detect GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            return True
        try:
            import torch_directml
            if torch_directml.is_available():
                return True
        except ImportError:
            pass
        return False
    except ImportError:
        return False

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
        
        use_gpu = detect_gpu()
        print(f"\nGPU Available: {use_gpu}")
        
        print("Initializing FinAI...")
        model = FinAI(use_gpu=use_gpu)
        
        print("Starting training...")
        model.train(texts)
        
        model_dir = "models"
        model_path = os.path.join(model_dir, dataset_name.replace('/', '_'))
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Saving model to {model_path}")
        model.save_model(model_path)
        
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
        
        # Update the dataset info
        current_dataset['date_trained'] = time.strftime("%Y-%m-%d %H:%M:%S")
        current_dataset['status'] = 'completed' if success else 'failed'
        current_dataset['model_path'] = result if success else ''
        
        # Add to trained datasets
        trained_datasets.append(current_dataset)
        save_datasets_csv("trained_datasets.csv", trained_datasets)
        
        # Remove from pending datasets
        datasets = [d for d in datasets if d['name'] != current_dataset['name']]
        save_datasets_csv("datasets.csv", datasets)
        
        status = "completed successfully" if success else f"failed: {result}"
        print(f"\nDataset '{current_dataset['name']}' {status}")
        print("\n" + "="*80 + "\n")

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
