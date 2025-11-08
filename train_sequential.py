#!/usr/bin/env python3
"""
FinAI Sequential Training Script
Trains datasets ONE AT A TIME (not combined) into the single model
Automatically tracks progress and commits to git after each dataset
"""
import os
import sys
import time
import csv
import subprocess
from datetime import datetime

import torch
from datasets import load_dataset
from src.core.finai import FinAI
from src.config import Config

# CSV file headers
DATASET_HEADERS = ['name', 'config', 'split', 'date_trained', 'model_path', 'status']

def get_hf_token() -> str | None:
    """Return a Hugging Face token if available via env or local store."""
    try:
        from huggingface_hub import HfFolder
        return os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN') or HfFolder.get_token()
    except Exception:
        return os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

def load_datasets_csv(file_path):
    """Load datasets from a CSV file"""
    if not os.path.exists(file_path):
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

def update_dataset_status(dataset_name, status, model_path=None):
    """Update the status of a dataset"""
    datasets = load_datasets_csv("datasets.csv")
    trained_datasets = load_datasets_csv("trained_datasets.csv")
    
    for dataset in datasets[:]:
        if dataset['name'] == dataset_name:
            dataset['status'] = status
            dataset['date_trained'] = time.strftime("%Y-%m-%d %H:%M:%S")
            if model_path:
                dataset['model_path'] = model_path
            
            if status == 'completed':
                # Move to trained_datasets.csv
                trained_datasets.append(dataset)
                datasets.remove(dataset)
                break
    
    save_datasets_csv("datasets.csv", datasets)
    save_datasets_csv("trained_datasets.csv", trained_datasets)

def git_commit_and_push(dataset_number):
    """Commit and push to git after dataset training"""
    try:
        print(f"\n{'='*80}")
        print("Committing to Git...")
        print("="*80)
        
        # Git add all changes
        result = subprocess.run(['git', 'add', '.'], 
                              capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"Git add failed: {result.stderr}")
            return False
        
        # Git commit with message
        commit_msg = f"Model Dataset #{dataset_number} Trained"
        result = subprocess.run(['git', 'commit', '-m', commit_msg],
                              capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            if 'nothing to commit' in result.stdout:
                print("No changes to commit")
                return True
            else:
                print(f"Git commit failed: {result.stderr}")
                return False
        
        print(f"Committed: {commit_msg}")
        
        # Git push
        result = subprocess.run(['git', 'push', 'origin', 'main'],
                              capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            # Try 'master' if 'main' fails
            result = subprocess.run(['git', 'push', 'origin', 'master'],
                                  capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                print(f"Git push failed: {result.stderr}")
                print("Tip: Commit was successful, but push failed.")
                print("      Check your remote configuration and network.")
                return False
        
        print("Pushed to remote repository")
        print("="*80 + "\n")
        return True
        
    except FileNotFoundError:
        print("Git not found. Please install git or commit manually.")
        return False
    except Exception as e:
        print(f"Git operation failed: {e}")
        return False

def extract_text_from_dataset(dataset, split="train"):
    """Extract text from a dataset, handling different formats"""
    texts = []
    
    try:
        if split and split in dataset:
            data = dataset[split]
        else:
            data = dataset[list(dataset.keys())[0]]
        
        print(f"Processing {len(data)} examples...")
        
        # Try multiple common field names
        text_fields = ['text', 'input', 'question', 'instruction', 'content', 'prompt', 'query', 'answer', 'response', 'output']
        
        for item in data:
            text = None
            
            # Try known fields first
            for field in text_fields:
                if field in item and item[field]:
                    text = item[field]
                    if isinstance(text, str) and len(text.strip()) > 10:
                        break
            
            # Fallback: find longest string field
            if not text or not isinstance(text, str) or len(text.strip()) < 10:
                for key, value in item.items():
                    if isinstance(value, str) and value.strip():
                        if not text or len(value) > len(text):
                            text = value
            
            # Last resort: concatenate all fields
            if not text or len(text.strip()) < 10:
                text = " ".join([str(v) for k, v in item.items() if isinstance(v, (str, int, float)) and str(v).strip()])
            
            if text and isinstance(text, str) and len(text.strip()) > 10:
                texts.append(text.strip())
    
    except Exception as e:
        print(f"Warning: Error processing dataset: {e}")
    
    return texts

def train_single_dataset(dataset_info, dataset_number, cfg):
    """Train on a single dataset"""
    dataset_name = dataset_info['name']
    dataset_config = dataset_info.get('config') or None
    dataset_split = dataset_info.get('split', 'train')
    
    print(f"\n{'='*80}")
    print(f"Training on dataset {dataset_number}: {dataset_name}")
    print("="*80)
    
    try:
        # Load the dataset
        print("Loading dataset...")
        token = get_hf_token()
        dataset = None
        try:
            if dataset_config:
                try:
                    dataset = load_dataset(dataset_name, dataset_config, token=token) if token else load_dataset(dataset_name, dataset_config)
                except TypeError:
                    # Older datasets versions
                    dataset = load_dataset(dataset_name, dataset_config, use_auth_token=token) if token else load_dataset(dataset_name, dataset_config)
            else:
                try:
                    dataset = load_dataset(dataset_name, token=token) if token else load_dataset(dataset_name)
                except TypeError:
                    dataset = load_dataset(dataset_name, use_auth_token=token) if token else load_dataset(dataset_name)
        except Exception as e:
            print(f"Could not load dataset (auth or access issue?): {e}")
            return False
        
        # Extract text
        print(f"Extracting text...")
        texts = extract_text_from_dataset(dataset, dataset_split)
        
        if not texts:
            print(f"No text data found in {dataset_name}, skipping.")
            return False
        
        print(f"Extracted {len(texts):,} text samples")
        
        # Save to temporary file in datasets folder
        os.makedirs("datasets", exist_ok=True)
        temp_file = f"datasets/temp_dataset_{dataset_number}.txt"
        print(f"Writing to temporary file...")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(texts))
        file_size_mb = os.path.getsize(temp_file) / 1024 / 1024
        print(f"Written {file_size_mb:.2f} MB")
        
        # Train using FinAI
        print(f"\n{'='*80}")
        print(f"Training on {dataset_name}...")
        print(f"{'='*80}\n")
        
        finai = FinAI()
        
        try:
            finai.train_from_file(
                temp_file,
                steps=cfg.TRAIN_STEPS,
                batch_size=cfg.BATCH_SIZE,
                learning_rate=cfg.LEARNING_RATE,
                use_accelerate='auto',
                grad_accum_steps=cfg.GRADIENT_ACCUM_STEPS,
                mixed_precision='auto',
                weight_decay=cfg.WEIGHT_DECAY,
                warmup_steps=cfg.WARMUP_STEPS,
                max_grad_norm=cfg.MAX_GRAD_NORM
            )
            
            # Update status
            update_dataset_status(dataset_name, 'completed', cfg.LANGUAGE_MODEL_PATH)
            
            print(f"\nDataset #{dataset_number} completed: {dataset_name}")
            
            # Git commit and push
            git_commit_and_push(dataset_number)
            
            return True
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file\n")
    
    except Exception as e:
        print(f"\nFailed to train on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function - trains datasets ONE AT A TIME"""
    print("\n" + "="*80)
    print("FinAI - Sequential Training (One Dataset at a Time)")
    print("="*80 + "\n")
    
    # Load datasets
    datasets = load_datasets_csv("datasets.csv")
    trained_datasets = load_datasets_csv("trained_datasets.csv")
    
    # Filter out already trained datasets
    trained_names = {d['name'] for d in trained_datasets}
    pending_datasets = [d for d in datasets if d['name'] not in trained_names]
    
    if not pending_datasets:
        print("No new datasets to train. All datasets have been processed.")
        print("  The model has already been trained on all available data.\n")
        return
    
    print(f"Found {len(pending_datasets)} pending dataset(s):")
    for i, d in enumerate(pending_datasets, 1):
        print(f"  {i}. {d['name']}")
    print()
    
    cfg = Config()
    
    # Train each dataset one at a time
    dataset_number = len(trained_datasets) + 1
    
    for dataset_info in pending_datasets:
        success = train_single_dataset(dataset_info, dataset_number, cfg)
        
        if success:
            dataset_number += 1
        else:
            print(f"\nWarning: Skipping {dataset_info['name']} due to errors")
            print("   Continuing to next dataset...\n")
    
    print("\n" + "="*80)
    print("Sequential training completed!")
    print(f"   Total datasets trained in this session: {dataset_number - len(trained_datasets) - 1}")
    print(f"   Total datasets in history: {dataset_number - 1}")
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
