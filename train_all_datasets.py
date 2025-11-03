#!/usr/bin/env python3
"""
Train FinAI on multiple Hugging Face financial datasets automatically.
This script downloads, combines, and trains on all specified datasets.
"""
import sys
import os
from datasets import load_dataset
from src.core.finai import FinAI
from src.data.tokenizer import Tokenizer
from src.data.dataset_loader import DatasetLoader

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

def extract_text_from_dataset(dataset, split="train"):
    """Extract text from a dataset, handling different formats"""
    texts = []
    
    try:
        if split in dataset:
            data = dataset[split]
        else:
            # Use first available split
            data = dataset[list(dataset.keys())[0]]
        
        print(f"  Processing {len(data)} examples...")
        
        # Try common text fields
        text_fields = ['text', 'input', 'question', 'instruction', 'content', 'prompt', 'query']
        
        for item in data:
            text = None
            
            # Try to find a text field
            for field in text_fields:
                if field in item and item[field]:
                    text = item[field]
                    if isinstance(text, str):
                        break
            
            # If no standard field, try to combine string fields
            if not text or not isinstance(text, str):
                # Look for any string field
                for key, value in item.items():
                    if isinstance(value, str) and value.strip():
                        # Prefer longer strings
                        if not text or len(value) > len(text):
                            text = value
                
                # If still no text, combine all string values
                if not text or len(text) < 10:
                    text = " ".join([str(v) for k, v in item.items() if isinstance(v, (str, int, float)) and str(v).strip()])
            
            if text and isinstance(text, str) and len(text.strip()) > 10:  # Minimum 10 chars
                texts.append(text.strip())
    
    except Exception as e:
        print(f"  ⚠️  Error processing dataset: {e}")
    
    return texts

def download_and_prepare_datasets():
    """Download all datasets and combine them"""
    print("=" * 70)
    print("📥 Downloading Financial Datasets")
    print("=" * 70)
    print()
    
    # Define all datasets with their configs/splits
    datasets_config = [
        {
            "name": "AdaptLLM/finance-tasks",
            "config": "ConvFinQA",
            "split": None,  # Will use default
            "description": "ConvFinQA financial QA dataset"
        },
        {
            "name": "FinGPT/fingpt-forecaster-dow30-202305-202405",
            "config": None,
            "split": None,
            "description": "FinGPT Dow30 forecaster dataset"
        },
        {
            "name": "Josephgflowers/Finance-Instruct-500k",
            "config": None,
            "split": None,
            "description": "Finance Instruct 500k (requires login)"
        },
        {
            "name": "sujet-ai/Sujet-Finance-Instruct-177k",
            "config": None,
            "split": None,
            "description": "Sujet Finance Instruct 177k"
        },
        {
            "name": "virattt/financial-qa-10K",
            "config": None,
            "split": None,
            "description": "Financial QA 10K"
        },
    ]
    
    all_texts = []
    successful = 0
    failed = 0
    
    for i, ds_config in enumerate(datasets_config, 1):
        print(f"[{i}/{len(datasets_config)}] Loading {ds_config['name']}...")
        print(f"  Description: {ds_config['description']}")
        
        try:
            # Load dataset
            if ds_config['config']:
                dataset = load_dataset(ds_config['name'], ds_config['config'])
            else:
                dataset = load_dataset(ds_config['name'])
            
            # Extract texts from all splits
            for split_name in dataset.keys():
                split_texts = extract_text_from_dataset(dataset, split_name)
                all_texts.extend(split_texts)
                print(f"  ✓ Extracted {len(split_texts)} texts from split '{split_name}'")
            
            successful += 1
            print(f"  ✓ Successfully loaded {ds_config['name']}\n")
        
        except Exception as e:
            failed += 1
            print(f"  ❌ Failed to load {ds_config['name']}: {str(e)}\n")
            continue
    
    print("=" * 70)
    print(f"✓ Downloaded {successful} datasets successfully")
    if failed > 0:
        print(f"⚠️  {failed} datasets failed to load")
    print(f"📊 Total texts collected: {len(all_texts)}")
    print("=" * 70)
    print()
    
    if len(all_texts) == 0:
        print("❌ No texts extracted from any dataset!")
        return None
    
    # Save combined dataset to file
    output_file = "datasets/combined_all_datasets.txt"
    os.makedirs("datasets", exist_ok=True)
    
    print(f"💾 Saving combined dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + "\n")
    
    print(f"✓ Saved {len(all_texts)} training examples to {output_file}\n")
    return output_file

def main():
    """Main training function"""
    print("\n" + "=" * 70)
    print("🤖 FinAI - Automatic Multi-Dataset Training")
    print("=" * 70)
    print()
    
    # Step 1: Download and combine all datasets
    dataset_file = download_and_prepare_datasets()
    if not dataset_file:
        print("❌ Failed to prepare datasets. Exiting.")
        return 1
    
    # Step 2: Auto-detect GPU
    print("=" * 70)
    print("🔍 Detecting GPU...")
    use_gpu = detect_gpu()
    if use_gpu:
        print("✓ GPU detected and will be used for training")
    else:
        print("⚠️  No GPU detected, using CPU for training")
    print("=" * 70)
    print()
    
    # Step 3: Train the model
    print("=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    print()
    
    try:
        finai = FinAI()
        finai.train_from_file(
            dataset_file,
            stream=True,  # Use streaming for large datasets
            auto=True,    # Auto-tune hyperparameters
            use_gpu=use_gpu
        )
        
        print("\n" + "=" * 70)
        print("✓ Training completed successfully!")
        print("=" * 70)
        print("\nYou can now use the trained model:")
        print("  python main.py chat")
        print("  python main.py generate \"your prompt here\"")
        print()
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

