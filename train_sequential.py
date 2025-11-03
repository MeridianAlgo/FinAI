#!/usr/bin/env python3
"""
Train FinAI on datasets sequentially.
Reads datasets from datasets_list.py, trains on each one individually,
and removes them from the list after training completes.
"""
import sys
import os
from datasets import load_dataset
from src.core.finai import FinAI

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
        if split and split in dataset:
            data = dataset[split]
        else:
            # Use first available split
            data = dataset[list(dataset.keys())[0]]
        
        print(f"  Processing {len(data)} examples...")
        
        # Try common text fields
        text_fields = ['text', 'input', 'question', 'instruction', 'content', 'prompt', 'query', 'answer', 'response']
        
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
        print(f"  WARNING: Error processing dataset: {e}")
    
    return texts

def save_dataset_list(remaining_datasets):
    """Save remaining datasets back to datasets_list.py"""
    with open("datasets_list.py", 'w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('Dataset list for sequential training.\n')
        f.write('Add your datasets here, and train_sequential.py will train on each one individually.\n')
        f.write('"""\n\n')
        f.write('# Add all your datasets here in this format:\n')
        f.write('# Each entry is: ("dataset_name", "config_name" or None, "split_name" or None)\n\n')
        f.write('DATASETS = [\n')
        for ds_name, ds_config, ds_split in remaining_datasets:
            config_str = f'"{ds_config}"' if ds_config else 'None'
            split_str = f'"{ds_split}"' if ds_split else 'None'
            f.write(f'    ("{ds_name}", {config_str}, {split_str}),\n')
        f.write(']\n')

def load_dataset_list():
    """Load datasets from datasets_list.py"""
    try:
        from datasets_list import DATASETS
        return DATASETS.copy()
    except ImportError:
        print("ERROR: Could not import DATASETS from datasets_list.py")
        print("   Make sure datasets_list.py exists and contains a DATASETS list")
        return []

def train_on_dataset(dataset_name, config, split, use_gpu):
    """Train on a single dataset"""
    print(f"\n{'=' * 70}")
    print(f"[LOADING] Loading dataset: {dataset_name}")
    if config:
        print(f"   Config: {config}")
    if split:
        print(f"   Split: {split}")
    print(f"{'=' * 70}\n")
    
    try:
        # Load dataset
        if config:
            dataset = load_dataset(dataset_name, config)
        else:
            dataset = load_dataset(dataset_name)
        
        # Extract texts from all splits or specified split
        all_texts = []
        splits_to_process = [split] if split else list(dataset.keys())
        
        for split_name in splits_to_process:
            if split_name in dataset:
                split_texts = extract_text_from_dataset(dataset, split_name)
                all_texts.extend(split_texts)
                print(f"  [OK] Extracted {len(split_texts)} texts from split '{split_name}'")
        
        if len(all_texts) == 0:
            print(f"  [WARNING] No texts extracted from {dataset_name}, skipping...")
            return False
        
        print(f"\n  [INFO] Total texts: {len(all_texts)}")
        
        # Save to temporary file
        temp_file = f"datasets/temp_{dataset_name.replace('/', '_').replace('-', '_')}.txt"
        os.makedirs("datasets", exist_ok=True)
        
        print(f"  [SAVING] Saving to {temp_file}...")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in all_texts:
                f.write(text + "\n")
        
        print(f"  [OK] Saved {len(all_texts)} training examples")
        
        # Train the model
        print(f"\n  [TRAINING] Starting training on {dataset_name}...\n")
        finai = FinAI()
        finai.train_from_file(
            temp_file,
            stream=True,  # Use streaming for large datasets
            auto=True,    # Auto-tune hyperparameters
            use_gpu=use_gpu
        )
        
        # Clean up temporary file
        try:
            os.remove(temp_file)
            print(f"  [OK] Cleaned up temporary file")
        except:
            pass
        
        print(f"\n  [OK] Training completed on {dataset_name}!\n")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error training on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function"""
    print("\n" + "=" * 70)
    print("FinAI - Sequential Dataset Training")
    print("=" * 70)
    print()
    print("This script will train on each dataset individually from datasets_list.py")
    print("Each dataset will be removed from the list after training completes.")
    print()
    
    # Load datasets list
    datasets = load_dataset_list()
    if not datasets:
        return 1
    
    print(f"[INFO] Found {len(datasets)} datasets to train on:")
    for i, (name, config, split) in enumerate(datasets, 1):
        config_str = f" (config: {config})" if config else ""
        split_str = f" (split: {split})" if split else ""
        print(f"  {i}. {name}{config_str}{split_str}")
    print()
    
    # Auto-detect GPU
    print("=" * 70)
    print("[INFO] Detecting GPU...")
    use_gpu = detect_gpu()
    if use_gpu:
        print("[OK] GPU detected and will be used for training")
    else:
        print("[WARNING] No GPU detected, using CPU for training")
    print("=" * 70)
    print()
    
    # Confirm (skip in non-interactive mode)
    try:
        response = input("Start training? (y/n, default: y): ").strip().lower()
        if response and response != 'y':
            print("Training cancelled.")
            return 0
    except (EOFError, KeyboardInterrupt):
        print("\nRunning in non-interactive mode, starting training automatically...")
        print()
    
    # Train on each dataset sequentially
    remaining_datasets = datasets.copy()
    successful = 0
    failed = 0
    
    while remaining_datasets:
        dataset_name, config, split = remaining_datasets[0]
        
        print(f"\n{'=' * 70}")
        print(f"Training on dataset {len(datasets) - len(remaining_datasets) + 1}/{len(datasets)}")
        print(f"{'=' * 70}")
        
        success = train_on_dataset(dataset_name, config, split, use_gpu)
        
        if success:
            successful += 1
            # Remove from list
            remaining_datasets.pop(0)
            # Save updated list
            save_dataset_list(remaining_datasets)
            print(f"[OK] Removed {dataset_name} from datasets_list.py")
        else:
            failed += 1
            # Ask if we should skip or retry
            try:
                response = input(f"\nFailed to train on {dataset_name}. Skip and continue? (y/n, default: y): ").strip().lower()
                if response and response != 'y':
                    print("Stopping training.")
                    break
            except (EOFError, KeyboardInterrupt):
                print("Non-interactive mode: skipping failed dataset and continuing...")
            
            remaining_datasets.pop(0)
            save_dataset_list(remaining_datasets)
        
        if remaining_datasets:
            try:
                response = input(f"\n{len(remaining_datasets)} datasets remaining. Continue? (y/n, default: y): ").strip().lower()
                if response and response != 'y':
                    print("Training stopped by user.")
                    save_dataset_list(remaining_datasets)
                    break
            except (EOFError, KeyboardInterrupt):
                print("Non-interactive mode: continuing automatically...\n")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"[OK] Successfully trained on: {successful} datasets")
    if failed > 0:
        print(f"[ERROR] Failed: {failed} datasets")
    print(f"[INFO] Remaining in list: {len(remaining_datasets)} datasets")
    print("=" * 70)
    
    if len(remaining_datasets) == 0:
        print("\n[SUCCESS] All datasets have been trained!")
        print("You can add more datasets to datasets_list.py and run again.")
    else:
        print(f"\n[INFO] Remaining datasets saved to datasets_list.py")
        print("Run train_sequential.py again to continue training.")
    
    print()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user")
        # Save remaining datasets
        try:
            from datasets_list import DATASETS
            save_dataset_list(DATASETS)
            print("Progress saved to datasets_list.py")
        except:
            pass
        sys.exit(1)

