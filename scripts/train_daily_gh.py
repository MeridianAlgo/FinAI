
import os
import sys
import csv
import traceback
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

try:
    from datasets import load_dataset
    from src.core.finai import FinAI
    from src.config import Config
except ImportError:
    print("Installing dependencies...")
    os.system("pip install torch datasets transformers accelerate")
    from datasets import load_dataset
    from src.core.finai import FinAI
    from src.config import Config


def read_datasets_csv():
    """Read datasets from datasets.csv"""
    datasets = []
    csv_file = "datasets.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return datasets
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('name'):  # Skip empty rows
                datasets.append({
                    'name': row['name'].strip(),
                    'config': row.get('config', '').strip() or None,
                    'split': row.get('split', 'train').strip()
                })
    
    return datasets


def get_next_dataset():
    """Get the next dataset to train from datasets.csv"""
    datasets = read_datasets_csv()
    
    if not datasets:
        print("No datasets found in datasets.csv!")
        return None, None, None
    
    # Return the first dataset in the list
    ds = datasets[0]
    return ds['name'], ds['config'], ds['split']


def extract_text(dataset, split='train'):
    """Extract text from dataset generically"""
    texts = []
    
    # Try to find the right split
    if split not in dataset:
        available_splits = list(dataset.keys())
        if available_splits:
            split = available_splits[0]
        else:
            return texts
    
    data = dataset[split]
    
    # Common text fields to look for
    fields = ['text', 'sentence', 'input', 'instruction', 'content', 'headline', 
              'question', 'answer', 'output', 'response', 'completion', 'prompt',
              'context', 'document', 'passage', 'article', 'body', 'description']
    
    for item in data:
        text = ""
        # Try to extract text from any available field
        for f in fields:
            if f in item:
                value = item[f]
                if isinstance(value, str) and value.strip():
                    text += value.strip() + " "
                elif isinstance(value, list):
                    # Handle list fields
                    for v in value:
                        if isinstance(v, str) and v.strip():
                            text += v.strip() + " "
        
        # Only add if we found meaningful text
        if len(text.strip()) > 10:
            texts.append(text.strip())
            
    return texts


def log_training(ds_name, config_name, status='success', error_msg=''):
    """Log the training run to trained_datasets.csv"""
    csv_file = "trained_datasets.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["name", "config", "split", "date_trained", "model_path", "status", "error"])
            
        writer.writerow([
            ds_name,
            config_name if config_name else "default",
            "train",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models/finai_gpt.pt",
            status,
            error_msg
        ])
    print(f"Logged training to {csv_file}")


def remove_from_datasets_csv(ds_name):
    """Remove a dataset from datasets.csv"""
    csv_file = "datasets.csv"
    temp_file = "datasets_temp.csv"
    
    if not os.path.exists(csv_file):
        return
    
    with open(csv_file, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get('name', '').strip() != ds_name]
    
    with open(temp_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    os.replace(temp_file, csv_file)
    print(f"Removed {ds_name} from datasets.csv")


def move_to_trained(ds_name, config_name):
    """Move dataset from datasets.csv to trained_datasets.csv"""
    remove_from_datasets_csv(ds_name)
    print(f"Moved {ds_name} to trained datasets")


def cycle_datasets():
    """Move all trained datasets back to datasets.csv for another round"""
    trained_file = "trained_datasets.csv"
    datasets_file = "datasets.csv"
    
    if not os.path.exists(trained_file):
        print("No trained_datasets.csv found")
        return
    
    # Read trained datasets
    trained = []
    with open(trained_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('name') and row.get('status') == 'success':
                trained.append(row)
    
    if not trained:
        print("No successful trained datasets to cycle")
        return
    
    # Append to datasets.csv
    with open(datasets_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'config', 'split', 'date_trained', 'model_path', 'status'])
        for row in trained:
            writer.writerow({
                'name': row['name'],
                'config': row.get('config', ''),
                'split': row.get('split', 'train'),
                'date_trained': '',
                'model_path': '',
                'status': ''
            })
    
    # Clear trained_datasets.csv but keep header
    with open(trained_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["name", "config", "split", "date_trained", "model_path", "status", "error"])
    
    print(f"Cycled {len(trained)} datasets back to datasets.csv")


def count_failed_attempts(ds_name):
    """Count how many times a dataset has failed"""
    trained_file = "trained_datasets.csv"
    
    if not os.path.exists(trained_file):
        return 0
    
    count = 0
    with open(trained_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('name') == ds_name and row.get('status') == 'failed':
                count += 1
    
    return count


def main():
    print("Starting Daily FinAI Training...")
    
    # Check if datasets.csv is empty, if so cycle from trained
    datasets = read_datasets_csv()
    if not datasets:
        print("datasets.csv is empty! Cycling trained datasets back...")
        cycle_datasets()
        datasets = read_datasets_csv()
        
        if not datasets:
            print("No datasets available even after cycling. Exiting.")
            sys.exit(1)
    
    # Get next dataset
    ds_name, config_name, split = get_next_dataset()
    
    if not ds_name:
        print("No dataset found to train!")
        sys.exit(1)
    
    print(f"Selected Dataset: {ds_name} (config: {config_name})")
    
    # Try to load and extract text
    try:
        print(f"Loading dataset {ds_name}...")
        if config_name:
            dataset = load_dataset(ds_name, config_name)
        else:
            dataset = load_dataset(ds_name)
        
        print(f"Extracting text from dataset...")
        texts = extract_text(dataset, split)
        print(f"Extracted {len(texts)} samples")
        
        if not texts or len(texts) < 10:
            print(f"No sufficient text found in {ds_name}! Moving to next dataset...")
            
            # Log the failure
            log_training(ds_name, config_name, status='failed', error_msg='No text found')
            
            # Check if this dataset has failed multiple times
            fail_count = count_failed_attempts(ds_name)
            if fail_count >= 3:
                print(f"Dataset {ds_name} has failed {fail_count} times. Deleting from datasets.csv...")
                remove_from_datasets_csv(ds_name)
            else:
                # Move to end of queue (remove and re-add)
                remove_from_datasets_csv(ds_name)
            
            print("Exiting - GitHub Actions will retry with next dataset in next run")
            sys.exit(1)
            
    except Exception as e:
        print(f"Failed to load {ds_name}: {e}")
        print(traceback.format_exc())
        
        # Log the failure
        log_training(ds_name, config_name, status='failed', error_msg=str(e))
        
        # Check if this dataset has failed multiple times
        fail_count = count_failed_attempts(ds_name)
        if fail_count >= 3:
            print(f"Dataset {ds_name} has failed {fail_count} times. Deleting from datasets.csv...")
            remove_from_datasets_csv(ds_name)
        else:
            # Move to end of queue
            remove_from_datasets_csv(ds_name)
        
        print("Exiting - GitHub Actions will retry with next dataset in next run")
        sys.exit(1)
    
    # Save to temp file (limit samples for reasonable training time)
    temp_file = "daily_train_data.txt"
    sample_limit = 10000  # Sufficient for 2-hour training sessions
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts[:sample_limit]))
    
    print(f"Saved {min(len(texts), sample_limit)} samples to {temp_file}")
    
    # Train the model
    print("Initializing FinAI...")
    finai = FinAI()
    
    # Calculate steps for ~2 hours training on GitHub Actions CPU
    # Empirically, ~1000 steps takes about 1 hour on GitHub Actions
    # So 2000 steps ≈ 2 hours
    steps = 2000
    
    print(f"Training for {steps} steps (approximately 2 hours)...")
    try:
        finai.train_from_file(
            temp_file,
            steps=steps,
            batch_size=8,  # Small batch for CPU
            learning_rate=6e-4,
            dataset_name=ds_name,
            training_mode='daily_gh'
        )
        
        # Log successful training
        log_training(ds_name, config_name, status='success')
        
        # Move dataset from datasets.csv to trained
        move_to_trained(ds_name, config_name)
        
        print(f"Successfully trained on {ds_name}!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print(traceback.format_exc())
        log_training(ds_name, config_name, status='failed', error_msg=str(e))
        
        # Check if should delete
        fail_count = count_failed_attempts(ds_name)
        if fail_count >= 3:
            remove_from_datasets_csv(ds_name)
        
        sys.exit(1)
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("Daily training complete!")


if __name__ == "__main__":
    main()
