
import os
import sys
import random
import traceback

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

# List of finance datasets on Hugging Face
# List of finance datasets on Hugging Face
# Add new datasets here!
DATASETS = [
    "financial_phrasebank",
    "zeroshot/twitter-financial-news-sentiment",
    "gbharti/finance-alpaca",
    "nickmuchi/financial-classification",
    "dair-ai/emotion", # Not finance but good for sentiment
    "shawhin/imdb-financial-aspect",
    "takala/financial_phrasebank",
    "nickmuchi/trade-the-event-finance",
    "emilpartow/reddit_finance_posts_sp500",
    "sweatSmile/FinanceQA",
    "PatronusAI/financebench",
    "lumalik/Quant-Trading-Instruct",
]

def get_random_dataset():
    """Pick a random dataset and config"""
    ds_name = random.choice(DATASETS)
    config_name = None
    
    if ds_name == "financial_phrasebank":
        config_name = "sentences_allagree"
    
    return ds_name, config_name

def extract_text(dataset):
    """Extract text from dataset generically"""
    texts = []
    # Try common splits
    split = "train"
    if split not in dataset:
        split = list(dataset.keys())[0]
    
    data = dataset[split]
    
    # Common text fields
    fields = ['text', 'sentence', 'input', 'instruction', 'content', 'headline', 'question', 'answer']
    
    for item in data:
        text = ""
        for f in fields:
            if f in item and isinstance(item[f], str):
                text += item[f] + " "
        
        if len(text.strip()) > 10:
            texts.append(text.strip())
            
    return texts

import csv
from datetime import datetime
import argparse

# ... (imports remain the same)

def log_training(ds_name, config_name):
    """Log the training run to CSV"""
    csv_file = "trained_datasets.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["name", "config", "split", "date_trained", "model_path", "status"])
            
        writer.writerow([
            ds_name,
            config_name if config_name else "default",
            "train",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models/finai_gpt.pt",
            "success"
        ])
    print(f"Logged training to {csv_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-model", action="store_true", help="Force training a new model from scratch")
    args = parser.parse_args()

    print("Starting Daily FinAI Training...")
    
    # Handle New Model Flag
    if args.new_model:
        print("⚠️  --new-model flag detected! Deleting existing model to start fresh...")
        if os.path.exists("models/finai_gpt.pt"):
            os.remove("models/finai_gpt.pt")
        if os.path.exists("models/tokenizer.pkl"):
            os.remove("models/tokenizer.pkl")
        print("Existing model files removed.")

    # 1. Select Dataset
    ds_name, config_name = get_random_dataset()
    print(f"Selected Dataset: {ds_name} (config: {config_name})")
    
    try:
        if config_name:
            dataset = load_dataset(ds_name, config_name)
        else:
            dataset = load_dataset(ds_name)
    except Exception as e:
        print(f"Failed to load {ds_name}: {e}")
        # Fallback
        ds_name = "gbharti/finance-alpaca"
        print(f"Falling back to {ds_name}")
        dataset = load_dataset(ds_name)

    # 2. Extract Text
    texts = extract_text(dataset)
    print(f"Extracted {len(texts)} samples")
    
    if not texts:
        print("No text found!")
        sys.exit(1)
        
    # Save to temp file
    temp_file = "daily_train_data.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts[:5000])) # Limit to 5000 samples for speed
        
    # 3. Train
    print("Initializing FinAI...")
    finai = FinAI()
    
    # Train for limited steps to fit in GH Actions time limits
    # 14M param model (Mini-GPT) is smarter but slower.
    # 100 steps should take ~45-60 mins on CPU.
    steps = 100 
    
    print(f"Training for {steps} steps...")
    finai.train_from_file(
        temp_file,
        steps=steps,
        batch_size=8, # Small batch for CPU
        learning_rate=6e-4, # Adjusted for new model size
        dataset_name=ds_name,
        training_mode='daily_gh'
    )
    
    # 4. Log Training
    log_training(ds_name, config_name)
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    print("Daily training complete!")

if __name__ == "__main__":
    main()
