#!/usr/bin/env python3
from datasets import load_dataset
import sys

def download_and_prepare_all():
    """Download all financial datasets and combine them"""
    
    datasets_to_load = [
        ("PatronusAI/financebench", "train"),
        ("gbharti/finance-alpaca", "train"),
        ("emilpartow/reddit_finance_posts_sp500", "train"),
        ("nickmuchi/trade-the-event-finance", "train"),
        ("FinanceInc/auditor_sentiment", "train"),
    ]
    
    output_file = "combined_financial_training.txt"
    total_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for dataset_name, split in datasets_to_load:
            print(f"\nDownloading {dataset_name}...")
            try:
                ds = load_dataset(dataset_name)
                
                # Get the split
                if split in ds:
                    data = ds[split]
                else:
                    data = ds[list(ds.keys())[0]]
                
                print(f"Processing {len(data)} examples...")
                count = 0
                
                for item in data:
                    # Try different field combinations
                    question = None
                    answer = None
                    
                    # Try common field names
                    if 'question' in item and 'answer' in item:
                        question = item['question']
                        answer = item['answer']
                    elif 'instruction' in item and 'output' in item:
                        question = item['instruction']
                        answer = item['output']
                    elif 'input' in item and 'output' in item:
                        question = item['input']
                        answer = item['output']
                    elif 'text' in item:
                        text = item['text']
                        if 'user:' in text.lower() and 'assistant:' in text.lower():
                            f.write(text.lower() + "\n\n")
                            count += 1
                            continue
                    elif 'title' in item and 'selftext' in item:
                        question = item['title']
                        answer = item['selftext']
                    
                    if question and answer:
                        question = str(question).strip().lower()
                        answer = str(answer).strip().lower()
                        
                        if len(question) > 10 and len(answer) > 10:
                            f.write(f"user: {question}\n")
                            f.write(f"assistant: {answer}\n\n")
                            count += 1
                    
                    if count >= 10000:  # Limit per dataset
                        break
                
                print(f"✓ Added {count} examples from {dataset_name}")
                total_count += count
                
            except Exception as e:
                print(f"✗ Error with {dataset_name}: {e}")
                continue
    
    print(f"\n✓ Total examples: {total_count}")
    print(f"✓ Saved to: {output_file}")
    print(f"\nNow train with: python main.py train {output_file}")
    
    return output_file

if __name__ == "__main__":
    download_and_prepare_all()
