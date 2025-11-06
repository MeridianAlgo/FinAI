#!/usr/bin/env python3
"""
Sort datasets.csv by approximate dataset size (smallest to largest).

This version uses the Hugging Face Hub API to get dataset sizes.
"""
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "datasets.csv"
HEADERS = ['name', 'config', 'split', 'date_trained', 'model_path', 'status']

# Known dataset sizes in bytes (as fallback)
KNOWN_SIZES = {
    # Small datasets
    'yukiarimo/english-vocabulary': 10_000,
    'tner/fin': 50_000,
    'vumichien/financial-sentiment': 100_000,
    'TimKoornstra/financial-tweets-sentiment': 150_000,
    'sjyuxyz/financial-sentiment-analysis': 200_000,
    'LLukas22/fiqa': 1_000_000,
    'virattt/financial-qa-10K': 10_000_000,
    'sujet-ai/Sujet-Finance-Instruct-177k': 20_000_000,
    'Josephgflowers/Finance-Instruct-500k': 50_000_000,
    'FinGPT/fingpt-forecaster-dow30-202305-202405': 100_000_000,
    'nickmuchi/trade-the-event-finance': 150_000_000,
    'emilpartow/reddit_finance_posts_sp500': 200_000_000,
    'zeroshot/twitter-financial-news-sentiment': 250_000_000,
    'sweatSmile/FinanceQA': 300_000_000,
    'PatronusAI/financebench': 350_000_000,
    'lumalik/Quant-Trading-Instruct': 400_000_000,
    'gtfintechlab/finer-ord': 450_000_000,
    'gbharti/finance-alpaca': 500_000_000,
    'FinanceInc/auditor_sentiment': 600_000_000,
    'fka/awesome-chatgpt-prompts': 1_000_000_000,
    'snorkelai/agent-finance-reasoning': 1_500_000_000,
    'Anurich/finance_dataset': 2_000_000_000,
    'Abirate/english_quotes': 2_500_000_000,
}

def get_dataset_size(repo_id: str) -> int:
    """Get dataset size from Hugging Face Hub API or use known sizes."""
    # First check known sizes
    if repo_id in KNOWN_SIZES:
        return KNOWN_SIZES[repo_id]
    
    try:
        # Try to get size from Hugging Face Hub API
        api_url = f"https://huggingface.co/api/datasets/{repo_id}"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Try to get size from dataset card
        if 'cardData' in data and 'size_categories' in data['cardData']:
            size_str = data['cardData']['size_categories'][0].lower()
            if '100k' in size_str:
                return 100_000
            elif '1m' in size_str or '1m<' in size_str:
                return 1_000_000
            elif '10m' in size_str:
                return 10_000_000
            elif '100m' in size_str:
                return 100_000_000
            elif '1b' in size_str or '1b<' in size_str:
                return 1_000_000_000
    except Exception as e:
        print(f"  ⚠️  Could not get size for {repo_id}: {str(e)[:100]}...")
    
    # Default to a large number so it goes to the end
    return 1_000_000_000_000  # 1TB as default for unknown datasets

def read_csv() -> List[Dict[str, str]]:
    """Read datasets from CSV file."""
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found")
        sys.exit(1)
        
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv(rows: List[Dict[str, str]]) -> None:
    """Write datasets to CSV file."""
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

def main():
    print("🔄 Fetching dataset sizes...")
    
    # Read existing datasets
    datasets = read_csv()
    
    # Add size information
    for dataset in tqdm(datasets, desc="Processing datasets"):
        repo_id = dataset['name'].strip()
        dataset['_size'] = get_dataset_size(repo_id)
    
    # Sort by size (smallest first)
    datasets.sort(key=lambda x: x['_size'])
    
    # Remove temporary size field
    for dataset in datasets:
        if '_size' in dataset:
            del dataset['_size']
    
    # Save back to CSV
    write_csv(datasets)
    print(f"✅ Successfully sorted {len(datasets)} datasets by size in {CSV_PATH}")
    
    # Print the order
    print("\nDatasets sorted by size (smallest to largest):")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i:2d}. {dataset['name']}")

if __name__ == "__main__":
    main()
