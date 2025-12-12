#!/usr/bin/env python3
"""
Quick status check for FinAI training system
"""

import os
import csv


def main():
    print("\n" + "="*60)
    print("FinAI Training Status")
    print("="*60 + "\n")
    
    # Count datasets in queue
    datasets_count = 0
    next_dataset = None
    if os.path.exists("datasets.csv"):
        with open("datasets.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            datasets = [row for row in reader if row.get('name', '').strip()]
            datasets_count = len(datasets)
            if datasets:
                next_dataset = datasets[0]['name']
    
    # Count trained datasets
    trained_success = 0
    trained_failed = 0
    last_trained = None
    if os.path.exists("trained_datasets.csv"):
        with open("trained_datasets.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows:
                if row.get('status') == 'success':
                    trained_success += 1
                    last_trained = row.get('name')
                elif row.get('status') == 'failed':
                    trained_failed += 1
    
    # Display
    print(f"📋 Datasets in queue: {datasets_count}")
    if next_dataset:
        print(f"🎯 Next to train: {next_dataset}")
    
    print(f"\n✅ Successfully trained: {trained_success}")
    if last_trained:
        print(f"   Last trained: {last_trained}")
    
    print(f"❌ Failed attempts: {trained_failed}")
    
    total = datasets_count + trained_success
    if total > 0:
        progress = (trained_success / total) * 100
        print(f"\n📊 Progress: {trained_success}/{total} ({progress:.1f}%)")
    
    print("\n" + "="*60)
    print("\nFor detailed management, run: python scripts/manage_datasets.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
