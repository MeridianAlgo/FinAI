#!/usr/bin/env python3
"""
Helper script to manage datasets.csv and trained_datasets.csv
"""

import os
import csv
from datetime import datetime


def show_status():
    """Show current status of datasets"""
    print("\n" + "="*60)
    print("DATASET STATUS")
    print("="*60)
    
    # Count datasets in datasets.csv
    datasets_count = 0
    if os.path.exists("datasets.csv"):
        with open("datasets.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            datasets = [row for row in reader if row.get('name', '').strip()]
            datasets_count = len(datasets)
            
        print(f"\n📋 Datasets waiting to train: {datasets_count}")
        if datasets_count > 0:
            print("\nNext 5 datasets:")
            for i, ds in enumerate(datasets[:5], 1):
                print(f"  {i}. {ds['name']}")
    
    # Count trained datasets
    trained_success = 0
    trained_failed = 0
    if os.path.exists("trained_datasets.csv"):
        with open("trained_datasets.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'success':
                    trained_success += 1
                elif row.get('status') == 'failed':
                    trained_failed += 1
    
    print(f"\n✅ Successfully trained: {trained_success}")
    print(f"❌ Failed datasets: {trained_failed}")
    print(f"\n📊 Total progress: {trained_success}/{datasets_count + trained_success} datasets")
    print("="*60 + "\n")


def cycle_datasets():
    """Move all successfully trained datasets back to datasets.csv"""
    trained_file = "trained_datasets.csv"
    datasets_file = "datasets.csv"
    
    if not os.path.exists(trained_file):
        print("❌ No trained_datasets.csv found")
        return
    
    # Read successful trained datasets
    trained = []
    failed = []
    with open(trained_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('name'):
                if row.get('status') == 'success':
                    trained.append(row)
                elif row.get('status') == 'failed':
                    failed.append(row)
    
    if not trained:
        print("❌ No successful trained datasets to cycle")
        return
    
    print(f"\n🔄 Cycling {len(trained)} successfully trained datasets back to datasets.csv...")
    
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
    
    # Keep only failed datasets in trained_datasets.csv
    with open(trained_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'config', 'split', 'date_trained', 'model_path', 'status', 'error'])
        writer.writeheader()
        writer.writerows(failed)
    
    print(f"✅ Cycled {len(trained)} datasets")
    print(f"📝 Kept {len(failed)} failed datasets in trained_datasets.csv for review")


def clean_failed():
    """Remove all failed datasets from trained_datasets.csv"""
    trained_file = "trained_datasets.csv"
    
    if not os.path.exists(trained_file):
        print("❌ No trained_datasets.csv found")
        return
    
    # Read and filter
    kept = []
    removed = 0
    with open(trained_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') == 'success':
                kept.append(row)
            else:
                removed += 1
    
    # Write back
    with open(trained_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'config', 'split', 'date_trained', 'model_path', 'status', 'error'])
        writer.writeheader()
        writer.writerows(kept)
    
    print(f"🗑️  Removed {removed} failed datasets")
    print(f"✅ Kept {len(kept)} successful datasets")


def reset_all():
    """Reset everything - move all datasets back to datasets.csv and clear trained"""
    print("\n⚠️  WARNING: This will reset all training progress!")
    response = input("Are you sure? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    # First cycle successful ones
    cycle_datasets()
    
    # Then clear trained_datasets.csv completely
    with open("trained_datasets.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'config', 'split', 'date_trained', 'model_path', 'status', 'error'])
    
    print("✅ Reset complete! All datasets are back in datasets.csv")


def main():
    print("\n🤖 FinAI Dataset Manager")
    
    while True:
        show_status()
        
        print("Options:")
        print("  1. Show status (refresh)")
        print("  2. Cycle datasets (move trained back to queue)")
        print("  3. Clean failed datasets")
        print("  4. Reset all (move everything back)")
        print("  5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            continue
        elif choice == '2':
            cycle_datasets()
        elif choice == '3':
            clean_failed()
        elif choice == '4':
            reset_all()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid option")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
