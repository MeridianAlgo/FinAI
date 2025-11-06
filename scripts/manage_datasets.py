#!/usr/bin/env python3
import csv
import os
from datetime import datetime
from pathlib import Path

# File paths
DATASETS_CSV = os.path.join(Path(__file__).parent.parent, 'datasets.csv')
TRAINED_DATASETS_CSV = os.path.join(Path(__file__).parent.parent, 'trained_datasets.csv')

def read_csv(filepath):
    """Read a CSV file and return its rows as a list of dictionaries."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(filepath, rows, fieldnames=None):
    """Write rows to a CSV file."""
    if not fieldnames and rows:
        fieldnames = rows[0].keys()
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def get_current_time():
    """Get current timestamp in YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def mark_dataset_as_trained(dataset_name, model_path=None):
    """
    Mark a dataset as trained by moving it from datasets.csv to trained_datasets.csv
    and adding a timestamp.
    """
    # Read both CSV files
    datasets = read_csv(DATASETS_CSV)
    trained_datasets = read_csv(TRAINED_DATASETS_CSV)
    
    # Find the dataset in datasets.csv
    dataset_to_move = None
    updated_datasets = []
    
    for dataset in datasets:
        if dataset['name'] == dataset_name:
            dataset_to_move = dataset
            # Update the dataset with training info
            dataset_to_move['date_trained'] = get_current_time()
            dataset_to_move['status'] = 'completed'
            if model_path:
                dataset_to_move['model_path'] = model_path
            trained_datasets.append(dataset_to_move)
        else:
            updated_datasets.append(dataset)
    
    if dataset_to_move:
        # Save the updated files
        if updated_datasets:
            write_csv(DATASETS_CSV, updated_datasets)
        else:
            # If all datasets are moved, create an empty file with header
            with open(DATASETS_CSV, 'w', newline='', encoding='utf-8') as f:
                f.write('name,config,split,date_trained,model_path,status\n')
        
        # Ensure trained_datasets has the correct field order
        fieldnames = ['name', 'config', 'split', 'date_trained', 'model_path', 'status']
        write_csv(TRAINED_DATASETS_CSV, trained_datasets, fieldnames)
        
        print(f"Successfully marked {dataset_name} as trained at {dataset_to_move['date_trained']}")
        return True
    else:
        print(f"Dataset {dataset_name} not found in {DATASETS_CSV}")
        return False

def list_datasets():
    """List all datasets and their training status."""
    datasets = read_csv(DATASETS_CSV)
    trained_datasets = {d['name']: d for d in read_csv(TRAINED_DATASETS_CSV)}
    
    print("\n=== Datasets ===")
    print("Available datasets (not yet trained):")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset['name']} (Status: {dataset.get('status', 'pending')})")
    
    print("\nTrained datasets:")
    for i, (name, dataset) in enumerate(trained_datasets.items(), 1):
        date = dataset.get('date_trained', 'Unknown date')
        print(f"  {i}. {name} (Trained on: {date}, Status: {dataset.get('status', 'completed')})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage training datasets")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Mark dataset as trained
    mark_parser = subparsers.add_parser('mark_trained', help='Mark a dataset as trained')
    mark_parser.add_argument('dataset_name', help='Name of the dataset to mark as trained')
    mark_parser.add_argument('--model-path', help='Path to the trained model')
    
    # List datasets
    subparsers.add_parser('list', help='List all datasets and their status')
    
    args = parser.parse_args()
    
    if args.command == 'mark_trained':
        mark_dataset_as_trained(args.dataset_name, args.model_path)
    elif args.command == 'list':
        list_datasets()
    else:
        parser.print_help()
