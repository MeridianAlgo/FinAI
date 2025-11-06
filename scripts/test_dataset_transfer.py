#!/usr/bin/env python3
"""
Test script to demonstrate transferring a dataset from datasets.csv to trained_datasets.csv
with a timestamp.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path to import manage_datasets
sys.path.append(str(Path(__file__).parent.parent))
from scripts.manage_datasets import mark_dataset_as_trained, list_datasets

def main():
    print("=== Testing Dataset Transfer ===\n")
    
    # List datasets before transfer
    print("Current dataset status:")
    list_datasets()
    
    # Example: Mark a dataset as trained
    dataset_name = input("\nEnter the name of the dataset to mark as trained: ")
    model_path = input("Enter the model path (or press Enter to skip): ") or None
    
    print(f"\nMarking {dataset_name} as trained...")
    success = mark_dataset_as_trained(dataset_name, model_path)
    
    if success:
        print(f"\nSuccess! {dataset_name} has been marked as trained.")
    else:
        print(f"\nFailed to mark {dataset_name} as trained. Please check the dataset name.")
    
    # Show updated status
    print("\nUpdated dataset status:")
    list_datasets()

if __name__ == "__main__":
    main()
