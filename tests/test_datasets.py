#!/usr/bin/env python3
"""
Test all datasets to ensure they load correctly

Usage:
    python test_datasets.py
"""

import yaml
from datasets import load_dataset

print("🧪 Testing All Datasets\n")

# Load config
with open("config/datasets.yaml", "r") as f:
    config = yaml.safe_load(f)

datasets = config.get("datasets", [])
print(f"Found {len(datasets)} datasets to test\n")

working_datasets = []
failed_datasets = []

for i, ds_config in enumerate(datasets, 1):
    name = ds_config["name"]
    subset = ds_config.get("subset")
    split = ds_config.get("split", "train")
    text_column = ds_config.get("text_column", "text")
    
    print(f"{i}. Testing: {name}")
    print(f"   Subset: {subset}")
    print(f"   Split: {split}")
    print(f"   Column: {text_column}")
    
    try:
        # Try to load dataset
        if subset:
            dataset = load_dataset(name, subset, split=split)
        else:
            dataset = load_dataset(name, split=split)
        
        # Check if text column exists
        if len(dataset) > 0:
            sample = dataset[0]
            if text_column in sample:
                text = sample[text_column]
                print(f"   ✅ SUCCESS - {len(dataset):,} samples")
                print(f"   Sample: {str(text)[:100]}...")
                working_datasets.append(ds_config)
            else:
                print(f"   ❌ FAILED - Column '{text_column}' not found")
                print(f"   Available columns: {list(sample.keys())}")
                failed_datasets.append((name, f"Column '{text_column}' not found"))
        else:
            print(f"   ❌ FAILED - Dataset is empty")
            failed_datasets.append((name, "Empty dataset"))
            
    except Exception as e:
        print(f"   ❌ FAILED - {str(e)[:100]}")
        failed_datasets.append((name, str(e)[:100]))
    
    print()

print("=" * 60)
print(f"✅ Working: {len(working_datasets)}/{len(datasets)}")
print(f"❌ Failed: {len(failed_datasets)}/{len(datasets)}")
print("=" * 60)

if failed_datasets:
    print("\nFailed datasets:")
    for name, error in failed_datasets:
        print(f"  - {name}: {error}")

print(f"\n✅ {len(working_datasets)} datasets are ready to use!")
