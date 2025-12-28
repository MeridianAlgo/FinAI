"""Test new datasets before adding to config"""

from datasets import load_dataset

# Test datasets to add
test_datasets = [
    # Math reasoning
    {"name": "openai/gsm8k", "subset": "main", "split": "train", "column": "question"},
    # Common sense reasoning
    {"name": "commonsense_qa", "subset": None, "split": "train", "column": "question"},
    # Code - Python
    {"name": "codeparrot/github-code", "subset": "Python", "split": "train", "column": "code"},
    # Instruction following
    {"name": "tatsu-lab/alpaca", "subset": None, "split": "train", "column": "text"},
    # Conversations
    {"name": "HuggingFaceH4/ultrachat_200k", "subset": None, "split": "train_sft", "column": "messages"},
    # Scientific
    {"name": "allenai/c4", "subset": "en", "split": "train", "column": "text"},
    # Books
    {"name": "pg19", "subset": None, "split": "train", "column": "text"},
]

print("🧪 Testing New Datasets\n")

working = []
failed = []

for i, ds in enumerate(test_datasets, 1):
    name = ds["name"]
    subset = ds["subset"]
    split = ds["split"]
    column = ds["column"]
    
    print(f"{i}. Testing: {name}")
    print(f"   Subset: {subset}")
    print(f"   Split: {split}")
    print(f"   Column: {column}")
    
    try:
        if subset:
            dataset = load_dataset(name, subset, split=split, streaming=True)
        else:
            dataset = load_dataset(name, split=split, streaming=True)
        
        # Try to get first item
        first_item = next(iter(dataset))
        text = first_item.get(column, "")
        
        if text and len(str(text)) > 10:
            print(f"   ✅ SUCCESS")
            print(f"   Sample: {str(text)[:100]}...\n")
            working.append(ds)
        else:
            print(f"   ❌ FAILED - Empty or invalid column\n")
            failed.append({"dataset": ds, "error": "Empty column"})
    except Exception as e:
        print(f"   ❌ FAILED - {str(e)[:100]}\n")
        failed.append({"dataset": ds, "error": str(e)})

print("=" * 60)
print(f"✅ Working: {len(working)}/{len(test_datasets)}")
print(f"❌ Failed: {len(failed)}/{len(test_datasets)}")
print("=" * 60)

if working:
    print("\n✅ Working datasets:")
    for ds in working:
        print(f"  - {ds['name']}")

if failed:
    print("\n❌ Failed datasets:")
    for item in failed:
        print(f"  - {item['dataset']['name']}: {item['error'][:50]}")
