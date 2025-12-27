
import yaml
from datasets import load_dataset
import os
import sys

def test_datasets():
    print("Starting dataset verification...")
    with open("config/datasets.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    datasets = config.get("datasets", [])
    results = []
    
    for ds in datasets:
        name = ds["name"]
        subset = ds.get("subset")
        split = ds.get("split", "train")
        print(f"Testing {name} (subset: {subset}, split: {split})...")
        try:
            # For testing, we use streaming=True to avoid downloading the whole thing
            if subset:
                d = load_dataset(name, subset, split=split, trust_remote_code=True, streaming=True)
            else:
                d = load_dataset(name, split=split, trust_remote_code=True, streaming=True)
            
            # Try to get one item
            it = iter(d)
            sample = next(it)
            
            # Check text column
            text_col = ds.get("text_column", "text")
            if text_col in sample:
                print(f"  OK - Found column '{text_col}'")
                results.append((name, True, None))
            else:
                available = list(sample.keys())
                print(f"  FAIL - Column '{text_col}' not found. Available columns: {available}")
                results.append((name, False, f"Missing column {text_col}"))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append((name, False, str(e)))

    print("\n" + "="*50)
    print("DATASET VERIFICATION SUMMARY")
    print("="*50)
    all_ok = True
    for name, ok, error in results:
        status = "OK" if ok else "FAIL"
        err_msg = f" - {error}" if error else ""
        print(f"{status} | {name}{err_msg}")
        if not ok:
            all_ok = False
            
    if not all_ok:
        print("\nSome datasets failed to load!")
        sys.exit(1)
    else:
        print("\nAll datasets are working properly!")
        sys.exit(0)

if __name__ == "__main__":
    test_datasets()
