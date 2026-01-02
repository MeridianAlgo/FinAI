"""Rigorous dataset testing script - moved to legacy/ to avoid automatic pytest collection.

Run manually: python legacy/dataset_comprehensive.py
"""

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
import sys
import time
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def test_dataset_comprehensive(name, subset, split, text_column, max_samples=None):
    """Comprehensive test of dataset loading and processing."""
    try:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        if subset:
            print(f"Subset: {subset}")
        print(f"Split: {split}, Text column: {text_column}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Load dataset
        load_kwargs = {"split": split, "streaming": True}
        
        if subset:
            dataset = load_dataset(name, subset, **load_kwargs)
        else:
            dataset = load_dataset(name, **load_kwargs)
        
        # Get first 10 items to test thoroughly
        items = []
        for i, item in enumerate(dataset):
            if i >= 10:
                break
            items.append(item)
        
        if not items:
            print(f"❌ FAILED: No items loaded")
            return False
        
        print(f"✓ Loaded {len(items)} test items")
        
        # Test text extraction
        texts = []
        for item in items:
            text = extract_text(item, text_column)
            if text and len(text.strip()) > 10:
                texts.append(text)
        
        if not texts:
            print(f"❌ FAILED: No valid text extracted from column '{text_column}'")
            print(f"   Available keys: {list(items[0].keys())[:10]}")
            return False
        
        print(f"✓ Extracted {len(texts)} valid texts")
        print(f"   Sample text length: {len(texts[0])} chars")
        print(f"   Sample: {texts[0][:100]}...")
        
        # Test tokenization
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        try:
            tokens = tokenizer.encode(texts[0], add_special_tokens=False)
            print(f"✓ Tokenization works: {len(tokens)} tokens")
        except Exception as e:
            print(f"❌ FAILED: Tokenization error: {e}")
            return False
        
        elapsed = time.time() - start_time
        print(f"✅ SUCCESS: {name} ({elapsed:.2f}s)")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {name}")
        print(f"   Error: {str(e)[:200]}")
        return False


def extract_text(item, text_column):
    """Extract text from item, handling nested fields."""
    if '.' in text_column:
        parts = text_column.split('.')
        value = item
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, "")
            elif isinstance(value, list) and value:
                value = value[0]
                if isinstance(value, dict):
                    value = value.get(part, "")
            else:
                return ""
        
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)
        
        return str(value) if value else ""
    else:
        value = item.get(text_column, "")
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)
        return str(value) if value else ""


def main():
    print("This script has been moved to legacy/ to avoid downloads during pytest. Run manually if needed.")


if __name__ == "__main__":
    main()
