"""
Lightweight test suite for FinAI model validation
Runs before training to ensure basic system integrity
"""

import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("FinAI Pre-Training Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Check Python packages
    print("\n[1/5] Testing Python packages...")
    try:
        import torch
        import transformers
        import datasets
        print("  ✓ All required packages installed")
    except ImportError as e:
        print(f"  ✗ Missing package: {e}")
        all_passed = False
    
    # Test 2: Check project structure
    print("\n[2/5] Testing project structure...")
    base_path = Path(__file__).parent.parent
    required_paths = {
        "src": base_path / "src",
        "models": base_path / "models",
        "scripts": base_path / "scripts",
    }
    
    for name, path in required_paths.items():
        if path.exists():
            print(f"  ✓ {name}/ exists")
        else:
            print(f"  ✗ {name}/ missing")
            all_passed = False
    
    # Test 3: Check model files
    print("\n[3/5] Testing model files...")
    model_file = base_path / "models" / "finai_gpt.pt"
    tokenizer_file = base_path / "models" / "tokenizer.pkl"
    
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Model file exists ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ Model file missing")
        all_passed = False
    
    if tokenizer_file.exists():
        print(f"  ✓ Tokenizer file exists")
    else:
        print(f"  ✗ Tokenizer file missing")
        all_passed = False
    
    # Test 4: Check training script
    print("\n[4/5] Testing training script...")
    train_script = base_path / "scripts" / "train_daily_gh.py"
    if train_script.exists():
        print(f"  ✓ Training script exists")
    else:
        print(f"  ✗ Training script missing")
        all_passed = False
    
    # Test 5: Check configuration
    print("\n[5/5] Testing configuration...")
    try:
        sys.path.insert(0, str(base_path))
        from src.config import Config
        config = Config()
        print(f"  ✓ Config loaded (embd={config.N_EMBD}, layers={config.N_LAYER})")
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for training")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME TESTS FAILED - Please fix before training")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
