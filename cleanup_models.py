#!/usr/bin/env python3
"""
Cleanup old model checkpoints - keeps only the main model
"""
import os
import shutil
from pathlib import Path

def cleanup_models():
    """Remove all model folders except the main finai_gpt.pt and tokenizer.pkl"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("No models directory found.")
        return
    
    # Files/folders to keep
    keep_files = {'finai_gpt.pt', 'tokenizer.pkl'}
    
    # Count what we're removing
    removed_count = 0
    removed_size = 0
    
    print("\nCleaning up models directory...")
    print("="*60)
    
    for item in models_dir.iterdir():
        if item.name not in keep_files:
            try:
                if item.is_file():
                    size = item.stat().st_size
                    item.unlink()
                    removed_size += size
                    removed_count += 1
                    print(f"  ✓ Removed file: {item.name}")
                elif item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    shutil.rmtree(item)
                    removed_size += size
                    removed_count += 1
                    print(f"  ✓ Removed folder: {item.name}/")
            except Exception as e:
                print(f"  ✗ Could not remove {item.name}: {e}")
    
    print("="*60)
    if removed_count > 0:
        print(f"✅ Cleaned up {removed_count} items ({removed_size / 1024 / 1024:.2f} MB freed)")
        print(f"\nKept files:")
        for f in keep_files:
            path = models_dir / f
            if path.exists():
                size = path.stat().st_size / 1024 / 1024
                print(f"  • {f} ({size:.2f} MB)")
    else:
        print("✅ No cleanup needed - directory is already clean")
    print()

if __name__ == "__main__":
    cleanup_models()
