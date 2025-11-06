#!/usr/bin/env python3
"""
Project cleanup script - removes unnecessary files and organizes structure
"""
import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up unnecessary files from the project root"""
    
    root = Path(".")
    
    # Files to remove (if they exist)
    files_to_remove = [
        "train_sequential_v2.py",  # Old version
        "train_all_datasets.py",   # Deprecated
        "datasets_list.py",        # Old approach
        "run_prompt.py",           # Obsolete
    ]
    
    # Directories to archive (move to archive/)
    dirs_to_archive = []
    
    print("\n" + "="*60)
    print("FinAI Project Cleanup")
    print("="*60 + "\n")
    
    # Ensure archive directory exists
    archive_dir = root / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    removed_count = 0
    archived_count = 0
    
    # Remove unnecessary files
    print("Removing unnecessary files...")
    for filename in files_to_remove:
        filepath = root / filename
        if filepath.exists():
            try:
                filepath.unlink()
                print(f"  ✓ Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {filename}: {e}")
    
    # Archive old directories
    if dirs_to_archive:
        print("\nArchiving old directories...")
        for dirname in dirs_to_archive:
            dirpath = root / dirname
            if dirpath.exists() and dirpath.is_dir():
                try:
                    dest = archive_dir / dirname
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(dirpath), str(dest))
                    print(f"  ✓ Archived: {dirname} → archive/{dirname}")
                    archived_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to archive {dirname}: {e}")
    
    print("\n" + "="*60)
    if removed_count > 0 or archived_count > 0:
        print(f"✅ Cleanup complete!")
        print(f"   Files removed: {removed_count}")
        print(f"   Directories archived: {archived_count}")
    else:
        print("✅ Project already clean - no changes needed")
    print("="*60 + "\n")
    
    # Show project structure
    print("Current project structure:")
    print("  FinAI/")
    print("    ├── src/              (core source code)")
    print("    ├── models/           (finai_gpt.pt + tokenizer.pkl)")
    print("    ├── datasets/         (local datasets)")
    print("    ├── scripts/          (utility scripts)")
    print("    ├── tests/            (test suites)")
    print("    ├── archive/          (old files)")
    print("    ├── main.py           (CLI)")
    print("    ├── train_all.py      (main training script)")
    print("    ├── cleanup_models.py (model cleanup)")
    print("    ├── datasets.csv      (pending datasets)")
    print("    ├── trained_datasets.csv (completed datasets)")
    print("    ├── requirements.txt  (dependencies)")
    print("    └── README.md         (documentation)")
    print()

if __name__ == "__main__":
    cleanup_project()
