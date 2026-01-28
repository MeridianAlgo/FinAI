#!/usr/bin/env python3
"""
Test the GitHub Actions workflow locally
"""

import os
import time

from huggingface_hub import list_repo_files, snapshot_download

print("==========================================")
print("Step 1: Download model from HuggingFace")
print("==========================================")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("")

token = os.environ.get("HF_TOKEN")
if not token:
    # Try reading from .env file
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    except:
        pass

repo_id = "MeridianAlgo/FinAI-Lite"

print(f"Token available: {bool(token)}")
print(f"Token length: {len(token) if token else 0}")
print(f"Repo: {repo_id}")
print("")

try:
    # First, list files to see what's available
    print("Listing files in repository...")
    start = time.time()
    files = list_repo_files(repo_id=repo_id, token=token)
    elapsed = time.time() - start
    print(f"Found {len(files)} files in {elapsed:.1f}s:")
    for f in files[:10]:  # Show first 10
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    print("")

    # Download with progress
    print("Starting download...")
    print("This will download ~4GB, may take 5-10 minutes...")
    start_time = time.time()

    snapshot_download(
        repo_id=repo_id,
        local_dir="checkpoints/model",
        token=token,
        max_workers=8,
        resume_download=True,
        local_dir_use_symlinks=False,
    )

    elapsed = time.time() - start_time
    print("")
    print(
        f"[OK] Model downloaded successfully in {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)"
    )

    # Verify download
    import os

    model_files = os.listdir("checkpoints/model")
    print(f"Downloaded files: {len(model_files)}")
    total_size = 0
    for f in sorted(model_files):
        fpath = f"checkpoints/model/{f}"
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            print(f"  - {f}: {size / 1024 / 1024:.1f} MB")
    print(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")

except Exception as e:
    print("")
    print(f"[ERROR] Download failed: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback

    traceback.print_exc()
    print("")
    print("Will initialize new model during training")

print("")
print("Download step completed at", time.strftime("%Y-%m-%d %H:%M:%S"))
print("==========================================")
print("")

# Step 2: Check model directory
print("==========================================")
print("Step 2: Check Model Directory")
print("==========================================")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("")

if os.path.exists("checkpoints/model"):
    print("Model directory exists")
    print("Files in checkpoints/model:")
    for f in os.listdir("checkpoints/model")[:20]:
        fpath = os.path.join("checkpoints/model", f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1024 / 1024
            print(f"  {f}: {size:.1f} MB")
    print("")
else:
    print("No model directory found - will initialize new model")
    print("")

# Check dataset state
if os.path.exists("checkpoints/dataset_state.json"):
    print("Dataset state exists:")
    with open("checkpoints/dataset_state.json", "r") as f:
        print(f.read())
    print("")
else:
    print("No dataset state found - starting from beginning")
    print("")

print("==========================================")
print("")

# Step 3: Run training
print("==========================================")
print("Step 3: Run Training")
print("==========================================")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("")
print("Starting train.py...")
print("==========================================")
print("")

# Import and run training
import sys

sys.path.insert(0, ".")

# Run the training
exec(open("train.py").read())
