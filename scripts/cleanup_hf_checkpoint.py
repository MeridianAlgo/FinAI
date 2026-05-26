"""Remove stale pytorch_model.bin from HuggingFace checkpoint.

The checkpoint currently has both model.safetensors (942 MB) and
pytorch_model.bin (942 MB) — a duplicate that wastes ~942 MB.
This script deletes the .bin file from the HF repo.

Usage:
    python scripts/cleanup_hf_checkpoint.py
"""

import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def main() -> None:
    token = os.getenv("huggingface_token") or os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID", "meridianal/FinAI")

    if not token:
        print("[FAIL] No HuggingFace token found. Set HF_TOKEN or huggingface_token in .env")
        return

    api = HfApi()

    # Check what files exist
    print(f"[INFO] Listing files in {repo_id}...")
    files = list(api.list_repo_files(repo_id=repo_id, token=token))
    checkpoint_files = [f for f in files if f.startswith("checkpoint/")]
    print(f"  Found {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"    {f}")

    # Delete stale pytorch_model.bin if both exist
    bin_path = "checkpoint/pytorch_model.bin"
    safetensors_path = "checkpoint/model.safetensors"

    if bin_path in files and safetensors_path in files:
        print(f"\n[INFO] Both model files exist. Deleting stale {bin_path}...")
        try:
            api.delete_file(
                path_in_repo=bin_path,
                repo_id=repo_id,
                token=token,
                commit_message="chore: remove stale pytorch_model.bin (model.safetensors is canonical)",
            )
            print(f"  [OK] Deleted {bin_path}")
        except Exception as e:
            print(f"  [FAIL] Could not delete {bin_path}: {e}")
    elif bin_path in files and safetensors_path not in files:
        print("  [INFO] Only .bin file exists — keeping as is (no safetensors to replace it).")
    elif safetensors_path in files and bin_path not in files:
        print("  [OK] Only model.safetensors exists — no cleanup needed.")
    else:
        print("  [WARN] Neither model file found in checkpoint/.")


if __name__ == "__main__":
    main()
