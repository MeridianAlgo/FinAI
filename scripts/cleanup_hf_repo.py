#!/usr/bin/env python3
"""
Clean up legacy V2 files from Hugging Face repository
"""

import os

from huggingface_hub import HfApi


def cleanup_hf_repo():
    """Remove legacy V2 files from the Hugging Face repository"""

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ No HF_TOKEN found in environment")
        return

    repo_id = "MeridianAlgo/Fin.AI"
    api = HfApi(token=token)

    # Files to delete (V2 legacy files)
    files_to_delete = [
        "final_config.json",
        "model.pt",
        "model.safetensors",  # We'll use pytorch_model.bin or model.safetensors from save_pretrained
    ]

    print(f"🧹 Cleaning up legacy V2 files from {repo_id}...")

    for file_path in files_to_delete:
        try:
            print(f"  Deleting: {file_path}")
            api.delete_file(
                path_in_repo=file_path,
                repo_id=repo_id,
                token=token,
                commit_message=f"Remove legacy V2 file: {file_path}",
            )
            print(f"  ✅ Deleted: {file_path}")
        except Exception as e:
            print(f"  ⚠️  Could not delete {file_path}: {e}")

    print("\n✨ Cleanup complete!")
    print("\nRemaining files should be V3-compatible:")
    print("  - config.json (from FinAIConfig)")
    print("  - model.safetensors or pytorch_model.bin (from save_pretrained)")
    print("  - generation_config.json")
    print("  - README.md")
    print("  - version.json")


if __name__ == "__main__":
    cleanup_hf_repo()
