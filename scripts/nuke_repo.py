"""Nuke (delete) the HuggingFace repository for a clean restart."""

import os

from huggingface_hub import HfApi


def main():
    token = os.getenv("HF_TOKEN")
    repo_id = "MeridianAlgo/MeridianFormer"

    if not token:
        print("✗ HF_TOKEN not set")
        return

    api = HfApi()
    try:
        api.delete_repo(repo_id=repo_id, token=token)
        print(f"✓ Deleted {repo_id}")
    except Exception as e:
        print(f"Delete note: {e}")


if __name__ == "__main__":
    main()
