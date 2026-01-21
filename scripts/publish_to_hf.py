"""Publish a model directory to Hugging Face Hub.

Usage: set the env var `HF_TOKEN` to a valid token with repo write permissions, then run:
    python scripts/publish_to_hf.py --model-dir checkpoints/model --repo-id MeridianAlgo/Fin.AI --private

This script uses `huggingface_hub` and will raise if token is missing.
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="checkpoints/model")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    if k.strip() == "HF_TOKEN":
                        token = v.strip().strip('"').strip("'")
                        break
        except Exception:
            token = token
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set in env. Provide a token with write access."
        )

    HfApi()
    try:
        create_repo(repo_id=args.repo_id, token=token, private=args.private)
    except Exception:
        # repo may already exist
        pass

    print(f"Uploading {args.model_dir} to {args.repo_id} (private={args.private})")
    upload_folder(
        folder_path=args.model_dir, repo_id=args.repo_id, token=token, path_in_repo="/"
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
