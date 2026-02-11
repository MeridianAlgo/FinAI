"""Seed the HuggingFace repository with a fresh MeridianFormer model."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

from meridian.model.configuration import MeridianConfig
from meridian.model.modeling import MeridianForCausalLM


def main():
    token = os.getenv("HF_TOKEN")
    repo_id = "MeridianAlgo/FinAI-Lite"

    if not token:
        print("✗ HF_TOKEN not set")
        return

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, private=False, exist_ok=True)
        print(f"✓ Repo {repo_id} ready")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Initialize model with full config
    config = MeridianConfig()
    model = MeridianForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Save locally
    save_path = "./checkpoint"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(save_path)

    # Upload
    api = HfApi()
    api.upload_folder(
        folder_path=save_path,
        repo_id=repo_id,
        path_in_repo="checkpoint",
        commit_message="Initial MeridianFormer seed (300M params)",
        token=token,
    )
    print(f"✓ Model seeded to {repo_id}")


if __name__ == "__main__":
    main()
