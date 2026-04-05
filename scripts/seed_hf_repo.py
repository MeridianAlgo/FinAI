"""Seed the HuggingFace repository with a fresh Meridian.AI model."""

import os
import sys

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    token = os.getenv("HF_TOKEN")
    repo_id = "MeridianAlgo/FinAI"
    base_model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer_id = os.getenv("TOKENIZER_ID", "Qwen/Qwen2.5-0.5B")

    print(f"\n{'=' * 60}")
    print("  MeridianAI Parameter Report")
    print(f"{'=' * 60}\n")

    if not token:
        print("[FAIL] HF_TOKEN not set")
        return

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, private=False, exist_ok=True)
        print(f"[OK] Repo {repo_id} ready")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Initialize model from SmolLM2-360M (standard Llama arch, pre-trained on 600B tokens)
    print(f"  Fetching base model {base_model_id}...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Save locally
    save_path = "./checkpoint"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(save_path)

    # Upload
    api = HfApi()
    api.upload_folder(
        folder_path=save_path,
        repo_id=repo_id,
        path_in_repo="checkpoint",
        commit_message=f"Initial MeridianAI seed (Base: {base_model_id}, Tokenizer: {tokenizer_id})",
        token=token,
    )
    print(f"[OK] Model seeded to {repo_id}")


if __name__ == "__main__":
    main()
