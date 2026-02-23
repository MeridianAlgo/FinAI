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
    repo_id = "MeridianAlgo/MeridianAI"
    base_model_id = "hpcai-tech/openmoe-base"

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

    # Initialize model from OpenMoE base
    print(f"  Fetching base model {base_model_id} architecture...")
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    if hasattr(config, "hidden_act") and config.hidden_act == "swiglu":
        config.hidden_act = "silu"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

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
        commit_message=f"Initial MeridianAI seed (Base: {base_model_id})",
        token=token,
    )
    print(f"[OK] Model seeded to {repo_id}")


if __name__ == "__main__":
    main()
