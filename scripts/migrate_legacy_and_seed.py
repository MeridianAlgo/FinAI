"""Migrate current HF checkpoint to legacy/v5.1.0/ and seed a fresh Qwen2.5-0.5B.

Steps
-----
1. List all files under checkpoint/ in meridianal/FinAI
2. Download each file and re-upload under legacy/v5.1.0/
3. Download Qwen/Qwen2.5-0.5B, save as safetensors, upload to checkpoint/
4. Tokenizer + model card also re-uploaded

Usage
-----
    python scripts/migrate_legacy_and_seed.py
"""

import os
import tempfile

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


def main() -> None:
    token = os.getenv("huggingface_token") or os.getenv("HF_TOKEN")
    repo_id = "meridianal/FinAI"
    base_model_id = "Qwen/Qwen2.5-0.5B"
    legacy_prefix = "legacy/v5.1.0"
    checkpoint_prefix = "checkpoint"

    if not token:
        print("[FAIL] No HuggingFace token. Set HF_TOKEN or huggingface_token in .env")
        return

    api = HfApi()

    # ── 1. Discover checkpoint files ────────────────────────────────────────
    print(f"\n[INFO] Listing files in {repo_id}...")
    all_files = list(api.list_repo_files(repo_id=repo_id, token=token))
    ckpt_files = [f for f in all_files if f.startswith(f"{checkpoint_prefix}/")]
    print(f"  Found {len(ckpt_files)} checkpoint files:")
    for f in ckpt_files:
        print(f"    {f}")

    if not ckpt_files:
        print("[WARN] No checkpoint files found — skipping legacy migration step.")
    else:
        # ── 2. Copy to legacy/ ───────────────────────────────────────────────
        print(f"\n[INFO] Copying checkpoint -> {legacy_prefix}/...")
        with tempfile.TemporaryDirectory() as tmpdir:
            for remote_path in ckpt_files:
                filename = remote_path[len(checkpoint_prefix) + 1 :]  # strip "checkpoint/"
                # Skip the stale pytorch_model.bin — no need in legacy either
                if filename == "pytorch_model.bin":
                    print(f"  [SKIP] {filename} (stale duplicate)")
                    continue
                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=remote_path,
                        token=token,
                        local_dir=tmpdir,
                    )
                    legacy_path = f"{legacy_prefix}/{filename}"
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=legacy_path,
                        repo_id=repo_id,
                        commit_message=f"chore: migrate checkpoint -> {legacy_prefix} (v5.1.0 archive)",
                        token=token,
                    )
                    print(f"  [OK] {filename} -> {legacy_path}")
                except Exception as e:
                    print(f"  [WARN] Could not copy {remote_path}: {e}")

        print(f"\n[OK] Legacy migration complete -> {repo_id}/{legacy_prefix}/")

    # ── 3. Seed fresh Qwen2.5-0.5B as new checkpoint/ ───────────────────────
    print(f"\n[INFO] Loading fresh {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "checkpoint")
        os.makedirs(save_path)

        print(f"  Saving model to {save_path}...")
        model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)

        # Patch tokenizer_config.json: extra_special_tokens list -> dict
        _tok_cfg_path = os.path.join(save_path, "tokenizer_config.json")
        if os.path.exists(_tok_cfg_path):
            import json

            with open(_tok_cfg_path) as _f:
                _tok_cfg = json.load(_f)
            if isinstance(_tok_cfg.get("extra_special_tokens"), list):
                _tok_cfg["extra_special_tokens"] = {}
                with open(_tok_cfg_path, "w") as _f:
                    json.dump(_tok_cfg, _f, indent=2)
                print("  [FIX] Patched tokenizer_config.json extra_special_tokens list -> dict")

        print(f"\n[INFO] Uploading fresh checkpoint to {repo_id}/{checkpoint_prefix}/...")
        api.upload_folder(
            folder_path=save_path,
            repo_id=repo_id,
            path_in_repo=checkpoint_prefix,
            commit_message=f"chore: seed fresh {base_model_id} for v6.0.0 training",
            token=token,
            delete_patterns=[
                "checkpoint/pytorch_model.bin",
                "checkpoint/trainer_state.pt",
                "checkpoint/ewc_state.pt",
                "checkpoint/dataset_state.json",
            ],
        )
        print(f"[OK] Fresh model uploaded to {repo_id}/{checkpoint_prefix}/")

    # ── 4. Upload model card (README.md) ────────────────────────────────────
    readme_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md"
    )
    if os.path.exists(readme_path):
        print("\n[INFO] Uploading model card...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="docs: update model card for v6.0.0",
            token=token,
        )
        print(f"[OK] Model card uploaded to {repo_id}/README.md")

    print(f"\n{'=' * 60}")
    print("  Migration complete.")
    print(f"  Legacy weights: {repo_id}/{legacy_prefix}/")
    print(f"  New checkpoint: {repo_id}/{checkpoint_prefix}/ (fresh {base_model_id})")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
