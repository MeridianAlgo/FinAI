"""Meridian.AI Training Script.

Orchestrates:
 1. Model initialization / checkpoint resume
 2. Finance-focused data pipeline
 3. Continual training with EWC
 4. Checkpoint saving for HuggingFace upload
"""

import json
import multiprocessing
import os
import signal
import time
import traceback

try:
    import comet_ml  # noqa: F401
except Exception:
    pass

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from meridian.data.pipeline import create_dataloader, create_smoke_dataloader
from meridian.model import (
    MeridianSMoEConfig as MeridianConfig,
)
from meridian.model import (
    MeridianSMoEForCausalLM as MeridianForCausalLM,
)
from meridian.training.trainer import MeridianTrainer, TrainingConfig

load_dotenv()

# Windows multiprocessing fix
if os.name == "nt":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def sigterm_handler(signum, frame):
    print("\n[CRITICAL] Received SIGTERM! Triggering graceful shutdown...")
    raise KeyboardInterrupt()


signal.signal(signal.SIGTERM, sigterm_handler)


def hf_load_with_retry(build, label: str, max_attempts: int = 5):
    """Run an HF ``from_pretrained`` loader with backoff, then fall back to cache.

    HuggingFace frequently returns HTTP 429 (rate limit) to shared GitHub Actions
    IPs. ``build(offline)`` constructs the object; we retry the online path with
    exponential backoff, and as a last resort load from the local HF cache
    (``local_files_only=True``) — which succeeds whenever a previous run cached
    the base model/tokenizer.
    """
    delay = 4
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return build(False)
        except Exception as e:  # noqa: BLE001 — we genuinely want to retry on anything
            last_err = e
            msg = str(e)
            rate = "429" in msg or "Too Many Requests" in msg or "couldn't connect" in msg
            kind = "rate-limit/connection" if rate else "error"
            print(f"  [WARN] {label} load attempt {attempt}/{max_attempts} failed ({kind}).")
            if attempt < max_attempts:
                print(f"  [INFO] Retrying {label} in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
    try:
        print(f"  [INFO] {label}: Hub unreachable — trying local HF cache (offline)...")
        return build(True)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: local cache fallback failed ({e}).")
        raise last_err


def main():
    print("=" * 70)
    print("  MeridianAI v1.0.1 (Production) — Finance LLM Training (Qwen2.5-0.5B base)")
    print("  Fine-tuning: Qwen2.5-0.5B via AdaFactor + EWC continual learning")
    print("=" * 70)

    # FAST_MODE is for quick local debugging on CPU (keeps the real model, but avoids heavy pipelines)
    if os.getenv("FAST_MODE", "0") == "1":
        os.environ.setdefault("USE_LIGHT_DATASETS", "1")
        os.environ.setdefault("MAX_STEPS", "5")
        os.environ.setdefault("BATCH_SIZE", "1")
        os.environ.setdefault("GRAD_ACCUM", "1")
        os.environ.setdefault("BLOCK_SIZE", "32")
        os.environ.setdefault("MAX_BYTES", str(2 * 1024 * 1024))
        os.environ.setdefault("USE_EWC", "0")
        os.environ.setdefault("EWC_SAMPLES", "0")
        os.environ.setdefault("FREE_OPTIMIZER_BEFORE_FISHER", "1")
        os.environ.setdefault("SKIP_FISHER", "1")
        os.environ.setdefault("DEBUG_STEPS", "0")

    smoke_test = os.getenv("SMOKE_TEST", "0") == "1"
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "./checkpoint")
    state_path = "dataset_state.json"

    # ── Smoke Test Mode ──────────────────────────────────────────────────
    if smoke_test:
        print("\n[MODE] Smoke Test — verifying architecture works\n")
        config = MeridianConfig(
            vocab_size=4096,
            hidden_size=128,
            intermediate_size=352,
            num_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_token=2,
            expert_intermediate_size=176,
            moe_layer_frequency=2,
            max_position_embeddings=256,
            gradient_checkpointing=False,
            use_numeracy_encoding=True,
            numeracy_embed_dim=32,
        )
        model = MeridianForCausalLM(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Smoke model params: {total_params:,}")

        dl = create_smoke_dataloader(
            vocab_size=config.vocab_size,
            batch_size=int(os.getenv("BATCH_SIZE", "2")),
            block_size=int(os.getenv("BLOCK_SIZE", "64")),
        )

        train_config = TrainingConfig(
            batch_size=int(os.getenv("BATCH_SIZE", "2")),
            gradient_accumulation_steps=1,
            max_steps=int(os.getenv("MAX_STEPS", "10")),
            total_steps=int(os.getenv("MAX_STEPS", "10")),
            learning_rate=5e-4,
            output_dir=checkpoint_path,
            save_steps=int(os.getenv("MAX_STEPS", "10")),
            use_ewc=False,
        )

        trainer = MeridianTrainer(model, dl, train_config)
        trainer.train()
        print("\n[OK] Smoke test passed!")
        return

    # ── Full Training Mode ───────────────────────────────────────────────
    print("\n[MODE] Full Training — Hourly Continual Learning\n")

    # 1. Load dataset state
    processed_items = 0
    state_files = [
        os.path.join(checkpoint_path, "dataset_state.json"),
        state_path,
        "dataset_state.json",
    ]

    found_items = []
    for sp in state_files:
        if os.path.exists(sp):
            try:
                with open(sp, "r") as f:
                    state = json.load(f)
                    val = state.get("processed_items", 0)
                    found_items.append(val)
                    print(f"  Found state in {sp}: {val}")
            except Exception:
                pass

    if found_items:
        processed_items = max(found_items)
        print(f"  Resuming from maximum dataset index: {processed_items}")
    else:
        print("  No dataset state found, starting from 0.")

    # 2. Configuration — Upgrading to Qwen2.5-0.5B for better capacity and reasoning
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer_id = os.getenv("TOKENIZER_ID", "Qwen/Qwen2.5-0.5B")
    print(f"\n[INFO] Using base model: {model_id}")
    print(f"[INFO] Using tokenizer: {tokenizer_id}")

    # 3. Model initialization or resume
    model_loaded = False
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "./checkpoint")
    checkpoint_weights = os.path.join(checkpoint_path, "model.safetensors")
    # Also accept pytorch_model.bin (older format)
    if not os.path.exists(checkpoint_weights):
        bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            checkpoint_weights = bin_path

    requested_dtype = os.getenv("DTYPE", "bfloat16").lower()
    use_bf16 = requested_dtype in {"bf16", "bfloat16"}

    if os.path.exists(checkpoint_weights):
        print(f"  [DEBUG] Found model weights at {checkpoint_weights}. Checking architecture...")
        # Verify checkpoint architecture matches expected model before loading
        arch_ok = False
        try:
            ckpt_cfg_path = os.path.join(checkpoint_path, "config.json")
            if os.path.exists(ckpt_cfg_path):
                with open(ckpt_cfg_path) as _f:
                    ckpt_cfg = json.load(_f)
                arch_ok = ckpt_cfg.get("model_type") in ["llama", "qwen2"]
        except Exception:
            pass

        if not arch_ok:
            print(
                "  [WARN] Checkpoint architecture mismatch (old model). "
                f"Discarding checkpoint and loading {model_id} fresh."
            )
        else:
            try:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                    low_cpu_mem_usage=True,
                )
                print("  [OK] Checkpoint loaded - continuing training")
                model_loaded = True
            except Exception as e:
                if use_bf16:
                    print(f"  [WARN] bf16 resume load failed ({e}). Falling back to float32.")
                    from transformers import AutoModelForCausalLM

                    model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_path,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                    )
                    print("  [OK] Checkpoint loaded (float32 fallback) - continuing training")
                    model_loaded = True
                else:
                    print(f"  [FAIL] Checkpoint load failed: {e}")
    else:
        print(f"  [DEBUG] No checkpoint weights found at {checkpoint_weights}.")

    if not model_loaded:
        print(f"  Loading pre-trained model {model_id} from HuggingFace...")
        from transformers import AutoModelForCausalLM

        hf_token = os.getenv("HF_TOKEN")

        def _build_model(offline: bool):
            common = dict(low_cpu_mem_usage=True, local_files_only=offline)
            if hf_token:
                common["token"] = hf_token
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                    **common,
                )
            except Exception as e:
                if use_bf16 and "429" not in str(e) and "Too Many Requests" not in str(e):
                    print(f"  [WARN] bf16 load failed ({e}). Falling back to float32.")
                    return AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float32, **common
                    )
                raise

        model = hf_load_with_retry(_build_model, label=f"base model {model_id}")

    if os.getenv("GRADIENT_CHECKPOINTING", "1") == "1":
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            print("  [OK] Gradient Checkpointing enabled")
        except Exception as e:
            print(f"  [WARN] Failed to enable gradient checkpointing: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # 4. Tokenizer — prefer the copy saved in the checkpoint (no Hub call); else Qwen Hub.
    hf_token = os.getenv("HF_TOKEN")
    local_tok = os.path.join(checkpoint_path, "tokenizer_config.json")
    if os.path.exists(local_tok):
        tok_source = checkpoint_path
        print(f"  Loading tokenizer from local checkpoint {checkpoint_path}...")
    else:
        tok_source = tokenizer_id
        print(f"  Loading tokenizer from {tokenizer_id}...")

    def _build_tokenizer(offline: bool):
        kwargs = {"local_files_only": offline}
        if hf_token:
            kwargs["token"] = hf_token
        return AutoTokenizer.from_pretrained(tok_source, **kwargs)

    tokenizer = hf_load_with_retry(_build_tokenizer, label=f"tokenizer {tok_source}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 5. Data pipeline
    block_size = int(os.getenv("BLOCK_SIZE", "64"))
    batch_size = int(os.getenv("BATCH_SIZE", "2"))
    max_bytes = int(os.getenv("MAX_BYTES", str(15 * 1024 * 1024)))
    dataloader = create_dataloader(
        tokenizer,
        batch_size=batch_size,
        block_size=block_size,
        skip_items=processed_items,
        max_bytes=max_bytes,
    )

    # 6. Training configuration
    max_steps = int(os.getenv("MAX_STEPS", "50"))
    total_steps = int(os.getenv("TOTAL_STEPS", "100000"))
    train_config = TrainingConfig(
        batch_size=batch_size,
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "4")),
        max_steps=max_steps,
        total_steps=total_steps,
        learning_rate=float(os.getenv("LEARNING_RATE", "5e-5")),
        output_dir=checkpoint_path,
        save_steps=int(os.getenv("SAVE_STEPS", "50")),
        use_ewc=os.getenv("USE_EWC", "1") == "1",
        ewc_lambda=float(os.getenv("EWC_LAMBDA", "500.0")),
        ewc_samples=int(os.getenv("EWC_SAMPLES", "50")),
    )

    # 7. Create trainer & load state
    print("  [DEBUG] Initializing MeridianTrainer...")
    trainer = MeridianTrainer(model, dataloader, train_config)

    initial_global_step = 0
    if model_loaded:
        print(f"  [DEBUG] Restoration: Attempting to load trainer state from {checkpoint_path}...")
        success = trainer.load_checkpoint(checkpoint_path)
        if success:
            initial_global_step = trainer.global_step
            print(f"  [OK] Trainer state restored (global step {initial_global_step})")
        else:
            print(f"  [DEBUG] Restoration: No trainer state found in {checkpoint_path}.")

    # 8. Train!
    # 8. Single Training Run
    for run_count in range(1, 2):
        print(f"\n{'='*20} STARTING TRAINING RUN #{run_count} {'='*20}")
        print(f"  Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Fresh dataloader with current state
        dataloader = create_dataloader(
            tokenizer,
            batch_size=batch_size,
            block_size=block_size,
            skip_items=processed_items,
            max_bytes=max_bytes,
        )
        trainer.dataloader = dataloader  # Update trainer's dataloader

        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n  Training interrupted by user.")
            break
        except Exception as e:
            print(f"\n  ERROR during training: {e}")
            traceback.print_exc()

        # Save checkpoint (SKIPPING OPTIMIZER for fast tests)
        print(f"\n  Saving checkpoint to {checkpoint_path}...")
        trainer.save_checkpoint(checkpoint_path, skip_optimizer=True)

        if tokenizer:
            tokenizer.save_pretrained(checkpoint_path)
            # Fix: transformers saves extra_special_tokens as a list on Qwen2 tokenizers,
            # but the spec requires a dict. Patch the saved file to prevent load errors.
            _tok_cfg_path = os.path.join(checkpoint_path, "tokenizer_config.json")
            if os.path.exists(_tok_cfg_path):
                import json as _json

                with open(_tok_cfg_path) as _f:
                    _tok_cfg = _json.load(_f)
                if isinstance(_tok_cfg.get("extra_special_tokens"), list):
                    _tok_cfg["extra_special_tokens"] = {}
                    with open(_tok_cfg_path, "w") as _f:
                        _json.dump(_tok_cfg, _f, indent=2)
                    print("  [FIX] Patched tokenizer_config.json extra_special_tokens list -> dict")

        # Update dataset state
        if hasattr(trainer, "processed_batches"):
            batches_processed = trainer.processed_batches
        else:
            batches_processed = (
                trainer.global_step - initial_global_step
            ) * train_config.gradient_accumulation_steps

        items_processed = batches_processed * train_config.batch_size
        processed_items += items_processed  # Update for next loop

        print(f"\n  [INFO] Successfully processed {items_processed:,} data items in this run.")
        print(
            f"  [INFO] Advancing global dataset index to {processed_items:,} for the next training session."
        )

        for sp in [state_path, os.path.join(checkpoint_path, "dataset_state.json")]:
            with open(sp, "w") as f:
                json.dump({"processed_items": processed_items}, f)

        print(f"  [OK] Dataset state saved (total processed: {processed_items:,})")

        # Update initial global step for next iteration calculation if needed
        initial_global_step = trainer.global_step

        print(f"\n  TRAINING RUN #{run_count} COMPLETE.")


if __name__ == "__main__":
    main()
