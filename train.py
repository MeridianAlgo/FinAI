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
from meridian.model import MeridianConfig, MeridianForCausalLM
from meridian.training.trainer import MeridianTrainer, TrainingConfig

load_dotenv()

# Windows multiprocessing fix
if os.name == "nt":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def main():
    print("=" * 70)
    print("  MeridianAI v1.0 — Finance LLM Training")
    print("  Architecture: Sparse MoE + GQA + RoPE + SwiGLU + Numeracy Encoding")
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
            use_ewc=False,
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

    # 2. Configuration — Using OpenMoE-Base (650M Sparse MoE)
    model_id = "hpcai-tech/openmoe-base"
    tokenizer_id = os.getenv("TOKENIZER_ID", "google/umt5-small")
    print(f"\n[INFO] Using base model: {model_id}")
    print(f"[INFO] Using tokenizer: {tokenizer_id}")

    # 3. Model initialization or resume
    model_loaded = False
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "./checkpoint")
    checkpoint_weights = os.path.join(checkpoint_path, "model.safetensors")

    requested_dtype = os.getenv("DTYPE", "bfloat16").lower()
    use_bf16 = requested_dtype in {"bf16", "bfloat16"}

    if os.path.exists(checkpoint_weights):
        print(f"  [DEBUG] Found model weights at {checkpoint_weights}. Loading...")
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                low_cpu_mem_usage=True,
            )
            print("  [OK] Checkpoint loaded - continuing training")
            model_loaded = True
        except Exception as e:
            if use_bf16:
                print(f"  [WARN] bf16 resume load failed ({e}). Falling back to float32.")
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    trust_remote_code=True,
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
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(config, "hidden_act") and config.hidden_act == "swiglu":
            config.hidden_act = "silu"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
        except Exception as e:
            if use_bf16:
                print(f"  [WARN] bf16 load failed ({e}). Falling back to float32.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    ignore_mismatched_sizes=True,
                )
            else:
                raise

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

    # 4. Tokenizer
    print(f"  Loading tokenizer from {tokenizer_id}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
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
        learning_rate=float(os.getenv("LEARNING_RATE", "3e-4")),
        output_dir=checkpoint_path,
        save_steps=int(os.getenv("SAVE_STEPS", "50")),
        use_ewc=os.getenv("USE_EWC", "1") == "1",
        ewc_lambda=float(os.getenv("EWC_LAMBDA", "100.0")),
        ewc_samples=int(os.getenv("EWC_SAMPLES", "20")),
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

        # Update dataset state
        if hasattr(trainer, "processed_batches"):
            batches_processed = trainer.processed_batches
        else:
            batches_processed = (
                trainer.global_step - initial_global_step
            ) * train_config.gradient_accumulation_steps

        items_processed = batches_processed * train_config.batch_size
        processed_items += items_processed  # Update for next loop

        for sp in [state_path, os.path.join(checkpoint_path, "dataset_state.json")]:
            with open(sp, "w") as f:
                json.dump({"processed_items": processed_items}, f)

        print(f"  [OK] Dataset state saved (total processed: {processed_items:,})")

        # Update initial global step for next iteration calculation if needed
        initial_global_step = trainer.global_step

        print(f"\n  TRAINING RUN #{run_count} COMPLETE.")


if __name__ == "__main__":
    main()
