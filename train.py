"""MeridianFormer Training Script.

Orchestrates:
 1. Model initialization / checkpoint resume
 2. Finance-focused data pipeline
 3. Continual training with EWC
 4. Checkpoint saving for HuggingFace upload
"""

import json
import os

# Windows multiprocessing fix
if os.name == "nt":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

try:
    import comet_ml  # noqa: F401
except Exception:
    pass

import torch
from transformers import AutoTokenizer

from meridian.model.configuration import MeridianConfig
from meridian.model.modeling import MeridianForCausalLM
from meridian.data.pipeline import create_dataloader, create_smoke_dataloader
from meridian.training.trainer import MeridianTrainer, TrainingConfig


def main():
    print("=" * 70)
    print("  MeridianFormer v1.0 — Finance LLM Training")
    print("  Architecture: Sparse MoE + GQA + RoPE + SwiGLU + Numeracy Encoding")
    print("=" * 70)

    smoke_test = os.getenv("SMOKE_TEST", "0") == "1"
    checkpoint_path = "./checkpoint"
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
        print("\n✓ Smoke test passed!")
        return

    # ── Full Training Mode ───────────────────────────────────────────────
    print("\n[MODE] Full Training — Hourly Continual Learning\n")

    # 1. Load dataset state
    processed_items = 0
    for sp in [
        os.path.join(checkpoint_path, "dataset_state.json"),
        state_path,
    ]:
        if os.path.exists(sp):
            with open(sp, "r") as f:
                state = json.load(f)
                processed_items = state.get("processed_items", 0)
            print(f"  Resuming from dataset index: {processed_items}")
            break

    # 2. Configuration — 300M parameter architecture
    config = MeridianConfig(
        vocab_size=int(os.getenv("VOCAB_SIZE", "151665")),
        hidden_size=int(os.getenv("HIDDEN_SIZE", "768")),
        intermediate_size=int(os.getenv("INTERMEDIATE_SIZE", "1792")),
        num_layers=int(os.getenv("NUM_LAYERS", "14")),
        num_attention_heads=int(os.getenv("NUM_HEADS", "12")),
        num_key_value_heads=int(os.getenv("NUM_KV_HEADS", "4")),
        num_experts=int(os.getenv("NUM_EXPERTS", "8")),
        num_experts_per_token=int(os.getenv("EXPERTS_PER_TOKEN", "2")),
        expert_intermediate_size=int(os.getenv("EXPERT_INTER_SIZE", "896")),
        moe_layer_frequency=int(os.getenv("MOE_FREQ", "2")),
        max_position_embeddings=int(os.getenv("MAX_POS", "2048")),
        gradient_checkpointing=os.getenv("GRADIENT_CHECKPOINTING", "1") == "1",
        use_numeracy_encoding=True,
    )

    # 3. Model initialization or resume
    model_loaded = False
    checkpoint_config = os.path.join(checkpoint_path, "config.json")
    checkpoint_weights = os.path.join(checkpoint_path, "model.safetensors")
    trainer_state = os.path.join(checkpoint_path, "trainer_state.pt")

    if (
        os.path.exists(checkpoint_config)
        and os.path.exists(checkpoint_weights)
        and os.path.exists(trainer_state)
    ):
        print(f"  Loading checkpoint from {checkpoint_path}...")
        try:
            model = MeridianForCausalLM.from_pretrained(
                checkpoint_path, ignore_mismatched_sizes=False, low_cpu_mem_usage=False
            )
            print("  ✓ Checkpoint loaded — continuing training")
            model_loaded = True
        except Exception as e:
            print(f"  ✗ Checkpoint load failed: {e}")
            import traceback

            traceback.print_exc()

    if not model_loaded:
        print("  Initializing fresh model from scratch...")
        model = MeridianForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Config: {config}")

    # 4. Tokenizer
    max_retries = 3
    tokenizer = None
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Tokenizer retry {attempt + 1}: {e}")
                import time

                time.sleep(5)
            else:
                raise

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 5. Data pipeline
    block_size = int(os.getenv("BLOCK_SIZE", "512"))
    batch_size = int(os.getenv("BATCH_SIZE", "2"))
    dataloader = create_dataloader(
        tokenizer,
        batch_size=batch_size,
        block_size=block_size,
        skip_items=processed_items,
    )

    # 6. Training configuration
    max_steps = int(os.getenv("MAX_STEPS", "200"))
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
    )

    # 7. Create trainer & load state
    trainer = MeridianTrainer(model, dataloader, train_config)

    initial_global_step = 0
    if model_loaded:
        success = trainer.load_checkpoint(checkpoint_path)
        if success:
            initial_global_step = trainer.global_step
            print(f"  ✓ Trainer state restored (global step {initial_global_step})")

    # 8. Train!
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n  Training interrupted by user.")
    finally:
        # Save checkpoint
        print(f"\n  Saving checkpoint to {checkpoint_path}...")
        trainer.save_checkpoint(checkpoint_path)

        if tokenizer:
            tokenizer.save_pretrained(checkpoint_path)

        # Update dataset state
        batches_this_run = (
            (trainer.global_step - initial_global_step)
            * train_config.batch_size
            * train_config.gradient_accumulation_steps
        )
        new_processed = processed_items + batches_this_run

        for sp in [state_path, os.path.join(checkpoint_path, "dataset_state.json")]:
            with open(sp, "w") as f:
                json.dump({"processed_items": new_processed}, f)

        print(f"  ✓ Dataset state saved (total processed: {new_processed:,})")


if __name__ == "__main__":
    main()
