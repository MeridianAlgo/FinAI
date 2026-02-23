"""MeridianFormer Training Script.

Orchestrates:
 1. Model initialization / checkpoint resume
 2. Finance-focused data pipeline
 3. Continual training with EWC
 4. Checkpoint saving for HuggingFace upload
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

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
    state_files = [
        os.path.join(checkpoint_path, "dataset_state.json"),
        state_path,
        "dataset_state.json"
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
    print(f"\n[INFO] Using base model: {model_id}")

    # 3. Model initialization or resume
    model_loaded = False
    checkpoint_path = "./checkpoint"
    checkpoint_weights = os.path.join(checkpoint_path, "model.safetensors")
    trainer_state = os.path.join(checkpoint_path, "trainer_state.pt")

    if os.path.exists(checkpoint_weights):
        print(f"  Loading checkpoint from {checkpoint_path}...")
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path, 
                trust_remote_code=True,
                torch_dtype=torch.float32, # CPU-friendly
                low_cpu_mem_usage=True
            )
            print("  [OK] Checkpoint loaded - continuing training")
            model_loaded = True
        except Exception as e:
            print(f"  [FAIL] Checkpoint load failed: {e}")

    if not model_loaded:
        print(f"  Loading pre-trained model {model_id} from HuggingFace...")
        from transformers import AutoModelForCausalLM, AutoConfig
        
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(config, "hidden_act") and config.hidden_act == "swiglu":
            config.hidden_act = "silu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # 4. Tokenizer
    print(f"  Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
    )

    # 7. Create trainer & load state
    trainer = MeridianTrainer(model, dataloader, train_config)

    initial_global_step = 0
    if model_loaded:
        success = trainer.load_checkpoint(checkpoint_path)
        if success:
            initial_global_step = trainer.global_step
            print(f"  [OK] Trainer state restored (global step {initial_global_step})")

    # 8. Train!
    # 8. Continual Training Loop
    import time
    
    run_count = 1
    while True:
        print(f"\n{'='*20} STARTING TRAINING RUN #{run_count} {'='*20}")
        print(f"  Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Fresh dataloader with current state
        dataloader = create_dataloader(
            tokenizer,
            batch_size=batch_size,
            block_size=block_size,
            skip_items=processed_items,
        )
        trainer.dataloader = dataloader # Update trainer's dataloader

        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n  Training interrupted by user.")
            break
        except Exception as e:
            print(f"\n  ERROR during training: {e}")
            import traceback
            traceback.print_exc()

        # Save checkpoint (SKIPPING OPTIMIZER for fast tests)
        trained_checkpoint = "./checkpoint_trained"
        print(f"\n  Saving checkpoint to {trained_checkpoint}...")
        trainer.save_checkpoint(trained_checkpoint, skip_optimizer=True)

        if tokenizer:
            tokenizer.save_pretrained(trained_checkpoint)

        # Update dataset state
        if hasattr(trainer, "processed_batches"):
            batches_processed = trainer.processed_batches
        else:
            batches_processed = (
                trainer.global_step - initial_global_step
            ) * train_config.gradient_accumulation_steps

        items_processed = batches_processed * train_config.batch_size
        processed_items += items_processed # Update for next loop

        for sp in [state_path, os.path.join(checkpoint_path, "dataset_state.json")]:
            with open(sp, "w") as f:
                json.dump({"processed_items": processed_items}, f)

        print(f"  [OK] Dataset state saved (total processed: {processed_items:,})")
        
        # Update initial global step for next iteration calculation if needed
        initial_global_step = trainer.global_step

        print(f"\n  TRAINING RUN #{run_count} COMPLETE.")
        print(f"  Waiting 5 seconds before next run...")
        print(f"  Next run at roughly: {time.strftime('%H:%M:%S', time.localtime(time.time() + 5))}")
        
        run_count += 1
        time.sleep(5) # Fast wait for test


if __name__ == "__main__":
    main()
