"""FinAI-Next Training Script (Liquid-BitNet)"""

import json
import os

# Windows multiprocessing fix - must be before torch import
if os.name == 'nt':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

try:
    import comet_ml  # noqa: F401
except Exception:
    comet_ml = None

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from fin_ai.model.configuration_next import FinAINextConfig
from fin_ai.model.modeling_next import FinAINextForCausalLM
from fin_ai.training.next_trainer import NextTrainingConfig, TernaryTrainer


class CustomIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataloader_gen):
        self.dataloader_gen = dataloader_gen

    def __iter__(self):
        return self.dataloader_gen()


def create_dataloader(
    tokenizer,
    batch_size=4,
    block_size=1024,
    skip_items=0,
    max_bytes_per_slice=100 * 1024 * 1024,
):
    print(
        f"Initializing Sliced DataLoader (skipping {skip_items} items, max_bytes={max_bytes_per_slice}...)"
    )

    # Use only FineWeb-Edu as requested
    dataset_name = "HuggingFaceFW/fineweb-edu"
    print(f"  - Loading {dataset_name}...")

    import time

    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset = load_dataset(
                dataset_name, "default", split="train", streaming=True
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"Error loading dataset (attempt {attempt + 1}): {e}. Retrying in 10s..."
                )
                time.sleep(10)
            else:
                raise e

    if skip_items > 0:
        dataset = dataset.skip(skip_items)

    def gen():
        total_bytes_yielded = 0
        local_processed = 0

        try:
            for item in dataset:
                text = item.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    continue

                # Check slice limit
                text_bytes = len(text.encode("utf-8"))
                if total_bytes_yielded + text_bytes > max_bytes_per_slice:
                    print(
                        f"[INFO] 30MB slice limit reached ({total_bytes_yielded / 1024 / 1024:.2f} MB). Ending epoch."
                    )
                    return

                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=block_size,
                    padding="max_length",
                )

                input_ids = torch.tensor(tokens["input_ids"])
                labels = input_ids.clone()

                # Mask padding
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is not None:
                    labels[input_ids == pad_token_id] = -100

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "processed_idx": skip_items + local_processed,
                }

                total_bytes_yielded += text_bytes
                local_processed += 1
        except OSError as e:
            if getattr(e, "winerror", None) == 995:
                print("[INFO] Windows cancelled the operation (WinError 995). Ending data stream.")
                return
            raise

    return torch.utils.data.DataLoader(
        CustomIterableDataset(gen), batch_size=batch_size
    )


def create_smoke_dataloader(vocab_size: int, batch_size: int, block_size: int):
    def gen():
        for _ in range(1000):
            input_ids = torch.randint(0, vocab_size, (block_size,), dtype=torch.long)
            labels = input_ids.clone()
            yield {"input_ids": input_ids, "labels": labels}

    return torch.utils.data.DataLoader(CustomIterableDataset(gen), batch_size=batch_size)


def main():
    print("Initializing FinAI-Next (Liquid-BitNet) Overhaul...")

    smoke_test = os.getenv("SMOKE_TEST", "0") == "1"

    # Path settings
    model_path = "./model"
    checkpoint_path = "./checkpoint"
    state_path = "dataset_state.json"

    if smoke_test:
        config = FinAINextConfig(
            vocab_size=4096,
            hidden_size=128,
            num_layers=2,
            liquid_state_dim=64,
            gradient_checkpointing=False,
            tie_word_embeddings=False,
            use_vision_projector=False,
            use_audio_projector=False,
        )
        model = FinAINextForCausalLM(config)

        dataloader = create_smoke_dataloader(
            vocab_size=config.vocab_size,
            batch_size=int(os.getenv("BATCH_SIZE", "2")),
            block_size=int(os.getenv("BLOCK_SIZE", "64")),
        )

        max_steps = int(os.getenv("MAX_STEPS", "10"))
        total_steps = int(os.getenv("TOTAL_STEPS", str(max_steps)))
        train_config = NextTrainingConfig(
            batch_size=int(os.getenv("BATCH_SIZE", "2")),
            gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "1")),
            max_steps=max_steps,
            total_steps=total_steps,
            learning_rate=5e-4,
            output_dir=checkpoint_path,
            save_steps=max_steps,
        )

        trainer = TernaryTrainer(model, dataloader, train_config)
        trainer.train()
        return

    # 1. Load Dataset State
    processed_items = 0
    checkpoint_state_path = os.path.join(checkpoint_path, "dataset_state.json")
    model_state_path = os.path.join(model_path, "dataset_state.json")

    # Priority: checkpoint > model > root
    if os.path.exists(checkpoint_state_path):
        with open(checkpoint_state_path, "r") as f:
            state = json.load(f)
            processed_items = state.get("processed_items", 0)
        print(f"Resuming from checkpoint dataset index: {processed_items}")
    elif os.path.exists(model_state_path):
        with open(model_state_path, "r") as f:
            state = json.load(f)
            processed_items = state.get("processed_items", 0)
        print(f"Resuming from model dataset index: {processed_items}")
    elif os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
            processed_items = state.get("processed_items", 0)
        print(f"Resuming from root dataset index: {processed_items}")

    # 2. Configuration
    config = FinAINextConfig(
        vocab_size=int(os.getenv("VOCAB_SIZE", "151665")),
        hidden_size=int(os.getenv("HIDDEN_SIZE", "512")),
        num_layers=int(os.getenv("NUM_LAYERS", "8")),
        liquid_state_dim=int(os.getenv("LIQUID_STATE_DIM", "128")),
        gradient_checkpointing=os.getenv("GRADIENT_CHECKPOINTING", "1") == "1",
        tie_word_embeddings=False,  # Checkpoint has separate weights
        use_vision_projector=os.getenv("USE_VISION", "0") == "1",
        use_audio_projector=os.getenv("USE_AUDIO", "0") == "1",
    )
    print(f"Configuration: {config}")

    # 3. Model Initialization or Loading
    # Priority: checkpoint > model > fresh init
    checkpoint_exists = os.path.exists(os.path.join(checkpoint_path, "config.json"))
    checkpoint_weights_exist = os.path.exists(
        os.path.join(checkpoint_path, "model.safetensors")
    ) or os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin"))
    # Also check for trainer state to ensure we have a valid training checkpoint
    trainer_state_exists = os.path.exists(os.path.join(checkpoint_path, "trainer_state.pt"))

    model_exists = os.path.exists(os.path.join(model_path, "config.json"))
    weights_exist = os.path.exists(
        os.path.join(model_path, "model.safetensors")
    ) or os.path.exists(os.path.join(model_path, "pytorch_model.bin"))

    model_loaded = False

    # Try checkpoint first (requires both weights AND trainer state)
    if checkpoint_exists and checkpoint_weights_exist and trainer_state_exists:
        print(f"\n{'='*60}")
        print(f"ATTEMPTING TO LOAD CHECKPOINT FROM: {checkpoint_path}")
        print(f"{'='*60}")
        print(f"  config.json exists: {checkpoint_exists}")
        print(f"  model weights exist: {checkpoint_weights_exist}")

        # List files in checkpoint
        if os.path.exists(checkpoint_path):
            files = os.listdir(checkpoint_path)
            print(f"  Files in checkpoint: {files[:10]}")

        try:
            # Load without passing config to use the checkpoint's config
            model = FinAINextForCausalLM.from_pretrained(
                checkpoint_path,
                ignore_mismatched_sizes=False,
                low_cpu_mem_usage=False,
            )
            print("✓ Checkpoint model loaded successfully - CONTINUING TRAINING")

            # Verify weights loaded
            with torch.no_grad():
                weight_sample = model.model.embed_tokens.weight[0][:5].tolist()
                weight_mean = model.model.embed_tokens.weight.mean().item()
                weight_std = model.model.embed_tokens.weight.std().item()
                print(f"  Loaded weight sample: {[f'{x:.4f}' for x in weight_sample]}")
                print(f"  Loaded weight mean: {weight_mean:.6f}, std: {weight_std:.6f}")

            model_loaded = True
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            import traceback

            traceback.print_exc()
            print("Will try base model or initialize fresh...")
            print(f"{'='*60}\n")

    # Try base model if checkpoint failed
    if not model_loaded and model_exists and weights_exist:
        print(f"\n{'='*60}")
        print(f"ATTEMPTING TO LOAD BASE MODEL FROM: {model_path}")
        print(f"{'='*60}")
        try:
            model = FinAINextForCausalLM.from_pretrained(
                model_path,
                ignore_mismatched_sizes=False,
                low_cpu_mem_usage=False,
            )
            print("✓ Base model loaded successfully - CONTINUING TRAINING")
            model_loaded = True
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"✗ Error loading base model: {e}")
            import traceback

            traceback.print_exc()
            print("Will initialize fresh model...")
            print(f"{'='*60}\n")

    # Initialize fresh if nothing loaded
    if not model_loaded:
        print("⚠ No existing model found. Initializing new model from scratch.")
        print("⚠ This should only happen on the FIRST training run!")
        model = FinAINextForCausalLM(config)

    # Debug: Print initial weight sample to verify model state
    with torch.no_grad():
        weight_sample = model.model.embed_tokens.weight[0][:5].tolist()
        weight_mean = model.model.embed_tokens.weight.mean().item()
        weight_std = model.model.embed_tokens.weight.std().item()
        print(f"\n{'='*60}")
        print("INITIAL MODEL STATE")
        print(f"{'='*60}")
        print(f"Weight sample: {[f'{x:.4f}' for x in weight_sample]}")
        print(f"Weight mean: {weight_mean:.6f}, std: {weight_std:.6f}")
        print(f"{'='*60}\n")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Tokenizer
    import time

    max_retries = 3
    tokenizer = None
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"Error loading tokenizer (attempt {attempt + 1}): {e}. Retrying in 10s..."
                )
                time.sleep(10)
            else:
                raise e

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 5. Dataset with Skip (Rotational)
    dataloader = create_dataloader(
        tokenizer,
        batch_size=2,
        block_size=128,
        skip_items=processed_items,
    )

    # 6. Training Config
    max_steps = int(os.getenv("MAX_STEPS", "200"))
    total_steps = int(os.getenv("TOTAL_STEPS", "100000"))
    train_config = NextTrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        total_steps=total_steps,
        learning_rate=5e-5,
        output_dir=checkpoint_path,
        save_steps=50,  # Save more frequently to ensure checkpoints are created
    )

    # 7. Training
    trainer = TernaryTrainer(model, dataloader, train_config)

    initial_global_step = 0
    # Load trainer state - requires trainer_state.pt for valid checkpoint
    if checkpoint_exists and checkpoint_weights_exist and trainer_state_exists:
        print(f"\n{'='*60}")
        print("LOADING TRAINER STATE FROM CHECKPOINT")
        print(f"{'='*60}")
        success = trainer.load_checkpoint(checkpoint_path)
        if success:
            initial_global_step = trainer.global_step
            print("✓ Loaded trainer state from checkpoint")
            print(f"  Global step: {initial_global_step}")
            print(f"  Run step: {trainer.run_step}")
            print(f"  Will continue training from step {initial_global_step + 1}")
        else:
            print("✗ Failed to load trainer state - will start from step 0")
        print(f"{'='*60}\n")
    elif model_exists and weights_exist:
        print(f"\n{'='*60}")
        print("LOADING TRAINER STATE FROM BASE MODEL")
        print(f"{'='*60}")
        success = trainer.load_checkpoint(model_path)
        if success:
            initial_global_step = trainer.global_step
            print("✓ Loaded trainer state from base model")
            print(f"  Global step: {initial_global_step}")
            print(f"  Run step: {trainer.run_step}")
        else:
            print("✗ Failed to load trainer state - will start from step 0")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("NO CHECKPOINT FOUND - STARTING FROM STEP 0")
        print(f"{'='*60}\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Debug: Print final weight sample to verify training happened
        with torch.no_grad():
            weight_sample = model.model.embed_tokens.weight[0][:5].tolist()
            weight_mean = model.model.embed_tokens.weight.mean().item()
            weight_std = model.model.embed_tokens.weight.std().item()
            print(f"\n{'='*60}")
            print("FINAL MODEL STATE")
            print(f"{'='*60}")
            print(f"Weight sample: {[f'{x:.4f}' for x in weight_sample]}")
            print(f"Weight mean: {weight_mean:.6f}, std: {weight_std:.6f}")
            print(f"{'='*60}\n")

        # Final Save - CRITICAL: Always save to checkpoint path
        print(f"\n{'='*60}")
        print("SAVING CHECKPOINT - THIS IS CRITICAL FOR PROGRESSIVE TRAINING")
        print(f"{'='*60}")
        print(f"Saving to: {checkpoint_path}")
        trainer.save_checkpoint(checkpoint_path)
        print("✓ Model weights saved")

        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_path)
            print("✓ Tokenizer saved")

        # Save dataset state
        batches_processed_in_this_run = (
            (trainer.global_step - initial_global_step)
            * train_config.batch_size
            * train_config.gradient_accumulation_steps
        )
        new_processed = processed_items + batches_processed_in_this_run

        with open(state_path, "w") as f:
            json.dump({"processed_items": new_processed}, f)

        # Save state synced with checkpoint
        checkpoint_state_backup = os.path.join(checkpoint_path, "dataset_state.json")
        with open(checkpoint_state_backup, "w") as f:
            json.dump({"processed_items": new_processed}, f)

        print(f"Checkpoint saved. Total processed: {new_processed}")


if __name__ == "__main__":
    main()
