"""FinAI-Next Training Script (Liquid-BitNet)"""

import json
import os

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


def create_dataloader(tokenizer, batch_size=4, block_size=1024, skip_items=0):
    print(f"Initializing Rotational DataLoader (skipping {skip_items} items)...")

    # Dataset Registry for "Best Model" training
    # Mixing Encyclopedia, Web Edu, and Instruction/Chat data
    dataset_configs = [
        ("wikitext", "wikitext-103-raw-v1", "train", "text"),
        ("HuggingFaceFW/fineweb-edu", "default", "train", "text"),
        ("mlabonne/guanaco-llama2-1k", "default", "train", "text"),
    ]

    iterators = []
    for path, name, split, text_col in dataset_configs:
        try:
            print(f"  - Loading {path}/{name}...")
            # Use distinct buffer sizes to avoid synchronization artifacts
            ds = load_dataset(path, name, split=split, streaming=True)
            if skip_items > 0:
                ds = ds.skip(skip_items // len(dataset_configs))
            iterators.append((iter(ds), text_col))
        except Exception as e:
            print(f"  [WARN] Failed to load {path}: {e}")

    def gen():
        # Round-robin rotation strategy
        cycle_idx = 0
        local_processed = 0

        while True:
            # Get next iterator config
            iterator, text_col = iterators[cycle_idx % len(iterators)]

            try:
                item = next(iterator)
                cycle_idx += 1

                text = item.get(text_col, "")
                # Handle inconsistent column names or empty text
                if not isinstance(text, str) or not text.strip():
                    continue

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
                local_processed += 1

            except StopIteration:
                # If a stream ends, warn and remove or restart?
                # For now, restarting is safer for infinite training.
                print(
                    f"\n[INFO] Dataset {cycle_idx % len(iterators)} exhausted. cycling..."
                )
                # In a real infinite stream we shouldn't hit this often for web data.
                pass
            except Exception as e:
                print(f"[WARN] Data processing error: {e}")
                cycle_idx += 1

    return torch.utils.data.DataLoader(
        CustomIterableDataset(gen), batch_size=batch_size
    )


def main():
    print("Initializing FinAI-Next (Liquid-BitNet) Overhaul...")

    # Path settings
    model_path = "./checkpoints_next/model"
    state_path = "dataset_state.json"

    # 1. Load Dataset State
    processed_items = 0
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
            processed_items = state.get("processed_items", 0)
        print(f"Resuming from dataset index: {processed_items}")

    # 2. Configuration
    config = FinAINextConfig(
        vocab_size=151665,
        hidden_size=1536,
        num_layers=24,
        liquid_state_dim=384,
        gradient_checkpointing=True,
        tie_word_embeddings=True,
    )
    print(f"Configuration: {config}")

    # 3. Model Initialization or Loading
    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Loading existing model from {model_path}...")
        try:
            model = FinAINextForCausalLM.from_pretrained(
                model_path, config=config, ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(f"Error loading model: {e}. Reinitializing from scratch.")
            model = FinAINextForCausalLM(config)
    else:
        print("Initializing new model from scratch.")
        model = FinAINextForCausalLM(config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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
    # If running in GHA, we might want to cap steps (e.g. 100 steps per hour)
    max_steps = int(os.getenv("MAX_STEPS", "1000"))

    train_config = NextTrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        learning_rate=5e-5,
        output_dir="./checkpoints_next",
    )

    # 7. Training
    trainer = TernaryTrainer(model, dataloader, train_config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Final Save - Fix Windows file locking issue
        print("Saving final state to local storage...")

        # Move model to CPU and clear CUDA cache to release file handles
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Delete old checkpoint files to release memory-mapped handles
        import shutil

        if os.path.exists(model_path):
            print(f"Removing old checkpoint at {model_path}...")
            shutil.rmtree(model_path, ignore_errors=True)
            import time

            time.sleep(1.0)  # Give Windows time to release handles

        model.save_pretrained(model_path, safe_serialization=True)

        # Save dataset state (use the trainer's global step to estimate or pass back from gen)
        # For simple tracking, we'll update based on steps * batch *
        # accumulation
        new_processed = processed_items + (
            trainer.global_step
            * train_config.batch_size
            * train_config.gradient_accumulation_steps
        )
        with open(state_path, "w") as f:
            json.dump({"processed_items": new_processed}, f)
        print(f"Final state saved. Total processed: {new_processed}")


if __name__ == "__main__":
    main()
