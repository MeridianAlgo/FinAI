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

    return torch.utils.data.DataLoader(
        CustomIterableDataset(gen), batch_size=batch_size
    )


def main():
    print("Initializing FinAI-Next (Liquid-BitNet) Overhaul...")

    # Path settings
    model_path = "./model"
    state_path = "dataset_state.json"

    # 1. Load Dataset State
    processed_items = 0
    # Prefer state synced with weights if it exists
    backup_state_path = os.path.join(model_path, "dataset_state.json")

    if os.path.exists(backup_state_path):
        with open(backup_state_path, "r") as f:
            state = json.load(f)
            processed_items = state.get("processed_items", 0)
        print(f"Resuming from synced dataset index: {processed_items}")
    elif os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
            processed_items = state.get("processed_items", 0)
        print(f"Resuming from root dataset index: {processed_items}")

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
    model_exists = os.path.exists(os.path.join(model_path, "config.json"))
    weights_exist = os.path.exists(
        os.path.join(model_path, "model.safetensors")
    ) or os.path.exists(os.path.join(model_path, "pytorch_model.bin"))

    if model_exists:
        if weights_exist:
            print(f"Loading existing model and weights from {model_path}...")
            try:
                model = FinAINextForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=False,
                )
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights: {e}. Reinitializing weights.")
                model = FinAINextForCausalLM(config)
        else:
            print(
                f"Config found at {model_path} but weights (model.safetensors) are missing. Initializing fresh weights."
            )
            model = FinAINextForCausalLM(config)
    else:
        print("Initializing new model and config from scratch.")
        model = FinAINextForCausalLM(config)

    # Debug: Print initial weight sample
    with torch.no_grad():
        weight_sample = model.model.embed_tokens.weight[0][:5].tolist()
        print(f"DEBUG: Initial weight sample: {weight_sample}")

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
        output_dir="./model",
    )

    # 7. Training
    trainer = TernaryTrainer(model, dataloader, train_config)

    initial_global_step = 0
    # Load trainer state if it exists
    if model_exists:
        trainer.load_checkpoint(model_path)
        initial_global_step = trainer.global_step

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Debug: Print final weight sample
        with torch.no_grad():
            weight_sample = model.model.embed_tokens.weight[0][:5].tolist()
            print(f"DEBUG: Final weight sample: {weight_sample}")

        # Final Save
        print("Saving final state to local storage...")
        trainer.save_checkpoint(model_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(model_path)

        # Save dataset state
        # `processed_items` loaded from dataset_state.json is the initial skip count for this run.
        # `initial_global_step` is the global_step loaded from trainer_state.pt (or 0 if new).
        # `trainer.global_step` is the final global_step after training.
        # Number of optimization steps completed in this run = trainer.global_step - initial_global_step.
        # Each optimization step processes `train_config.batch_size * train_config.gradient_accumulation_steps` dataloader batches.
        batches_processed_in_this_run = (
            (trainer.global_step - initial_global_step)
            * train_config.batch_size
            * train_config.gradient_accumulation_steps
        )
        new_processed = processed_items + batches_processed_in_this_run

        with open(state_path, "w") as f:
            json.dump({"processed_items": new_processed}, f)

        # Save a backup synced with weights
        with open(backup_state_path, "w") as f:
            json.dump({"processed_items": new_processed}, f)

        print(f"Final state saved. Total processed: {new_processed}")


if __name__ == "__main__":
    main()
