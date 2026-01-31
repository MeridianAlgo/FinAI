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
    dataset_name, tokenizer, batch_size=4, block_size=1024, skip_items=0
):
    print(f"Loading dataset: {dataset_name} (skipping {skip_items} items)...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    def gen():
        for i, item in enumerate(dataset):
            if i < skip_items:
                continue
            tokens = tokenizer(
                item["text"],
                truncation=True,
                max_length=block_size,
                padding="max_length",
            )
            yield {
                "input_ids": torch.tensor(tokens["input_ids"]),
                "labels": torch.tensor(tokens["input_ids"]),
                "processed_idx": i,
            }

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
    )

    # 3. Model Initialization or Loading
    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Loading existing model from {model_path}...")
        model = FinAINextForCausalLM.from_pretrained(model_path)
    else:
        print("Initializing new model from scratch.")
        model = FinAINextForCausalLM(config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # 5. Dataset with Skip
    dataloader = create_dataloader(
        "HuggingFaceFW/fineweb-edu",
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
