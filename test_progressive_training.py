"""Test script to verify progressive loss decrease across multiple training runs"""

import json
import os
import shutil
import torch
from transformers import AutoTokenizer

from fin_ai.model.configuration_next import FinAINextConfig
from fin_ai.model.modeling_next import FinAINextForCausalLM
from fin_ai.training.next_trainer import NextTrainingConfig, TernaryTrainer


class SimpleDataset(torch.utils.data.IterableDataset):
    """Simple repeating dataset for testing"""

    def __init__(self, tokenizer, num_samples=100):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.texts = [
            "The stock market is a financial marketplace.",
            "Investors buy and sell securities daily.",
            "Economic indicators affect market trends.",
            "Portfolio diversification reduces investment risk.",
        ] * 25  # Repeat to get 100 samples

    def __iter__(self):
        for text in self.texts[: self.num_samples]:
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=64,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
            labels = input_ids.clone()

            # Mask padding
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels[input_ids == pad_token_id] = -100

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }


def run_training_iteration(iteration, checkpoint_path, state_path):
    """Run a single training iteration and return final loss"""
    print(f"\n{'='*60}")
    print(f"TRAINING ITERATION {iteration}")
    print(f"{'='*60}\n")

    # Configuration
    config = FinAINextConfig(
        vocab_size=151665,
        hidden_size=256,  # Smaller for faster testing
        num_layers=4,  # Fewer layers
        liquid_state_dim=64,
        gradient_checkpointing=False,
        tie_word_embeddings=True,
    )

    # Load or create model
    checkpoint_exists = os.path.exists(os.path.join(checkpoint_path, "config.json"))
    weights_exist = os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))

    if checkpoint_exists and weights_exist:
        print(f"Loading model from {checkpoint_path}...")
        model = FinAINextForCausalLM.from_pretrained(checkpoint_path)
        print("Model loaded successfully!")
    else:
        print("Creating new model...")
        model = FinAINextForCausalLM(config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Dataset
    dataset = SimpleDataset(
        tokenizer, num_samples=20
    )  # Small dataset for quick testing
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # Training config
    train_config = NextTrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=10,  # Just 10 steps per iteration
        total_steps=100,  # Total across all iterations
        learning_rate=5e-4,
        output_dir=checkpoint_path,
        save_steps=10,
    )

    # Trainer
    trainer = TernaryTrainer(model, dataloader, train_config)

    # Load checkpoint state if exists
    if checkpoint_exists and weights_exist:
        trainer.load_checkpoint(checkpoint_path)
        print(f"Resumed from global step: {trainer.global_step}")

    # Record starting loss
    model.eval()
    with torch.no_grad():
        first_batch = next(iter(dataloader))
        first_batch = {k: v.to(trainer.device) for k, v in first_batch.items()}
        outputs = model(**first_batch)
        starting_loss = outputs.loss.item()
    model.train()

    print(f"\n[ITERATION {iteration}] Starting Loss: {starting_loss:.4f}")
    print(f"[ITERATION {iteration}] Global Step: {trainer.global_step}")
    print(
        f"[ITERATION {iteration}] Learning Rate: {trainer.scheduler.get_last_lr()[0]:.2e}\n"
    )

    # Train
    trainer.train()

    # Record ending loss
    model.eval()
    with torch.no_grad():
        first_batch = next(iter(dataloader))
        first_batch = {k: v.to(trainer.device) for k, v in first_batch.items()}
        outputs = model(**first_batch)
        ending_loss = outputs.loss.item()

    print(f"\n[ITERATION {iteration}] Ending Loss: {ending_loss:.4f}")
    print(f"[ITERATION {iteration}] Loss Decrease: {starting_loss - ending_loss:.4f}")

    # Save checkpoint
    trainer.save_checkpoint(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    # Save state
    with open(state_path, "w") as f:
        json.dump(
            {
                "iteration": iteration,
                "global_step": trainer.global_step,
                "starting_loss": starting_loss,
                "ending_loss": ending_loss,
            },
            f,
            indent=2,
        )

    return starting_loss, ending_loss


def main():
    print("\n" + "=" * 60)
    print("PROGRESSIVE TRAINING TEST")
    print("Testing that loss decreases across multiple training runs")
    print("=" * 60 + "\n")

    # Setup paths
    checkpoint_path = "./test_checkpoint"
    state_path = "./test_state.json"

    # Clean up from previous tests
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    if os.path.exists(state_path):
        os.remove(state_path)

    # Run 3 training iterations
    results = []
    for i in range(1, 4):
        starting_loss, ending_loss = run_training_iteration(
            i, checkpoint_path, state_path
        )
        results.append(
            {
                "iteration": i,
                "starting_loss": starting_loss,
                "ending_loss": ending_loss,
            }
        )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60 + "\n")

    print("| Iteration | Starting Loss | Ending Loss | Decrease |")
    print("|-----------|---------------|-------------|----------|")
    for r in results:
        decrease = r["starting_loss"] - r["ending_loss"]
        print(
            f"| {r['iteration']:9d} | {r['starting_loss']:13.4f} | {r['ending_loss']:11.4f} | {decrease:8.4f} |"
        )

    # Verify progressive decrease
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60 + "\n")

    success = True
    for i in range(len(results) - 1):
        current_start = results[i]["starting_loss"]
        next_start = results[i + 1]["starting_loss"]

        if next_start < current_start:
            print(
                f"✓ Iteration {i+2} starting loss ({next_start:.4f}) < Iteration {i+1} starting loss ({current_start:.4f})"
            )
        else:
            print(
                f"✗ FAILED: Iteration {i+2} starting loss ({next_start:.4f}) >= Iteration {i+1} starting loss ({current_start:.4f})"
            )
            success = False

    # Cleanup
    print("\nCleaning up test files...")
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    if os.path.exists(state_path):
        os.remove(state_path)

    if success:
        print("\n✓ TEST PASSED: Loss decreases progressively across training runs!")
    else:
        print("\n✗ TEST FAILED: Loss does not decrease progressively!")
        exit(1)


if __name__ == "__main__":
    main()
