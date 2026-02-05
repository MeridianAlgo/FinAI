"""
Test Cascading Loss - Verify that loss continues from previous run
This script runs a mini training session and verifies the checkpoint is saved properly
"""

import os
import torch
from transformers import AutoTokenizer
from fin_ai.model.configuration_next import FinAINextConfig
from fin_ai.model.modeling_next import FinAINextForCausalLM
from fin_ai.training.next_trainer import NextTrainingConfig, TernaryTrainer


def create_dummy_dataloader(tokenizer, num_batches=10):
    """Create a small dummy dataset for testing"""

    class DummyDataset(torch.utils.data.IterableDataset):
        def __init__(self, tokenizer, num_batches):
            self.tokenizer = tokenizer
            self.num_batches = num_batches

        def __iter__(self):
            for i in range(self.num_batches):
                # Create dummy text
                text = f"This is test sentence number {i}. " * 20
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=128,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_ids = tokens["input_ids"].squeeze(0)
                labels = input_ids.clone()

                # Mask padding
                if self.tokenizer.pad_token_id is not None:
                    labels[input_ids == self.tokenizer.pad_token_id] = -100

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }

    dataset = DummyDataset(tokenizer, num_batches)
    return torch.utils.data.DataLoader(dataset, batch_size=2)


def run_mini_training(run_number, checkpoint_path="./test_checkpoint"):
    """Run a mini training session"""
    print(f"\n{'='*70}")
    print(f"TRAINING RUN #{run_number}")
    print(f"{'='*70}\n")

    # Configuration
    config = FinAINextConfig(
        vocab_size=151665,
        hidden_size=256,  # Smaller for faster testing
        num_layers=4,  # Fewer layers for faster testing
        liquid_state_dim=64,
        gradient_checkpointing=False,
        tie_word_embeddings=True,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model loading/initialization
    checkpoint_exists = os.path.exists(os.path.join(checkpoint_path, "config.json"))
    weights_exist = os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))

    if checkpoint_exists and weights_exist:
        print(f"✓ Loading model from checkpoint: {checkpoint_path}")
        model = FinAINextForCausalLM.from_pretrained(
            checkpoint_path,
            config=config,
            ignore_mismatched_sizes=False,
        )
        print("✓ Model loaded - CONTINUING TRAINING")
    else:
        print("⚠ No checkpoint found - initializing fresh model")
        model = FinAINextForCausalLM(config)

    # Print initial weights
    with torch.no_grad():
        weight_sample = model.model.embed_tokens.weight[0][:5].tolist()
        weight_mean = model.model.embed_tokens.weight.mean().item()
        print("\nInitial weights:")
        print(f"  Sample: {[f'{x:.4f}' for x in weight_sample]}")
        print(f"  Mean: {weight_mean:.6f}")

    # Create dataloader
    dataloader = create_dummy_dataloader(tokenizer, num_batches=20)

    # Training config
    train_config = NextTrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=5,  # Just 5 steps for quick testing
        total_steps=100,
        learning_rate=1e-4,
        output_dir=checkpoint_path,
        save_steps=10,  # Won't trigger during run, only final save
        log_steps=1,
    )

    # Trainer
    trainer = TernaryTrainer(model, dataloader, train_config)

    # Load trainer state if exists
    if checkpoint_exists and weights_exist:
        trainer.load_checkpoint(checkpoint_path)
        print(f"✓ Loaded trainer state - starting from step {trainer.global_step}")

    # Record initial loss
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        outputs = model(**batch)
        initial_loss = outputs.loss.item()
    model.train()

    print(f"\n{'='*70}")
    print(f"INITIAL LOSS: {initial_loss:.4f}")
    print(f"{'='*70}\n")

    # Train
    trainer.train()

    # Record final loss
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        outputs = model(**batch)
        final_loss = outputs.loss.item()
    model.train()

    print(f"\n{'='*70}")
    print(f"FINAL LOSS: {final_loss:.4f}")
    print(
        f"Loss change: {initial_loss:.4f} -> {final_loss:.4f} (Δ {final_loss - initial_loss:.4f})"
    )
    print(f"{'='*70}\n")

    # Save checkpoint
    print("Saving checkpoint...")
    trainer.save_checkpoint(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    # Save run info
    run_info = {
        "run_number": run_number,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "global_step": trainer.global_step,
    }

    # Print final weights
    with torch.no_grad():
        weight_sample = model.model.embed_tokens.weight[0][:5].tolist()
        weight_mean = model.model.embed_tokens.weight.mean().item()
        print("\nFinal weights:")
        print(f"  Sample: {[f'{x:.4f}' for x in weight_sample]}")
        print(f"  Mean: {weight_mean:.6f}")

    return run_info


def main():
    """Run multiple training sessions to test cascading"""
    checkpoint_path = "./test_checkpoint"

    # Clean up old test checkpoint
    if os.path.exists(checkpoint_path):
        import shutil

        print(f"Cleaning up old test checkpoint: {checkpoint_path}")
        shutil.rmtree(checkpoint_path)

    results = []

    # Run 3 training sessions
    for run_num in range(1, 4):
        run_info = run_mini_training(run_num, checkpoint_path)
        results.append(run_info)

        print(f"\n{'='*70}")
        print(f"RUN #{run_num} COMPLETE")
        print(f"{'='*70}\n")

    # Analyze results
    print(f"\n{'='*70}")
    print("CASCADING LOSS ANALYSIS")
    print(f"{'='*70}\n")

    for i, result in enumerate(results):
        print(f"Run {result['run_number']}:")
        print(f"  Initial loss: {result['initial_loss']:.4f}")
        print(f"  Final loss:   {result['final_loss']:.4f}")
        print(f"  Global step:  {result['global_step']}")

        if i > 0:
            prev_final = results[i - 1]["final_loss"]
            curr_initial = result["initial_loss"]
            diff = abs(curr_initial - prev_final)

            if diff < 0.1:  # Should be very close
                print(f"  ✓ CASCADING WORKS! (diff: {diff:.4f})")
            else:
                print(
                    f"  ✗ CASCADING FAILED! Previous final: {prev_final:.4f}, Current initial: {curr_initial:.4f} (diff: {diff:.4f})"
                )
        print()

    print(f"{'='*70}\n")

    # Clean up
    if os.path.exists(checkpoint_path):
        import shutil

        shutil.rmtree(checkpoint_path)
        print("Cleaned up test checkpoint")


if __name__ == "__main__":
    main()
