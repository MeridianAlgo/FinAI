#!/usr/bin/env python3
import sys
import argparse
from src.core.finai import FinAI


def main():
    parser = argparse.ArgumentParser(description="FinAI (Local LLM) CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train from local text file
    p_train = subparsers.add_parser("train", help="Train from a local .txt file (plain text corpus)")
    p_train.add_argument("dataset_file", help="Path to .txt dataset file")
    p_train.add_argument("--steps", type=int, default=None, help="Training steps (tokens batches)")
    p_train.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p_train.add_argument("--lr", type=float, default=None, help="Learning rate")
    p_train.add_argument("--block-size", type=int, default=None, help="Context window (tokens)")
    p_train.add_argument("--cpu", action="store_true", help="Force CPU training")

    # Chat
    subparsers.add_parser("chat", help="Interactive chat with the trained model")

    # Generate from prompt
    p_gen = subparsers.add_parser("generate", help="Generate from a prompt")
    p_gen.add_argument("prompt", help="Prompt text")

    args = parser.parse_args()
    finai = FinAI()

    if args.command == "train":
        finai.train_from_file(
            filepath=args.dataset_file,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            block_size=args.block_size,
            use_gpu=(not args.cpu),
        )
        return 0

    if args.command == "generate":
        if finai.initialize():
            out = finai.generate_response(args.prompt)
            print(out)
            return 0
        return 1

    # Default to chat
    finai.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
