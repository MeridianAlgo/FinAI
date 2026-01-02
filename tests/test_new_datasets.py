"""Test new datasets before adding to config"""

from datasets import load_dataset

# Test datasets to add
test_datasets = [
    # Math reasoning
    {"name": "openai/gsm8k", "subset": "main", "split": "train", "column": "question"},
    # Common sense reasoning
    {"name": "commonsense_qa", "subset": None, "split": "train", "column": "question"},
    # Code - Python
    {
        "name": "codeparrot/github-code",
        "subset": "Python",
        "split": "train",
        "column": "code",
    },
    # Instruction following
    {"name": "tatsu-lab/alpaca", "subset": None, "split": "train", "column": "text"},
    # Conversations
    {
        "name": "HuggingFaceH4/ultrachat_200k",
        "subset": None,
        "split": "train_sft",
        "column": "messages",
    },
    # Scientific
    {"name": "allenai/c4", "subset": "en", "split": "train", "column": "text"},
    # Books
    {"name": "pg19", "subset": None, "split": "train", "column": "text"},
]

"""This script was moved to `legacy/` to avoid running dataset downloads during pytest.

Run manually from `legacy/new_datasets_script.py` when you want to probe datasets.
"""


def main():
    print("New datasets checks were moved to legacy/ for manual runs.")


if __name__ == "__main__":
    main()
