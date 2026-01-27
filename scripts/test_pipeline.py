#!/usr/bin/env python3
"""
Test script for the full FinAI-Core v2.2 pipeline:
Ingestion -> Dataset Loading -> Training -> Generation
"""

import os
import torch
import json
from transformers import AutoTokenizer
from fin_ai.model.modeling_finai import FinAIForCausalLM
from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.training.trainer import FinAITrainer, TrainingConfig, DatasetCycler
from fin_ai.data.dataset import FinAIDataset, tokenize_and_chunk
from fin_ai.data.dataloader import create_dataloader

def test_pipeline():
    print("--- 1. Testing Ingestion ---")
    os.system("python continual_ingest.py --output data/pipeline_test.jsonl")

    if not os.path.exists("data/pipeline_test.jsonl"):
        print("Error: Ingestion failed to create data file.")
        return

    print("\n--- 2. Loading Ingested Data ---")
    texts = []
    with open("data/pipeline_test.jsonl", "r") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Small config for fast test
    config = FinAIConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        use_moe=True,
        num_experts=4
    )
    config.vocab_size = len(tokenizer)

    tokenized_chunks = tokenize_and_chunk(texts, tokenizer, max_seq_len=128)
    dataset = FinAIDataset(tokenized_chunks, max_seq_len=128)
    dataloader = create_dataloader(dataset, batch_size=1)

    print(f"Dataset size: {len(dataset)} sequences")

    print("\n--- 3. Testing Training (2 steps) ---")
    model = FinAIForCausalLM(config)
    train_config = TrainingConfig(
        max_steps=2,
        gradient_accumulation_steps=1,
        log_steps=1,
        save_steps=2,
        output_dir="./checkpoints/test_run"
    )

    # Mock datasets.yaml for DatasetCycler if needed, or pass None
    os.makedirs("config", exist_ok=True)
    with open("config/datasets_test.yaml", "w") as f:
        f.write("datasets:\n  - name: pipeline_test\n    path: data/pipeline_test.jsonl")

    cycler = DatasetCycler("config/datasets_test.yaml", "checkpoints/test_run/state.json")

    trainer = FinAITrainer(
        model=model,
        train_dataloader=dataloader,
        config=train_config,
        dataset_cycler=cycler
    )

    trainer.train()

    print("\n--- 4. Testing Generation ---")
    model.eval()
    prompt = "Financial markets are"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {gen_text}")

    print("\n--- Pipeline Test Complete ---")

if __name__ == "__main__":
    test_pipeline()
