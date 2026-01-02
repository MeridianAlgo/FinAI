from torch.utils.data import DataLoader

from fin_ai.data.dataset import FinAIDataset, tokenize_and_chunk
from fin_ai.model.config import FinAIConfig
from fin_ai.model.transformer import FinAIModel
from fin_ai.training.trainer import FinAITrainer, TrainingConfig


def make_dummy_dataset(vocab_size=1000, seq_len=32, samples=32):
    import random

    rng = random.Random(0)
    texts = [
        " ".join([str(rng.randint(1, 100)) for _ in range(seq_len)])
        for _ in range(samples)
    ]
    # Use a small GPT2 tokenizer for consistency
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenized = tokenize_and_chunk(texts, tokenizer, max_seq_len=seq_len, min_length=1)
    return FinAIDataset(tokenized, max_seq_len=seq_len)


def test_trainer_runs_and_decreases_loss(tmp_path):
    # Small model and dataset for a quick smoke test
    cfg = FinAIConfig.from_preset("tiny")
    cfg.vocab_size = 50257
    model = FinAIModel(cfg)

    ds = make_dummy_dataset(seq_len=32, samples=64)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    training_cfg = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        max_steps=4,
        save_steps=10000,
        use_wandb=False,
        fp16=False,
        output_dir=str(tmp_path / "checkpoints"),
    )

    trainer = FinAITrainer(
        model=model,
        train_dataloader=loader,
        config=training_cfg,
        dataset_cycler=None,
    )

    # Run training (should complete quickly)
    trainer.train()

    # After training, we should have progressed global_step
    assert trainer.global_step >= training_cfg.max_steps
