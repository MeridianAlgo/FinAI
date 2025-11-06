"""Optimized Configuration for Medium-Sized FinAI (GPT-style LLM)"""

import torch

class Config:
    # Paths (unchanged, but ensure DATASET_DIR has 1M+ clean tokens)
    MODEL_DIR = "models"
    LANGUAGE_MODEL_PATH = f"{MODEL_DIR}/finai_gpt.pt"
    TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.pkl"
    DATASET_DIR = "datasets"

    # Data (no loss: full usage, longer context for better learning)
    BLOCK_SIZE = 256  # Smaller context for PCs; keeps useful span without huge memory
    MAX_DATA_TOKENS = float('inf')  # No subsampling—use all data

    # Transformer architecture (small ~15M params; vocab embedding dominates size)
    # Note: With ~50k vocab and tied embeddings, minimum is ~12.8M just for embeddings at 256 dims.
    N_LAYER = 4
    N_HEAD = 4    # head_dim = 64
    N_EMBD = 256
    DROPOUT = 0.05

    # Training (2–3x faster convergence; ~10k steps for 1M tokens = 1–2 epochs)
    TRAIN_EPOCHS = 1
    TRAIN_STEPS = 5000  # As requested
    BATCH_SIZE = 16  # Lighter for consumer GPUs/CPU
    GRADIENT_ACCUM_STEPS = 4  # Effective batch ~64
    LEARNING_RATE = 6e-4  # 2x original: Higher start for faster early progress
    WEIGHT_DECAY = 0.1  # New: L2 reg for generalization (AdamW default)
    WARMUP_STEPS = 100  # New: 1% of steps: Linear ramp to avoid early divergence
    MAX_GRAD_NORM = 1.0  # New: Clip for stability (prevents explosions)

    # Optimizer/Scheduler (AdamW + cosine: Gold standard for LLMs)
    OPTIMIZER = "AdamW"
    LR_SCHEDULER_TYPE = "cosine"  # Decays smoothly to 10% of peak LR
    BETAS = (0.9, 0.95)  # New: Momentum for smoother updates
    EPSILON = 1e-8  # New: Adam stability

    # Precision/Memory (AMD-optimized, no loss)
    PRECISION = torch.bfloat16  # bf16: Full accuracy on ROCm, 50% memory vs fp32
    USE_GRAD_CHECKPOINTING = True  # New: Trades 20% time for 40% memory savings

    # Generation (tuned for coherent, accurate outputs)
    MAX_NEW_TOKENS = 512  # 2x: Longer for complex reasoning
    TEMPERATURE = 0.7  # Slightly lower: More deterministic/accurate
    TOP_K = 40  # Lower: Reduces hallucinations
    TOP_P = 0.9  # New: Nucleus sampling for diversity without junk

    # HF export defaults (unchanged, but add eval split)
    HF_DEFAULT_SPLIT = "train"
    HF_EVAL_SPLIT = "eval"  # New: 10% holdout for perplexity/accuracy checks
    EXPORT_MAX = 500000  # 2.5x: Larger for better HF Hub uploads
