"""Optimized Configuration for 1M Parameter FinAI (Nano-GPT)"""

import torch

class Config:
    # Paths
    MODEL_DIR = "models"
    LANGUAGE_MODEL_PATH = f"{MODEL_DIR}/finai_gpt.pt"
    TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.pkl"
    DATASET_DIR = "datasets"

    # Data
    BLOCK_SIZE = 256  # Context window
    MAX_DATA_TOKENS = float('inf')

    # Transformer architecture (~1M parameters - "Hella Smart & Fast")
    # Vocab (259) * 128 = 33k
    # Layers (5) * (12 * 128^2) = 983k
    # Total ≈ 1.05M params
    N_LAYER = 5
    N_HEAD = 4    # head_dim = 32
    N_EMBD = 128
    DROPOUT = 0.05

    # Training
    TRAIN_EPOCHS = 1
    TRAIN_STEPS = 2000  # Faster cycles for daily training
    BATCH_SIZE = 32     # Higher batch size for smaller model
    GRADIENT_ACCUM_STEPS = 4
    LEARNING_RATE = 1e-3  # Higher LR for smaller model
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 100
    MAX_GRAD_NORM = 1.0

    # Optimizer
    OPTIMIZER = "AdamW"
    LR_SCHEDULER_TYPE = "cosine"
    BETAS = (0.9, 0.95)
    EPSILON = 1e-8

    # Precision
    PRECISION = torch.float32  # Use fp32 for such a small model (better stability), or bf16 if available
    USE_GRAD_CHECKPOINTING = False  # Not needed for 1M params

    # Generation
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.8
    TOP_K = 40
    TOP_P = 0.9

    # HF export
    HF_DEFAULT_SPLIT = "train"
    HF_EVAL_SPLIT = "eval"
    EXPORT_MAX = 500000
