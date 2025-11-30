"""Optimized Configuration for FinAI (Smart & Efficient ~14M Params)"""

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

    # Transformer architecture (~14M parameters - "Smart & Efficient")
    # A significant upgrade from the previous 1M param model.
    # Capable of understanding more complex financial contexts.
    N_LAYER = 8
    N_HEAD = 8
    N_EMBD = 384
    DROPOUT = 0.1

    # Training
    TRAIN_EPOCHS = 1
    TRAIN_STEPS = 1000  # Adjusted for larger model/slower steps
    BATCH_SIZE = 16     # Lower batch size for larger model
    GRADIENT_ACCUM_STEPS = 8 # Higher accumulation to maintain effective batch size
    LEARNING_RATE = 6e-4  # Standard LR for this scale
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 100
    MAX_GRAD_NORM = 1.0

    # Optimizer
    OPTIMIZER = "AdamW"
    LR_SCHEDULER_TYPE = "cosine"
    BETAS = (0.9, 0.95)
    EPSILON = 1e-8

    # Precision
    PRECISION = torch.float32  # Keep fp32 for stability on CPU/MPS
    USE_GRAD_CHECKPOINTING = False

    # Generation
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.8
    TOP_K = 40
    TOP_P = 0.9

    # HF export
    HF_DEFAULT_SPLIT = "train"
    HF_EVAL_SPLIT = "eval"
    EXPORT_MAX = 500000
