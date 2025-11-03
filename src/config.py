"""Configuration settings for FinAI (GPT-style local LLM)"""

class Config:
    # Paths
    MODEL_DIR = "models"
    LANGUAGE_MODEL_PATH = f"{MODEL_DIR}/finai_gpt.pt"
    TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.pkl"

    # Data
    DATASET_DIR = "datasets"

    # Transformer architecture
    BLOCK_SIZE = 256
    N_LAYER = 4
    N_HEAD = 4
    N_EMBD = 256
    DROPOUT = 0.1

    # Training
    TRAIN_STEPS = 2000
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4

    # Generation
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.8
    TOP_K = 50
