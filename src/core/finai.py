"""Main FinAI application (local GPT-style LLM)"""
import os
import torch
from src.config import Config
from src.data.tokenizer import Tokenizer
from src.core.context import ConversationContext

try:
    from src.models.language_model_pytorch import LanguageModel as GPTModel
    PYTORCH_AVAILABLE = True
except Exception:
    PYTORCH_AVAILABLE = False
    GPTModel = None


class FinAI:
    def __init__(self):
        self.config = Config()
        self.tokenizer = None
        self.model = None
        self.context = ConversationContext()
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        os.makedirs(self.config.DATASET_DIR, exist_ok=True)

    def initialize(self):
        print("=" * 70)
        print("FinAI - Local LLM")
        print("=" * 70)
        print()
        if self._load_models():
            print("✓ Loaded trained model\n")
            print("Type 'exit' to quit.")
            return True
        else:
            print("No trained model found. Train first with: main.py train <file>")
            return False

    def train_from_file(self, filepath: str, steps: int = None, batch_size: int = None, learning_rate: float = None, block_size: int = None, use_gpu: bool = True):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPT training")
        print(f"Loading dataset from {filepath}...")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        # Tokenizer (byte-level, no fitting)
        self.tokenizer = Tokenizer()
        tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        # Model
        steps = steps or self.config.TRAIN_STEPS
        batch_size = batch_size or self.config.BATCH_SIZE
        learning_rate = learning_rate or self.config.LEARNING_RATE
        block_size = block_size or self.config.BLOCK_SIZE
        self.model = GPTModel(vocab_size=self.tokenizer.vocab_size, block_size=block_size, use_gpu=use_gpu)

        # Send tokens to device
        tokens_tensor = tokens_tensor.to(self.model.device)
        print(f"Training GPT: steps={steps}, batch_size={batch_size}, block_size={block_size}")
        self.model.train_on_tokens(tokens_tensor, steps=steps, batch_size=batch_size, learning_rate=learning_rate)

        self._save_models()
        print("✓ Training complete and model saved")

    def _load_models(self) -> bool:
        try:
            self.tokenizer = Tokenizer.load(self.config.TOKENIZER_PATH)
            self.model = GPTModel.load(self.config.LANGUAGE_MODEL_PATH, use_gpu=True)
            return True
        except Exception:
            return False

    def _save_models(self):
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        self.tokenizer.save(self.config.TOKENIZER_PATH)
        self.model.save(self.config.LANGUAGE_MODEL_PATH)

    def generate_response(self, user_input: str) -> str:
        if not self.model or not self.tokenizer:
            return "Model not loaded. Train first."
        # Build prompt with short context
        context_str = self.context.get_context_string(num_messages=3)
        if context_str:
            prompt = f"{context_str}\nuser: {user_input}\nassistant:"
        else:
            prompt = f"user: {user_input}\nassistant:"
        # Encode and generate
        x = self.tokenizer.encode(prompt)
        x = torch.tensor([x], dtype=torch.long, device=self.model.device)
        y = self.model.generate(
            x,
            max_new_tokens=self.config.MAX_NEW_TOKENS,
            temperature=self.config.TEMPERATURE,
            top_k=self.config.TOP_K,
        )
        text = self.tokenizer.decode(y[0].tolist())
        # Extract assistant part
        if "assistant:" in text:
            text = text.split("assistant:")[-1]
        return text.strip()

    def chat(self):
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "bye"}:
                    print("\nFinAI: Goodbye!\n")
                    break
                self.context.add_message('user', user_input)
                response = self.generate_response(user_input)
                self.context.add_message('assistant', response)
                print(f"\nFinAI: {response}\n")
            except KeyboardInterrupt:
                print("\n\nFinAI: Goodbye!\n")
                break

    def run(self):
        if self.initialize():
            self.chat()
        else:
            print("Train first with: main.py train <file>")
