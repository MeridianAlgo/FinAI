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

    def train_from_file(self, filepath: str, steps: int = None, batch_size: int = None, 
                       learning_rate: float = None, block_size: int = None, use_gpu: bool | None = None, 
                       use_accelerate: bool | str = 'auto', grad_accum_steps: int = None, 
                       mixed_precision: str = 'auto', weight_decay: float = None, 
                       warmup_steps: int = None, max_grad_norm: float = None):
        """Train or continue training the SAME model (no new models created)"""
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPT training")
        
        print(f"\n{'='*80}")
        print(f"FinAI Training - Single Model Continuous Learning")
        print(f"{'='*80}")
        print(f"Loading dataset from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Load or create tokenizer (must be consistent)
        if os.path.exists(self.config.TOKENIZER_PATH):
            self.tokenizer = Tokenizer.load(self.config.TOKENIZER_PATH)
            print("Loaded existing tokenizer")
        else:
            self.tokenizer = Tokenizer()
            print("Created new tokenizer")
        
        tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        print(f"Tokenized {len(tokens):,} tokens")

        # Use optimized defaults from config
        steps = steps or self.config.TRAIN_STEPS
        batch_size = batch_size or self.config.BATCH_SIZE
        learning_rate = learning_rate or self.config.LEARNING_RATE
        block_size = block_size or self.config.BLOCK_SIZE
        grad_accum_steps = grad_accum_steps or self.config.GRADIENT_ACCUM_STEPS
        weight_decay = weight_decay or self.config.WEIGHT_DECAY
        warmup_steps = warmup_steps or self.config.WARMUP_STEPS
        max_grad_norm = max_grad_norm or self.config.MAX_GRAD_NORM

        # GPU auto-detect
        if use_gpu is None:
            use_gpu = bool(torch.cuda.is_available())

        # Mixed precision: use bf16 for better accuracy
        if mixed_precision == 'auto':
            mixed_precision = 'bf16' if (use_gpu and torch.cuda.is_available()) else 'no'

        # Load existing model or create new one (SAME model always)
        training_state = {}
        if os.path.exists(self.config.LANGUAGE_MODEL_PATH):
            print("\nLoading existing model to continue training...")
            self.model, training_state = GPTModel.load(
                self.config.LANGUAGE_MODEL_PATH, 
                use_gpu=use_gpu,
                use_grad_checkpointing=self.config.USE_GRAD_CHECKPOINTING
            )
        else:
            print("\nCreating new model (first time)...")
            self.model = GPTModel(
                vocab_size=self.tokenizer.vocab_size,
                block_size=self.config.BLOCK_SIZE,
                n_layer=self.config.N_LAYER,
                n_head=self.config.N_HEAD,
                n_embd=self.config.N_EMBD,
                dropout=self.config.DROPOUT,
                use_gpu=use_gpu,
                use_grad_checkpointing=self.config.USE_GRAD_CHECKPOINTING
            )
            training_state = {'total_steps_completed': 0}

        # Send tokens to device
        tokens_tensor = tokens_tensor.to(self.model.device)

        # Accelerate training (recommended for multi-GPU)
        accel_enabled = False
        if use_accelerate is True or use_accelerate == 'auto':
            try:
                import accelerate  # noqa: F401
                accel_enabled = True
            except Exception:
                accel_enabled = False
                if use_accelerate is True:
                    print("Accelerate not installed; using standard training")

        if accel_enabled:
            print("\nUsing Accelerate for optimized training\n")
            try:
                is_main = self.model.train_on_tokens_accelerate(
                    tokens_tensor,
                    steps=steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    gradient_accumulation_steps=grad_accum_steps,
                    mixed_precision=mixed_precision,
                    weight_decay=weight_decay,
                    warmup_steps=warmup_steps,
                    max_grad_norm=max_grad_norm,
                )
                if is_main:
                    self._save_models()
            except Exception as e:
                print(f"Accelerate failed ({e}); falling back to standard training")
                self.model.train_on_tokens(
                    tokens_tensor, 
                    steps=steps, 
                    batch_size=batch_size, 
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    warmup_steps=warmup_steps,
                    grad_accum_steps=grad_accum_steps,
                    max_grad_norm=max_grad_norm
                )
                self._save_models()
        else:
            self.model.train_on_tokens(
                tokens_tensor, 
                steps=steps, 
                batch_size=batch_size, 
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                grad_accum_steps=grad_accum_steps,
                max_grad_norm=max_grad_norm
            )
            self._save_models()

        print("\n" + "="*80)
        print("Training complete - Model saved to:", self.config.LANGUAGE_MODEL_PATH)
        print("="*80 + "\n")

    def _load_models(self) -> bool:
        """Load the single trained model"""
        try:
            self.tokenizer = Tokenizer.load(self.config.TOKENIZER_PATH)
            self.model, _ = GPTModel.load(
                self.config.LANGUAGE_MODEL_PATH, 
                use_gpu=True,
                use_grad_checkpointing=self.config.USE_GRAD_CHECKPOINTING
            )
            return True
        except Exception:
            return False

    def _save_models(self, training_state: dict = None):
        """Save to the SAME model path (no new models created)"""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        self.tokenizer.save(self.config.TOKENIZER_PATH)
        self.model.save(self.config.LANGUAGE_MODEL_PATH, training_state=training_state)
        print(f"Model checkpoint saved (continuing same model)")

    def generate_response(self, user_input: str) -> str:
        """Generate response with improved sampling (top-k + top-p)"""
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
            top_p=self.config.TOP_P,
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
