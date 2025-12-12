"""Main FinAI application (local GPT-style LLM)"""
import os
import torch
from datetime import timedelta
from src.config import Config
from src.data.tokenizer import Tokenizer
from src.core.context import ConversationContext

try:
    from src.models.efficient_model import EfficientFinAI as GPTModel, create_model
    PYTORCH_AVAILABLE = True
except Exception:
    PYTORCH_AVAILABLE = False
    GPTModel = None
    create_model = None


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
                       warmup_steps: int = None, max_grad_norm: float = None, dataset_name: str = None, training_mode: str = 'single',
                       one_epoch: bool = False):
        """Train or continue training the SAME model (no new models created)
        
        Args:
            one_epoch: If True, automatically calculate steps for one full epoch based on dataset size
        """
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
        num_tokens = len(tokens)
        print(f"Tokenized {num_tokens:,} tokens")

        # Use optimized defaults from config
        batch_size = batch_size or self.config.BATCH_SIZE
        learning_rate = learning_rate or self.config.LEARNING_RATE
        block_size = block_size or self.config.BLOCK_SIZE
        grad_accum_steps = grad_accum_steps or self.config.GRADIENT_ACCUM_STEPS
        weight_decay = weight_decay or self.config.WEIGHT_DECAY
        warmup_steps = warmup_steps or self.config.WARMUP_STEPS
        max_grad_norm = max_grad_norm or self.config.MAX_GRAD_NORM
        
        # Calculate steps for one epoch if requested
        if one_epoch:
            # One epoch = process all possible sequences in the dataset
            # Each step processes batch_size * grad_accum_steps sequences
            # Each sequence is block_size tokens
            # Total possible sequences = num_tokens - block_size
            total_sequences = max(1, num_tokens - block_size)
            sequences_per_step = batch_size * grad_accum_steps
            steps = max(1, total_sequences // sequences_per_step)
            print(f"\n{'='*80}")
            print(f"ONE EPOCH MODE ENABLED")
            print(f"  Total tokens: {num_tokens:,}")
            print(f"  Block size: {block_size}")
            print(f"  Total sequences: {total_sequences:,}")
            print(f"  Batch size: {batch_size} | Gradient accumulation: {grad_accum_steps}")
            print(f"  Sequences per step: {sequences_per_step}")
            print(f"  Steps for one epoch: {steps:,}")
            print(f"{'='*80}\n")
            
            # Estimate training time based on hardware
            # Rough estimates: GPU ~0.1-0.5s/step, CPU ~1-3s/step
            if use_gpu is None:
                use_gpu = bool(torch.cuda.is_available())
            
            if use_gpu and torch.cuda.is_available():
                estimated_seconds_per_step = 0.3  # GPU estimate
                hardware = "GPU"
            else:
                estimated_seconds_per_step = 2.0  # CPU estimate (conservative)
                hardware = "CPU"
            
            estimated_total_seconds = steps * estimated_seconds_per_step
            estimated_time = timedelta(seconds=int(estimated_total_seconds))
            
            # Format time nicely
            hours = estimated_total_seconds // 3600
            minutes = (estimated_total_seconds % 3600) // 60
            if hours > 0:
                time_str = f"{int(hours)}h {int(minutes)}m"
            elif minutes > 0:
                time_str = f"{int(minutes)}m {int(estimated_total_seconds % 60)}s"
            else:
                time_str = f"{int(estimated_total_seconds)}s"
            
            print(f"Estimated training time ({hardware}): ~{time_str}")
            if hardware == "CPU":
                print(f"  Note: GPU training would be ~5-10x faster if available\n")
            else:
                print()
        else:
            steps = steps or self.config.TRAIN_STEPS

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
            try:
                checkpoint = torch.load(self.config.LANGUAGE_MODEL_PATH, map_location='cpu')
                self.model = GPTModel(
                    vocab_size=checkpoint.get('vocab_size', self.tokenizer.vocab_size),
                    n_embd=checkpoint.get('n_embd', self.config.N_EMBD),
                    n_head=checkpoint.get('n_head', self.config.N_HEAD),
                    n_layer=checkpoint.get('n_layer', self.config.N_LAYER),
                    block_size=checkpoint.get('block_size', self.config.BLOCK_SIZE),
                    dropout=checkpoint.get('dropout', self.config.DROPOUT)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                training_state = checkpoint.get('training_state', {'total_steps_completed': 0})
                print(f"  Loaded {self.model.get_num_params()/1e6:.2f}M parameter model")
            except Exception as e:
                print(f"  Failed to load existing model: {e}")
                print("  Creating new model instead...")
                self.model = GPTModel(
                    vocab_size=self.tokenizer.vocab_size,
                    n_embd=self.config.N_EMBD,
                    n_head=self.config.N_HEAD,
                    n_layer=self.config.N_LAYER,
                    block_size=self.config.BLOCK_SIZE,
                    dropout=self.config.DROPOUT
                )
                training_state = {'total_steps_completed': 0}
        else:
            print("\nCreating new EfficientFinAI model (first time)...")
            self.model = GPTModel(
                vocab_size=self.tokenizer.vocab_size,
                n_embd=self.config.N_EMBD,
                n_head=self.config.N_HEAD,
                n_layer=self.config.N_LAYER,
                block_size=self.config.BLOCK_SIZE,
                dropout=self.config.DROPOUT
            )
            training_state = {'total_steps_completed': 0}

        # Determine device and move model
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Send tokens to device
        try:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            tokens_tensor = tokens_tensor.to(device)
        except RuntimeError as e:
            print(f"[FinAI] GPU error: {e}, falling back to CPU...")
            device = torch.device('cpu')
            self.model = self.model.to(device)
            tokens_tensor = tokens_tensor.to(device)

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

        # Train using standard PyTorch training loop
        self._train_model(
            tokens_tensor,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            device=device
        )
        self._save_models(training_state)

        print("\n" + "="*80)
        print("Training complete - Model saved to:", self.config.LANGUAGE_MODEL_PATH)
        print("="*80 + "\n")

    def _train_model(self, tokens_tensor, steps, batch_size, learning_rate, weight_decay, 
                     warmup_steps, grad_accum_steps, max_grad_norm, device):
        """Train the model with modern optimization"""
        from transformers import get_cosine_schedule_with_warmup
        import time
        from datetime import timedelta
        from collections import deque
        
        self.model.train()
        
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        # Cosine scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps
        )
        
        n = tokens_tensor.numel()
        if n < self.model.block_size + 1:
            raise ValueError("Not enough tokens for training")
        
        print(f"\n{'='*80}")
        print(f"Training Configuration:")
        print(f"  Device: {device}")
        print(f"  Steps: {steps} | Batch: {batch_size} | Block: {self.model.block_size}")
        print(f"  LR: {learning_rate} | Warmup: {warmup_steps}")
        print(f"  Grad accum: {grad_accum_steps} (effective batch: {batch_size * grad_accum_steps})")
        print(f"  Weight decay: {weight_decay} | Max grad norm: {max_grad_norm}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        step_times = deque(maxlen=100)
        ema_step_time = None
        last_tick = time.time()
        last_step = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        for step in range(steps):
            # Gradient accumulation
            for micro_step in range(grad_accum_steps):
                ix = torch.randint(0, n - self.model.block_size - 1, (batch_size,), device=device)
                x = torch.stack([tokens_tensor[i:i + self.model.block_size] for i in ix])
                y = torch.stack([tokens_tensor[i + 1:i + 1 + self.model.block_size] for i in ix])
                
                logits, loss = self.model(x, y)
                loss = loss / grad_accum_steps
                loss.backward()
            
            # Gradient clipping and optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Progress reporting
            if (step + 1) % max(1, steps // 20) == 0 or step == 0:
                now = time.time()
                time_elapsed = now - last_tick
                steps_since_last = (step + 1) - last_step
                
                if steps_since_last > 0:
                    per_step_time = time_elapsed / steps_since_last
                    step_times.append(per_step_time)
                    
                    if ema_step_time is None:
                        ema_step_time = per_step_time
                    else:
                        ema_step_time = 0.3 * per_step_time + 0.7 * ema_step_time
                
                last_tick = now
                last_step = step + 1
                
                elapsed = now - start_time
                remaining = steps - (step + 1)
                
                if len(step_times) >= 5 and ema_step_time is not None:
                    eta_seconds = ema_step_time * remaining
                    eta = timedelta(seconds=int(eta_seconds))
                    eta_str = f"ETA {eta}"
                else:
                    eta_str = "ETA calculating..."
                
                elapsed_td = timedelta(seconds=int(elapsed))
                current_lr = scheduler.get_last_lr()[0]
                
                print(f"Step {step+1}/{steps} | loss {(loss.item() * grad_accum_steps):.4f} | "
                      f"lr {current_lr:.2e} | elapsed {elapsed_td} | {eta_str}")
        
        print(f"\n{'='*80}")
        print(f"Training completed in {timedelta(seconds=int(time.time() - start_time))}")
        print(f"{'='*80}\n")

    def _load_models(self) -> bool:
        """Load the single trained model"""
        try:
            self.tokenizer = Tokenizer.load(self.config.TOKENIZER_PATH)
            checkpoint = torch.load(self.config.LANGUAGE_MODEL_PATH, map_location='cpu')
            self.model = GPTModel(
                vocab_size=checkpoint.get('vocab_size', self.tokenizer.vocab_size),
                n_embd=checkpoint.get('n_embd', self.config.N_EMBD),
                n_head=checkpoint.get('n_head', self.config.N_HEAD),
                n_layer=checkpoint.get('n_layer', self.config.N_LAYER),
                block_size=checkpoint.get('block_size', self.config.BLOCK_SIZE),
                dropout=checkpoint.get('dropout', self.config.DROPOUT)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def _save_models(self, training_state: dict = None):
        """Save to the SAME model path (no new models created)"""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        self.tokenizer.save(self.config.TOKENIZER_PATH)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.model.vocab_size,
            'n_embd': self.model.n_embd,
            'n_head': self.model.n_head,
            'n_layer': self.model.n_layer,
            'block_size': self.model.block_size,
            'dropout': self.model.dropout,
            'training_state': training_state or {}
        }
        
        torch.save(checkpoint, self.config.LANGUAGE_MODEL_PATH)
        print(f"Model checkpoint saved (continuing same model)")

    def generate_response(self, user_input: str) -> str:
        """Generate response with improved sampling (top-k + top-p)"""
        if not self.model or not self.tokenizer:
            return "Model not loaded. Train first."
        
        self.model.eval()
        
        # Build prompt with short context
        context_str = self.context.get_context_string(num_messages=3)
        if context_str:
            prompt = f"{context_str}\nuser: {user_input}\nassistant:"
        else:
            prompt = f"user: {user_input}\nassistant:"
        
        # Encode and generate
        x = self.tokenizer.encode(prompt)
        device = next(self.model.parameters()).device
        x = torch.tensor([x], dtype=torch.long, device=device)
        
        with torch.no_grad():
            y = self.model.generate(
                x,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_k=self.config.TOP_K
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
