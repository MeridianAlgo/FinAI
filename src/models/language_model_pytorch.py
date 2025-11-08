"""Modern GPT-style Transformer with RoPE, SwiGLU, Flash Attention for FinAI"""
import math
import os
from typing import Optional, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RoPE(nn.Module):
    """Rotary Positional Embeddings for better long-context extrapolation"""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device, q.dtype)
        return (
            apply_rotary_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_emb(k, self._cos_cached, self._sin_cached),
        )

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor"""
    # Split last dim in half for rotation
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    # Rotate
    return torch.cat((
        x1 * cos[:, :, :x.shape[2], :d//2] - x2 * sin[:, :, :x.shape[2], :d//2],
        x1 * sin[:, :, :x.shape[2], :d//2] + x2 * cos[:, :, :x.shape[2], :d//2]
    ), dim=-1)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional Flash Attention 2"""
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int, use_flash: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # QKV projection in one go for efficiency
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # RoPE for positional encoding
        self.rope = RoPE(self.head_dim, max_seq_len=block_size)
        
        # Flash Attention 2 support
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        # Causal mask (only used if not using flash attention)
        if not self.use_flash:
            mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k)
        
        # Attention
        if self.use_flash:
            # Flash Attention 2: Faster and more memory efficient
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Standard attention with manual masking
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reshape back: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class SwiGLU(nn.Module):
    """SwiGLU activation (gated GLU with Swish): Modern upgrade over GELU for 10% better performance"""
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        hidden_dim = int(8 * n_embd / 3)  # ~2.67x for SwiGLU (compensates for gating)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  # Round to nearest 256 for efficiency
        
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)  # Down projection
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (Swish(W1 * x) ⊙ W3 * x) * W2
        # Where Swish(x) = x * sigmoid(x)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, RoPE attention, and SwiGLU FFN"""
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int, use_flash: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size, use_flash)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = SwiGLU(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (more stable for deep models)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    """Modern GPT-style decoder-only Transformer with RoPE, SwiGLU, Flash Attention"""

    def __init__(self, vocab_size: int, block_size: int = 1024, n_layer: int = 24, n_head: int = 16, 
                 n_embd: int = 1024, dropout: float = 0.05, use_gpu: bool = True, use_flash: bool = True,
                 use_grad_checkpointing: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.use_grad_checkpointing = use_grad_checkpointing
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        
        # Token embeddings (no positional embeddings - using RoPE instead)
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, n_embd),
            'drop': nn.Dropout(dropout),
            'h': nn.ModuleList([TransformerBlock(n_embd, n_head, dropout, block_size, use_flash) 
                               for _ in range(n_layer)]),
            'ln_f': nn.LayerNorm(n_embd),
        })
        
        # LM head (tied with input embeddings for parameter efficiency)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        self.device = self._get_device(use_gpu)
        self.to(self.device)
        self.is_trained = False
        
        # Calculate and print model size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"FinAI Model: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_device(self, use_gpu: bool):
        if not use_gpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device()
        except Exception:
            pass
        return torch.device('cpu')

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        # Token embeddings (no positional - RoPE handles that)
        x = self.transformer['wte'](idx)  # (B, T, n_embd)
        x = self.transformer['drop'](x)
        
        # Transformer blocks with optional gradient checkpointing
        for block in self.transformer['h']:
            if self.use_grad_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.7, 
                 top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
        """Generate text with top-k and top-p (nucleus) sampling"""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative prob > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        
        return idx

    def train_on_tokens(self, tokens: torch.Tensor, steps: int = 10000, batch_size: int = 32, 
                       learning_rate: float = 6e-4, weight_decay: float = 0.1, warmup_steps: int = 100,
                       grad_accum_steps: int = 8, max_grad_norm: float = 1.0, dataset_name: str = None, training_mode: str = 'single'):
        """Train with modern optimization: AdamW, cosine LR schedule, warmup, gradient accumulation"""
        self.train()
        
        # Initialize metrics tracker
        try:
            from src.training_metrics import get_metrics_tracker
            metrics = get_metrics_tracker()
            metrics.start_training(
                dataset_name=dataset_name or 'unknown',
                total_steps=steps,
                training_mode=training_mode,
                batch_size=batch_size,
                block_size=self.block_size,
                device=str(self.device)
            )
        except:
            metrics = None
        
        # AdamW optimizer with proper betas
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        n = tokens.numel()
        if n < self.block_size + 1:
            raise ValueError("Not enough tokens for training")

        # Cosine learning rate schedule with warmup
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps
        )

        # Progress/ETA setup
        import time
        from datetime import timedelta
        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Training Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Steps: {steps} | Batch size: {batch_size} | Block size: {self.block_size}")
        print(f"  Learning rate: {learning_rate} | Warmup steps: {warmup_steps}")
        print(f"  Gradient accumulation: {grad_accum_steps} (effective batch: {batch_size * grad_accum_steps})")
        print(f"  Weight decay: {weight_decay} | Max grad norm: {max_grad_norm}")
        print(f"{'='*80}\n")

        # Use bfloat16 if available for better accuracy
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        optimizer.zero_grad(set_to_none=True)
        step_times = deque(maxlen=100)
        ema_step_time = None  # Exponential moving average
        last_tick = time.time()
        last_step = 0  # Track last reported step
        for step in range(steps):
            # Gradient accumulation
            for micro_step in range(grad_accum_steps):
                ix = torch.randint(0, n - self.block_size - 1, (batch_size,), device=self.device)
                x = torch.stack([tokens[i:i + self.block_size] for i in ix])
                y = torch.stack([tokens[i + 1:i + 1 + self.block_size] for i in ix])

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits, loss = self(x, y)
                        loss = loss / grad_accum_steps  # Scale loss for accumulation
                    scaler.scale(loss).backward()
                else:
                    logits, loss = self(x, y)
                    loss = loss / grad_accum_steps
                    loss.backward()

            # Gradient clipping and optimizer step
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Report progress
            if (step + 1) % max(1, steps // 20) == 0 or step == 0:
                now = time.time()
                time_elapsed_since_last = now - last_tick
                steps_since_last = (step + 1) - last_step
                
                # Calculate per-step time
                if steps_since_last > 0:
                    per_step_time = time_elapsed_since_last / steps_since_last
                    step_times.append(per_step_time)
                    
                    # Exponential moving average for smoother ETA (alpha=0.3)
                    if ema_step_time is None:
                        ema_step_time = per_step_time
                    else:
                        ema_step_time = 0.3 * per_step_time + 0.7 * ema_step_time
                
                last_tick = now
                last_step = step + 1
                
                elapsed = now - start_time
                done = step + 1
                remaining = steps - done
                
                # Use EMA for ETA if we have enough samples, otherwise use simple average
                if len(step_times) >= 5 and ema_step_time is not None:
                    eta_seconds = ema_step_time * remaining
                else:
                    # Not enough data yet, show "calculating..."
                    eta_seconds = None
                
                elapsed_td = timedelta(seconds=int(elapsed))
                current_lr = scheduler.get_last_lr()[0]
                
                if eta_seconds is not None:
                    eta = timedelta(seconds=int(eta_seconds))
                    print(
                        f"Step {done}/{steps} | loss {(loss.item() * grad_accum_steps):.4f} | "
                        f"lr {current_lr:.2e} | elapsed {elapsed_td} | ETA {eta}"
                    )
                else:
                    print(
                        f"Step {done}/{steps} | loss {(loss.item() * grad_accum_steps):.4f} | "
                        f"lr {current_lr:.2e} | elapsed {elapsed_td} | ETA calculating..."
                    )
                
                # Update metrics tracker
                if metrics:
                    try:
                        metrics.update_step(
                            step=done,
                            loss=(loss.item() * grad_accum_steps),
                            learning_rate=current_lr,
                            step_time=step_time,
                            eta_seconds=eta_seconds
                        )
                    except:
                        pass

                # Periodic checkpoint save to unified model path
                try:
                    from src.config import Config
                    ckpt_path = Config().LANGUAGE_MODEL_PATH
                    self.save(ckpt_path, training_state={'total_steps_completed': step + 1})
                except Exception as _:
                    pass
        
        self.is_trained = True
        
        # Mark training complete
        if metrics:
            try:
                metrics.end_training(success=True)
            except:
                pass
        
        print(f"\n{'='*80}")
        print(f"Training completed in {timedelta(seconds=int(time.time() - start_time))}")
        print(f"{'='*80}\n")

    def train_on_tokens_accelerate(
        self,
        tokens: torch.Tensor,
        steps: int = 10000,
        batch_size: int = 32,
        learning_rate: float = 6e-4,
        gradient_accumulation_steps: int = 8,
        mixed_precision: str = 'bf16',
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        dataset_name: str = None,
        training_mode: str = 'single',
    ) -> bool:
        """Train using HF Accelerate with modern optimization. Returns True if main process."""
        try:
            from accelerate import Accelerator
            from transformers import get_cosine_schedule_with_warmup
        except Exception as e:
            raise RuntimeError("Install required packages: pip install accelerate transformers") from e

        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=None
        )
        
        self.train()
        
        # Initialize metrics tracker (main process only)
        metrics = None
        if accelerator.is_main_process:
            try:
                from src.training_metrics import get_metrics_tracker
                metrics = get_metrics_tracker()
                metrics.start_training(
                    dataset_name=dataset_name or 'unknown',
                    total_steps=steps,
                    training_mode=training_mode,
                    batch_size=batch_size,
                    block_size=self.block_size,
                    device=str(accelerator.device)
                )
            except:
                pass
        
        # AdamW optimizer with proper settings
        optimizer = torch.optim.AdamW(
            self.parameters(),
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

        # Prepare model, optimizer, and scheduler
        model, optimizer, scheduler = accelerator.prepare(self, optimizer, scheduler)

        n = tokens.numel()
        if n < self.block_size + 1:
            raise ValueError("Not enough tokens for training")

        # Move tokens to correct device
        tokens = tokens.to(accelerator.device)

        # Progress/ETA setup (main process only)
        import time
        from datetime import timedelta
        start_time = time.time()
        if accelerator.is_main_process:
            print(f"[Accelerate] Training on device={accelerator.device} | steps={steps}, batch_size={batch_size}, block_size={self.block_size} | mp={mixed_precision} | accum={gradient_accumulation_steps}")
        # Define progress trackers (were missing before)
        from collections import deque as _deque
        step_times = _deque(maxlen=100)
        ema_step_time = None  # Exponential moving average
        last_tick = time.time()
        last_step = 0  # Track last reported step

        for step in range(steps):
            with accelerator.accumulate(model):
                ix = torch.randint(0, n - self.block_size - 1, (batch_size,), device=accelerator.device)
                x = torch.stack([tokens[i:i + self.block_size] for i in ix])
                y = torch.stack([tokens[i + 1:i + 1 + self.block_size] for i in ix])

                with accelerator.autocast():
                    logits, loss = model(x, y)
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % max(1, steps // 20) == 0 or step == 0:
                if accelerator.is_main_process:
                    now = time.time()
                    time_elapsed_since_last = now - last_tick
                    steps_since_last = (step + 1) - last_step
                    
                    # Calculate per-step time
                    if steps_since_last > 0:
                        per_step_time = time_elapsed_since_last / steps_since_last
                        step_times.append(per_step_time)
                        
                        # Exponential moving average for smoother ETA (alpha=0.3)
                        if ema_step_time is None:
                            ema_step_time = per_step_time
                        else:
                            ema_step_time = 0.3 * per_step_time + 0.7 * ema_step_time
                    
                    last_tick = now
                    last_step = step + 1
                    
                    elapsed = now - start_time
                    done = step + 1
                    remaining = steps - done
                    
                    # Use EMA for ETA if we have enough samples
                    if len(step_times) >= 5 and ema_step_time is not None:
                        eta_seconds = ema_step_time * remaining
                    else:
                        eta_seconds = None
                    
                    elapsed_td = timedelta(seconds=int(elapsed))
                    current_lr = scheduler.get_last_lr()[0]
                    
                    if eta_seconds is not None:
                        eta = timedelta(seconds=int(eta_seconds))
                        print(f"Step {done}/{steps} | loss {loss.item():.4f} | lr {current_lr:.2e} | elapsed {elapsed_td} | ETA {eta}")
                    else:
                        print(f"Step {done}/{steps} | loss {loss.item():.4f} | lr {current_lr:.2e} | elapsed {elapsed_td} | ETA calculating...")
                    
                    # Update metrics tracker
                    if metrics:
                        try:
                            metrics.update_step(
                                step=done,
                                loss=loss.item(),
                                learning_rate=current_lr,
                                step_time=step_time,
                                eta_seconds=eta_seconds
                            )
                        except:
                            pass

                    # Periodic checkpoint save to unified model path (main process only)
                    try:
                        from src.config import Config
                        ckpt_path = Config().LANGUAGE_MODEL_PATH
                        model.save(ckpt_path, training_state={'total_steps_completed': step + 1})
                    except Exception as _:
                        pass

        self.is_trained = True
        
        # Mark training complete
        if metrics:
            try:
                metrics.end_training(success=True)
            except:
                pass
        
        return accelerator.is_main_process

    def save(self, path: str, training_state: dict = None):
        """Save model checkpoint - ALWAYS saves to same path to maintain single model"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'dropout': self.dropout,
            'is_trained': self.is_trained,
        }
        
        # Add training state if provided (for resumption)
        if training_state:
            checkpoint['training_state'] = training_state
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path: str, use_gpu: bool = True, use_grad_checkpointing: bool = False) -> tuple:
        """Load model checkpoint - continues training the SAME model
        Returns: (model, training_state_dict)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location='cpu')
        
        # Recreate model with same architecture
        from src.config import Config
        cfg = Config()
        
        # Infer missing architecture fields from checkpoint when possible
        state_dict = ckpt.get('model_state_dict', {})
        # Vocab size
        vocab_size = ckpt.get('vocab_size')
        if vocab_size is None:
            w = state_dict.get('lm_head.weight')
            if w is None:
                w = state_dict.get('transformer.wte.weight')
            if w is not None:
                vocab_size = w.shape[0]
        if vocab_size is None:
            vocab_size = 50257

        # n_embd
        n_embd = ckpt.get('n_embd')
        if n_embd is None:
            w = state_dict.get('lm_head.weight')
            if w is None:
                w = state_dict.get('transformer.wte.weight')
            if w is not None and w.dim() >= 2:
                n_embd = w.shape[1]
        if n_embd is None:
            n_embd = cfg.N_EMBD

        # Other fields
        block_size = ckpt.get('block_size', cfg.BLOCK_SIZE)
        n_layer = ckpt.get('n_layer', cfg.N_LAYER)
        n_head = ckpt.get('n_head', cfg.N_HEAD)
        dropout = ckpt.get('dropout', cfg.DROPOUT)

        model = LanguageModel(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            use_gpu=use_gpu,
            use_grad_checkpointing=use_grad_checkpointing
        )
        # Filter checkpoint to only keys with matching shapes to avoid size mismatch errors
        model_sd = model.state_dict()
        compatible = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
                compatible[k] = v
            else:
                skipped.append(k)
        if skipped:
            print(f"  Info: skipping {len(skipped)} parameter(s) due to shape mismatch (e.g., {skipped[:3]})")
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if missing or unexpected:
            print(f"  Info: loaded with partial weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
        model.is_trained = ckpt.get('is_trained', True)
        
        # Extract training state if available
        training_state = ckpt.get('training_state', {})
        
        print(f"Model loaded from {path} (continuing training on same model)")
        if training_state:
            total_steps = training_state.get('total_steps_completed', 0)
            print(f"  Previous training: {total_steps:,} steps completed")
        
        return model, training_state

