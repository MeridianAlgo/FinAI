"""Meridian.AI Trainer — CPU-optimized continual training engine.

Key features:
 - Memory-aware adaptive batch sizing (auto-detects available RAM)
 - Cosine annealing with warm restarts (perfect for hourly runs)
 - Gradient clipping + gradient accumulation
 - Comet ML experiment tracking
 - EWC continual learning integration
 - Checkpoint save/resume with full optimizer state
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import psutil
import torch
import torch.nn as nn

try:
    from comet_ml import Experiment

    HAS_COMET = True
except ImportError:
    HAS_COMET = False

from meridian.training.ewc import ElasticWeightConsolidation


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Core
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 200
    total_steps: int = 100_000

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.06

    # Scheduler: Cosine with warm restarts
    # Each hourly run is a "restart" — LR warms up then decays
    min_lr_ratio: float = 0.1

    # Saving
    output_dir: str = "./checkpoint"
    save_steps: int = 50
    log_steps: int = 5

    # EWC
    use_ewc: bool = True
    ewc_lambda: float = 100.0
    ewc_samples: int = 20

    # Experiment tracking
    project_name: str = "meridian-ai"
    experiment_name: str = "hourly-train"


class MeridianTrainer:
    """CPU-optimized training engine for Meridian.AI."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.dataloader = dataloader
        self.config = config

        # Mandatory for 16GB runners: reduce activation memory.
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
                if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
                print("[INFO] Gradient checkpointing enabled")
            except Exception as e:
                print(f"[WARN] Failed to enable gradient checkpointing: {e}")

        # Optimizer
        # NOTE: AdamW is memory-heavy on CPU for large models (m/v states ~= 2x params).
        # For 16GB GitHub runners, allow switching to Adafactor to reduce optimizer-state RAM.
        optimizer_name = os.getenv("OPTIMIZER", "adamw").strip().lower()
        if optimizer_name == "adafactor":
            from transformers.optimization import Adafactor

            self.optimizer = Adafactor(
                self.model.parameters(),
                lr=config.learning_rate,
                relative_step=False,
                scale_parameter=True,
                warmup_init=False,
                weight_decay=config.weight_decay,
                clip_threshold=1.0,
            )
            print("[INFO] Optimizer: Adafactor (low-memory)")
        else:
            # Optimizer: AdamW with decoupled weight decay
            param_groups = self._get_param_groups()
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(0.9, 0.95),  # Llama-3 style betas
                weight_decay=config.weight_decay,
                fused=False,  # CPU doesn't support fused
            )
            print("[INFO] Optimizer: AdamW")

        # State
        self.global_step = 0
        self.run_step = 0
        self.best_loss = float("inf")

        # EWC for continual learning
        self.ewc: Optional[ElasticWeightConsolidation] = None
        if config.use_ewc:
            self.ewc = ElasticWeightConsolidation(model, config.ewc_lambda)
            ewc_path = os.path.join(config.output_dir, "ewc_state.pt")
            if os.path.exists(ewc_path):
                if self.ewc.load(ewc_path):
                    print("[OK] Loaded EWC state from previous run")
                else:
                    print("[WARN] EWC state incompatible with current model — deleting stale file")
                    os.remove(ewc_path)

        # Comet ML
        self.experiment: Optional[Experiment] = None
        if HAS_COMET and os.getenv("COMET_API_KEY"):
            try:
                self.experiment = Experiment(
                    api_key=os.getenv("COMET_API_KEY"),
                    project_name=config.project_name,
                    auto_metric_logging=False,
                )
                self.experiment.set_name(config.experiment_name)
                self.experiment.log_parameters(
                    {
                        "batch_size": config.batch_size,
                        "grad_accum": config.gradient_accumulation_steps,
                        "max_steps": config.max_steps,
                        "lr": config.learning_rate,
                        "ewc": config.use_ewc,
                    }
                )
            except Exception as e:
                print(f"[WARN] Comet ML init failed: {e}")

    def _get_param_groups(self) -> list:
        """Separate parameters for weight decay (skip biases & norms)."""
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def _get_lr(self, step: int) -> float:
        """Cosine annealing with warm-up for this training run."""
        warmup_steps = int(self.config.max_steps * self.config.warmup_ratio)
        min_lr = self.config.learning_rate * self.config.min_lr_ratio

        if step < warmup_steps:
            return self.config.learning_rate * (step + 1) / (warmup_steps + 1)

        progress = (step - warmup_steps) / max(1, self.config.max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (self.config.learning_rate - min_lr) * cosine

    def _update_lr(self, step: int) -> float:
        lr = self._get_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _log_memory(self) -> None:
        mem = psutil.virtual_memory()
        print(f"  Memory: {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB ({mem.percent}% used)")

    def _memory_guard(self) -> bool:
        """Return True if training should stop to avoid OOM."""
        if os.getenv("HARD_RAM_GUARD", "0") != "1":
            return False
        max_gb = float(os.getenv("MAX_RAM_GB", "0"))
        max_pct = float(os.getenv("MAX_RAM_PCT", "0"))
        if max_gb <= 0 and max_pct <= 0:
            return False

        mem = psutil.virtual_memory()
        used_gb = mem.used / 1e9
        if (max_gb > 0 and used_gb >= max_gb) or (max_pct > 0 and mem.percent >= max_pct):
            print(
                f"\n[GUARD] Memory limit reached ({used_gb:.1f}GB, {mem.percent:.1f}%). "
                "Saving weights-only checkpoint and stopping early."
            )
            return True
        return False

    def _soft_cap_should_throttle(self) -> bool:
        soft_gb = float(os.getenv("SOFT_RAM_GB", "0"))
        soft_pct = float(os.getenv("SOFT_RAM_PCT", "0"))
        if soft_gb <= 0 and soft_pct <= 0:
            return False
        mem = psutil.virtual_memory()
        used_gb = mem.used / 1e9
        return (soft_gb > 0 and used_gb >= soft_gb) or (soft_pct > 0 and mem.percent >= soft_pct)

    def _truncate_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        new_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if new_len <= 0:
            return input_ids, labels, attention_mask
        if input_ids.dim() == 2 and input_ids.size(1) > new_len:
            input_ids = input_ids[:, -new_len:]
            labels = labels[:, -new_len:]
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask[:, -new_len:]
        return input_ids, labels, attention_mask

    def train(self) -> None:
        """Execute training loop."""
        debug = os.getenv("DEBUG_STEPS", "0") == "1"

        def dprint(msg: str) -> None:
            if debug:
                print(msg)

        print(f"\n{'=' * 70}")
        print("  MERIDIAN.AI TRAINING ENGINE")
        print(
            f"  Steps: {self.config.max_steps} | BS: {self.config.batch_size} "
            f"| Accum: {self.config.gradient_accumulation_steps}"
        )
        print(f"  LR: {self.config.learning_rate} | Global step: {self.global_step}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,} total | {trainable_params:,} trainable")
        self._log_memory()
        print(f"{'=' * 70}\n")

        self.model.train()
        accumulated_loss = 0.0
        accumulated_ewc_loss = 0.0
        tokens_processed = 0
        start_time = time.time()

        initial_loss_val = None
        final_loss_val = None

        data_iter = iter(self.dataloader)
        first_batch_logged = False

        import gc

        try:
            for micro_step in range(
                self.config.max_steps * self.config.gradient_accumulation_steps
            ):
                if self._memory_guard():
                    self.save_checkpoint(self.config.output_dir, skip_optimizer=True)
                    if self.experiment:
                        self.experiment.end()
                    return
                # Get batch
                try:
                    dprint(f"  [DEBUG] Micro-step {micro_step}: Fetching batch...")
                    batch = next(data_iter)
                    dprint(f"  [DEBUG] Micro-step {micro_step}: Batch received.")
                except StopIteration:
                    print("[INFO] Dataset exhausted. Ending training.")
                    break

                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                labels = batch.get("labels")
                if labels is None:
                    labels = input_ids

                if isinstance(input_ids, list):
                    input_ids = torch.stack(input_ids)
                if isinstance(labels, list):
                    labels = torch.stack(labels)
                if attention_mask is not None and isinstance(attention_mask, list):
                    attention_mask = torch.stack(attention_mask)

                # Hard-guard: check again after batch materialization (optional)
                if self._memory_guard():
                    self.save_checkpoint(self.config.output_dir, skip_optimizer=True)
                    if self.experiment:
                        self.experiment.end()
                    return

                # Soft RAM cap: dynamically reduce sequence length instead of stopping the run.
                # This reduces activation memory and keeps the process alive on 16GB runners.
                if self._soft_cap_should_throttle():
                    original_len = int(input_ids.shape[1]) if input_ids.dim() == 2 else -1
                    min_len = int(os.getenv("MIN_THROTTLE_SEQ_LEN", "128"))
                    new_len = (
                        max(min_len, int(original_len * 0.75)) if original_len > 0 else original_len
                    )
                    if original_len > 0 and new_len < original_len:
                        print(
                            f"[THROTTLE] RAM high. Truncating seq_len {original_len} -> {new_len} (micro_step={micro_step})"
                        )
                        input_ids, labels, attention_mask = self._truncate_batch(
                            input_ids, labels, attention_mask, new_len
                        )
                        gc.collect()

                # Forward pass
                try:
                    # Hard-guard: check right before forward (optional)
                    if self._memory_guard():
                        self.save_checkpoint(self.config.output_dir, skip_optimizer=True)
                        if self.experiment:
                            self.experiment.end()
                        return

                    dprint(f"  [DEBUG] Micro-step {micro_step}: Starting forward pass...")
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = outputs.loss
                    dprint(
                        f"  [DEBUG] Micro-step {micro_step}: Forward pass complete. Loss: {loss.item():.4f}"
                    )
                except Exception as e:
                    print(f"[ERROR] ERROR during forward pass: {e}")
                    continue

                if loss is None:
                    continue

                if not first_batch_logged:
                    initial_loss_val = loss.item()
                    print(f"\n  [CASCADE CHECK] Initial Loss of this run: {initial_loss_val:.4f}")
                    first_batch_logged = True

                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: G] Step {self.run_step}: Loss is NaN. Skipping batch.")
                    self.optimizer.zero_grad()
                    continue

                # Scale for gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps

                try:
                    dprint("  [DEBUG] Starting main backward pass...")
                    scaled_loss.backward()
                    dprint("  [DEBUG] Main backward pass complete.")
                except Exception as e:
                    print(f"[ERROR] ERROR during backward pass: {e}")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                finally:
                    # Free computation graph & intermediate tensors immediately
                    del scaled_loss, outputs

                # Apply EWC gradients manually (extreme memory optimization)
                if self.ewc is not None and self.ewc._initialized:
                    try:
                        dprint("  [DEBUG] Applying manual EWC gradients...")
                        # Scale EWC grad by accumulation steps to match main grad
                        self.ewc.apply_gradients(
                            self.model, scale=1.0 / self.config.gradient_accumulation_steps
                        )
                        dprint("  [DEBUG] Manual EWC gradients applied.")
                    except Exception as e:
                        print(f"[ERROR] Failed to apply manual EWC gradients: {e}")

                accumulated_loss += loss.item()
                del loss
                tokens_processed += input_ids.numel()
                del input_ids, labels
                if attention_mask is not None:
                    del attention_mask

                # Optimizer step
                if (micro_step + 1) % self.config.gradient_accumulation_steps == 0:
                    dprint(f"  [DEBUG] Step {self.run_step}: Starting optimization step...")
                    # Check for NaN gradients before clipping
                    has_nan_grads = False
                    for p in self.model.parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any():
                                has_nan_grads = True
                                break

                    if has_nan_grads:
                        print(
                            f"Warning: G] Step {self.run_step}: NaN gradients detected. Skipping step."
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        accumulated_loss = 0.0
                        accumulated_ewc_loss = 0.0
                        continue

                    # Compute EWC penalty once per optimizer step (not per micro-step)
                    current_ewc_loss = 0.0
                    if self.ewc is not None and self.ewc._initialized:
                        try:
                            dprint("  [DEBUG] Computing EWC penalty (no_grad)...")
                            current_ewc_loss = self.ewc.penalty_value(self.model)
                        except Exception as e:
                            print(f"[WARN] EWC penalty calculation failed: {e}")
                    accumulated_ewc_loss += current_ewc_loss

                    # Gradient clipping
                    dprint(f"  [DEBUG] Step {self.run_step}: Clipping gradients...")
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    dprint(f"  [DEBUG] Step {self.run_step}: Grad norm: {grad_norm:.3f}")

                    # Update LR
                    lr = self._update_lr(self.run_step)

                    dprint(f"  [DEBUG] Step {self.run_step}: Optimizer step...")
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    dprint(f"  [DEBUG] Step {self.run_step}: Optimization complete.")

                    # GC can be very slow on CPU; make it configurable.
                    gc_every = int(os.getenv("GC_EVERY_STEPS", "5"))
                    if gc_every > 0 and (self.run_step % gc_every == 0):
                        gc.collect()

                    self.global_step += 1
                    self.run_step += 1
                    avg_loss = (
                        accumulated_loss + accumulated_ewc_loss
                    ) / self.config.gradient_accumulation_steps
                    final_loss_val = avg_loss

                    # Logging
                    if self.run_step % self.config.log_steps == 0:
                        self._log_memory()
                        elapsed = time.time() - start_time
                        tps = tokens_processed / elapsed if elapsed > 0 else 0
                        print(
                            f"  Step {self.run_step:>5}/{self.config.max_steps} "
                            f"| Global {self.global_step:>6} "
                            f"| Loss: {avg_loss:.4f} "
                            f"| LR: {lr:.2e} "
                            f"| Grad: {grad_norm:.3f} "
                            f"| {tps:.0f} tok/s"
                        )

                        if self.experiment:
                            self.experiment.log_metrics(
                                {
                                    "loss": avg_loss,
                                    "lr": lr,
                                    "grad_norm": (
                                        grad_norm.item()
                                        if isinstance(grad_norm, torch.Tensor)
                                        else grad_norm
                                    ),
                                    "tokens_per_sec": tps,
                                    "global_step": self.global_step,
                                },
                                step=self.global_step,
                            )

                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss

                    accumulated_loss = 0.0
                    accumulated_ewc_loss = 0.0

                    # Save checkpoint
                    if self.run_step % self.config.save_steps == 0:
                        skip_opt = os.getenv("SKIP_OPTIMIZER_SAVE", "0") == "1"
                        self.save_checkpoint(self.config.output_dir, skip_optimizer=skip_opt)

                    if self._memory_guard():
                        self.save_checkpoint(self.config.output_dir, skip_optimizer=True)
                        if self.experiment:
                            self.experiment.end()
                        return

                    if self.run_step >= self.config.max_steps:
                        break
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Training stopped by user. Saving progress...")
            skip_opt = os.getenv("SKIP_OPTIMIZER_SAVE", "0") == "1"
            self.save_checkpoint(self.config.output_dir, skip_optimizer=skip_opt)
            if self.experiment:
                self.experiment.end()
            return

        # Compute Fisher for EWC (for next run)
        if self.ewc is not None and os.getenv("SKIP_FISHER", "0") != "1":
            import gc

            # Explicitly delete lingering loop variables to free up large gradient graphs
            try:
                del data_iter, batch, input_ids, attention_mask, labels, outputs, loss, scaled_loss
            except NameError:
                pass

            # Free up memory before heavy Fisher computation
            self.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Optional: free optimizer memory before Fisher compute on low-RAM machines
            if os.getenv("FREE_OPTIMIZER_BEFORE_FISHER", "1") == "1":
                try:
                    del self.optimizer
                except Exception:
                    pass
                gc.collect()

            print("\n[EWC] Computing Fisher Information Matrix for next run...")
            self._log_memory()
            self.ewc.compute_fisher(self.model, self.dataloader, self.config.ewc_samples)
            ewc_path = os.path.join(self.config.output_dir, "ewc_state.pt")
            os.makedirs(self.config.output_dir, exist_ok=True)
            self.ewc.save(ewc_path)
            print("[OK] Fisher matrix saved for continual learning")
            self._log_memory()

        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print("  TRAINING COMPLETE")
        print(f"  Steps: {self.run_step} | Time: {elapsed:.0f}s | Best loss: {self.best_loss:.4f}")
        if initial_loss_val is not None:
            final_print_val = final_loss_val if final_loss_val is not None else initial_loss_val
            diff = final_print_val - initial_loss_val
            print(
                f"  Initial loss: {initial_loss_val:.4f} | Final loss: {final_print_val:.4f} | Diff: {diff:+.4f}"
            )
        print(f"  Tokens processed: {tokens_processed:,}")
        print(f"{'=' * 70}\n")

        if self.experiment:
            self.experiment.end()

    def save_checkpoint(self, path: str, skip_optimizer: bool = False) -> None:
        """Save model + optimizer + trainer state."""
        os.makedirs(path, exist_ok=True)

        # Save model
        # Default: HF save_pretrained (used on Actions/Linux)
        # Local Windows can hit mmap locking issues; allow forcing a torch.save bin path.
        if os.getenv("FORCE_BIN_SAVE", "0") == "1":
            if hasattr(self.model, "config"):
                self.model.config.save_pretrained(path)
            torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        else:
            # Disabling safe_serialization to avoid mmap lock issues on Windows
            self.model.save_pretrained(path, safe_serialization=False)

        # Save trainer state
        # The optimizer state is 2GB+, so we allow skipping it for fast testing
        trainer_state = {
            "global_step": self.global_step,
            "run_step": self.run_step,
            "best_loss": self.best_loss,
        }

        if not skip_optimizer:
            print(f"  [SAVE] Checkpoint (including 2GB+ optimizer) -> {path}")
            trainer_state["optimizer_state_dict"] = self.optimizer.state_dict()
        else:
            print(f"  [SAVE] Checkpoint (weights only, skipping 2GB optimizer) -> {path}")

        torch.save(trainer_state, os.path.join(path, "trainer_state.pt"))

    def load_checkpoint(self, path: str) -> bool:
        """Load optimizer + trainer state from checkpoint."""
        state_path = os.path.join(path, "trainer_state.pt")
        if not os.path.exists(state_path):
            return False

        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.global_step = state.get("global_step", 0)
            self.run_step = 0  # Reset run step for new run
            self.best_loss = state.get("best_loss", float("inf"))

            if "optimizer_state_dict" in state:
                try:
                    self.optimizer.load_state_dict(state["optimizer_state_dict"])
                    print(f"  [OK] Optimizer state restored (global step {self.global_step})")
                except Exception as e:
                    print(f"  [WARN] Optimizer state mismatch, reinitializing: {e}")

            return True
        except Exception as e:
            print(f"  [FAIL] Failed to load trainer state: {e}")
            return False
