"""Elastic Weight Consolidation (EWC) for continual learning.

Prevents catastrophic forgetting during hourly incremental training.
The model learns new financial data each hour without losing previously
learned knowledge by penalizing changes to important parameters.
"""

from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn as nn


class ElasticWeightConsolidation:
    """EWC: Penalize changes to parameters important for previous tasks.

    After each training run, we compute the Fisher Information Matrix
    (diagonal approximation) to identify which parameters were critical.
    On subsequent runs, the loss includes a penalty for deviating from
    those critical parameter values.

    This is essential for hourly continual training — without it, the
    model would forget early knowledge as it trains on new data.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 100.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher_diag: Dict[str, torch.Tensor] = {}
        self._initialized = False

    def compute_fisher(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_samples: int = 200,
    ) -> None:
        """Compute diagonal Fisher Information Matrix from training data.

        Uses a subset of data (max_samples) for efficiency on CPU.
        """
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data, dtype=param.dtype)

        import psutil

        def log_mem(prefix):
            mem = psutil.virtual_memory()
            print(f"    [EWC RAM] {prefix}: {mem.used / 1e9:.1f}GB ({mem.percent}%)")

        log_mem("Allocated fisher tensors")

        count = 0
        log_mem("Starting dataloader loop")
        for batch in dataloader:
            if count >= max_samples:
                print(f"    [DEBUG] EWC: Reached max_samples ({max_samples}). Stopping.")
                break

            print(f"    [DEBUG] EWC: Processing sample {count+1}/{max_samples}...")
            log_mem(f"Sample {count+1}")

            input_ids = batch["input_ids"]
            labels = batch.get("labels")
            if labels is None:
                labels = input_ids

            if isinstance(input_ids, list):
                input_ids = torch.stack(input_ids)
            if isinstance(labels, list):
                labels = torch.stack(labels)

            max_fisher_len = int(os.getenv("FISHER_SEQ_LEN", "64"))
            if input_ids.dim() == 2 and input_ids.size(1) > max_fisher_len:
                input_ids = input_ids[:, :max_fisher_len]
                labels = labels[:, :max_fisher_len]

            try:
                print(f"    [DEBUG] EWC Sample {count+1}: Forward pass...")
                outputs = model(input_ids=input_ids, labels=labels)
                if outputs.loss is not None:
                    print(f"    [DEBUG] EWC Sample {count+1}: Backward pass...")
                    outputs.loss.backward()

                    print(f"    [DEBUG] EWC Sample {count+1}: Accumulating Fisher diag...")
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fisher[name] += param.grad.data.pow(2).to(param.dtype)

                    print(f"    [DEBUG] EWC Sample {count+1}: Zeroing grads...")
                    model.zero_grad(set_to_none=True)

                # Explicitly clear intermediate tensors
                print(f"    [DEBUG] EWC Sample {count+1}: Clearing tensors...")
                del outputs
                del input_ids
                del labels
            except Exception as e:
                print(f"[WARN] EWC sample failed: {e}")
                model.zero_grad(set_to_none=True)

            count += 1
            import gc

            gc.collect()
            if count % 10 == 0:
                log_mem(f"After GC (Sample {count})")

        # Average
        for name in fisher:
            fisher[name] /= max(count, 1)

        fisher_threshold = float(os.getenv("FISHER_THRESHOLD", "1e-6"))

        named_params = dict(model.named_parameters())
        kept_fisher: Dict[str, torch.Tensor] = {}
        kept_prev: Dict[str, torch.Tensor] = {}
        for name, f in fisher.items():
            # Keep only non-trivial fisher entries to reduce RAM
            mask = f > fisher_threshold
            if mask.any():
                kept_fisher[name] = f
                kept_prev[name] = named_params[name].data.to(named_params[name].dtype).clone()

        print(f"  [EWC] Keeping {len(kept_fisher)}/{len(fisher)} params above threshold")
        self.fisher_diag = kept_fisher
        self.prev_params = kept_prev
        self._initialized = True
        model.train()

    def penalty_value(self, model: nn.Module) -> float:
        """Compute EWC penalty value for logging ONLY (no autograd)."""
        if not self._initialized:
            return 0.0

        total_penalty = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.fisher_diag and name in self.prev_params:
                    # Cast stored bfloat16 tensors to param dtype in-place view
                    # (no copy if already matching dtype/device)
                    fisher = self.fisher_diag[name]
                    prev = self.prev_params[name]
                    if fisher.dtype != param.dtype:
                        fisher = fisher.to(dtype=param.dtype)
                    if prev.dtype != param.dtype:
                        prev = prev.to(dtype=param.dtype)

                    # diff = (w - w_old), computed in-place to avoid a 3rd tensor
                    diff = param.detach().sub(prev)
                    diff.mul_(diff)  # diff = diff^2  (in-place)
                    diff.mul_(fisher)  # diff = fisher * diff^2  (in-place)
                    total_penalty += diff.sum().item()

                    del fisher, prev, diff

        return 0.5 * self.ewc_lambda * total_penalty

    def apply_gradients(self, model: nn.Module, scale: float = 1.0) -> None:
        """Apply EWC gradients manually to current parameter gradients.

        This avoids the massive memory overhead of building the autograd
        graph for the EWC penalty across 478M parameters.

        Gradient of EWC Loss: lambda * Fisher * (param - prev_param)
        """
        if not self._initialized:
            return

        alpha = self.ewc_lambda * scale
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in self.fisher_diag and name in self.prev_params:
                    fisher = self.fisher_diag[name]
                    prev = self.prev_params[name]
                    if fisher.dtype != param.dtype:
                        fisher = fisher.to(dtype=param.dtype)
                    if prev.dtype != param.dtype:
                        prev = prev.to(dtype=param.dtype)

                    # ewc_grad = fisher * (param - prev), added in-place to existing gradient
                    diff = param.detach().sub(prev)
                    diff.mul_(fisher)
                    param.grad.add_(diff, alpha=alpha)

                    del fisher, prev, diff

    def save(self, path: str) -> None:
        """Save Fisher + previous params for next training run."""
        # Ensure we don't save with cross-thread locks
        state = {"fisher": self.fisher_diag, "prev_params": self.prev_params}
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load Fisher + previous params from previous training run."""
        # Use weights_only=True for security and speed
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            self.fisher_diag = data["fisher"]
            self.prev_params = data["prev_params"]
            self._initialized = True
        except Exception as e:
            print(f"  [WARN] Failed to load EWC state: {e}")
            self._initialized = False
