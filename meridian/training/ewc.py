"""Elastic Weight Consolidation (EWC) for continual learning.

Prevents catastrophic forgetting during hourly incremental training.
The model learns new financial data each hour without losing previously
learned knowledge by penalizing changes to important parameters.
"""

from __future__ import annotations

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
                fisher[name] = torch.zeros_like(param.data, dtype=torch.bfloat16)

        import psutil

        def log_mem(prefix):
            mem = psutil.virtual_memory()
            print(f"    [EWC RAM] {prefix}: {mem.used / 1e9:.1f}GB ({mem.percent}%)")

        log_mem("Allocated fisher bfloat16 tensors")

        count = 0
        log_mem("Starting dataloader loop")
        for batch in dataloader:
            if count >= max_samples:
                print(f"    [DEBUG] EWC: Reached max_samples ({max_samples}). Stopping.")
                break

            print(f"    [DEBUG] EWC: Processing sample {count+1}/{max_samples}...")
            log_mem(f"Sample {count+1}")

            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids.clone())

            if isinstance(input_ids, list):
                input_ids = torch.stack(input_ids)
            if isinstance(labels, list):
                labels = torch.stack(labels)

            try:
                print(f"    [DEBUG] EWC Sample {count+1}: Forward pass...")
                outputs = model(input_ids=input_ids, labels=labels)
                if outputs.loss is not None:
                    print(f"    [DEBUG] EWC Sample {count+1}: Backward pass...")
                    outputs.loss.backward()

                    print(f"    [DEBUG] EWC Sample {count+1}: Accumulating Fisher diag...")
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fisher[name] += param.grad.data.pow(2).to(torch.bfloat16)

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
            if count % 5 == 0:
                print(f"    [DEBUG] EWC: Periodic GC collect at sample {count}...")
                import gc

                gc.collect()
                log_mem(f"After GC (Sample {count})")

        # Average
        for name in fisher:
            fisher[name] /= max(count, 1)

        self.fisher_diag = fisher
        self.prev_params = {
            name: param.data.clone().to(torch.bfloat16)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
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
                    # Non-blocking transfer for speed
                    fisher = self.fisher_diag[name].to(device=param.device, dtype=param.dtype, non_blocking=True)
                    prev = self.prev_params[name].to(device=param.device, dtype=param.dtype, non_blocking=True)
                    
                    # Compute (w - w_old)^2 * Fisher
                    # We use a very tight loop to minimize memory high-water mark
                    p = (fisher * (param - prev).pow(2)).sum().item()
                    total_penalty += p
                    
                    del fisher, prev

        return 0.5 * self.ewc_lambda * total_penalty

    def apply_gradients(self, model: nn.Module, scale: float = 1.0) -> None:
        """Apply EWC gradients manually to current parameter gradients.
        
        This avoids the massive memory overhead of building the autograd 
        graph for the EWC penalty across 478M parameters.
        
        Gradient of EWC Loss: lambda * Fisher * (param - prev_param)
        """
        if not self._initialized:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in self.fisher_diag and name in self.prev_params:
                    fisher = self.fisher_diag[name].to(device=param.device, dtype=param.dtype, non_blocking=True)
                    prev = self.prev_params[name].to(device=param.device, dtype=param.dtype, non_blocking=True)
                    
                    # Update grad: grad = grad + scale * lambda * fisher * (param - prev)
                    # We do it in-place to save memory
                    param.grad.add_(fisher * (param - prev), alpha=self.ewc_lambda * scale)
                    
                    del fisher, prev

    def save(self, path: str) -> None:
        """Save Fisher + previous params for next training run."""
        # Ensure we don't save with cross-thread locks
        state = {
            "fisher": self.fisher_diag,
            "prev_params": self.prev_params
        }
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
