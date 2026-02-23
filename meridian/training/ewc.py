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

        count = 0
        for batch in dataloader:
            if count >= max_samples:
                break

            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids.clone())

            if isinstance(input_ids, list):
                input_ids = torch.stack(input_ids)
            if isinstance(labels, list):
                labels = torch.stack(labels)

            try:
                outputs = model(input_ids=input_ids, labels=labels)
                if outputs.loss is not None:
                    outputs.loss.backward()

                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fisher[name] += param.grad.data.pow(2).to(torch.bfloat16)

                    model.zero_grad(set_to_none=True)

                # Explicitly clear intermediate tensors
                del outputs
            except Exception as e:
                print(f"[WARN] EWC sample failed: {e}")
                model.zero_grad(set_to_none=True)

            count += 1
            if count % 20 == 0:
                import gc

                gc.collect()

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

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty loss."""
        if not self._initialized:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher_diag and name in self.prev_params:
                fisher = self.fisher_diag[name].to(param.device, dtype=param.dtype)
                prev = self.prev_params[name].to(param.device, dtype=param.dtype)
                loss += (fisher * (param - prev).pow(2)).sum()

        return 0.5 * self.ewc_lambda * loss

    def save(self, path: str) -> None:
        """Save Fisher + previous params for next training run."""
        torch.save(
            {"fisher": self.fisher_diag, "prev_params": self.prev_params},
            path,
        )

    def load(self, path: str) -> None:
        """Load Fisher + previous params from previous training run."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.fisher_diag = data["fisher"]
        self.prev_params = data["prev_params"]
        self._initialized = True
