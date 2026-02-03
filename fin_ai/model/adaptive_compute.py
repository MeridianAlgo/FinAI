"""Adaptive Compute"""

import torch
import torch.nn as nn
from .bitnet import BitLinear, BitRMSNorm


class AdaptiveComputeWrapper(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.threshold = config.dynamic_depth_threshold
        self.gate = BitLinear(config.hidden_size, 1)
        self.layer_idx = layer_idx

    def forward(self, x):
        confidence = torch.sigmoid(self.gate(x)).mean()
        should_skip = confidence > self.threshold
        return x, should_skip


class MultimodalProjector(nn.Module):
    def __init__(self, config, input_dim, modal_name="vision"):
        super().__init__()
        self.name = modal_name
        self.projector = nn.Sequential(
            BitLinear(input_dim, config.hidden_size),
            BitRMSNorm(config.hidden_size),
            nn.SiLU(),
            BitLinear(config.hidden_size, config.hidden_size)
        )

    def forward(self, x):
        return self.projector(x)
