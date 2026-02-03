"""BitNet b1.58 Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x):
    # Per-token quantization for better stability and handling of outliers
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    return (x * scale).round().clamp(-128, 127) / scale


def weight_quant(w):
    scale = w.abs().mean()
    e = w.mean()
    w_centered = w - e
    w_ternary = torch.sign(w_centered)
    return w_ternary * scale


class BitLinear(nn.Linear):
    def forward(self, x):
        x_quant = x + (activation_quant(x) - x).detach()
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return F.linear(x_quant, w_quant, self.bias)


class BitRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed
