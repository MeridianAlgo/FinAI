"""
Liquid Blocks - Adaptive Dynamical Systems for Sequence Modeling
Replaces standard Attention with linear-time stateful recurrence.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .bitnet import BitLinear

@torch.jit.script
def liquid_step_kernel(x_proj: torch.Tensor, gate_proj: torch.Tensor, h: torch.Tensor, state_weight: torch.Tensor):
    """
    Optimized JIT kernel for the liquid recurrence step.
    Calculates the adaptive evolution of the hidden state.
    """
    dt = torch.sigmoid(gate_proj)
    # Manual Linear for state_proj within JIT to avoid module overhead in loop
    inner = x_proj + torch.matmul(h, state_weight.t())
    return (1 - dt) * h + dt * torch.tanh(inner)

class LiquidBlock(nn.Module):
    """
    A single Liquid Block utilizing adaptive step-sizes to update hidden states.
    Scales O(n) with sequence length.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.state_dim = config.liquid_state_dim
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)
        
        # Binary-compatible input/gate projections
        self.proj_in = BitLinear(self.hidden_size, self.state_dim * 2)
        self.state_proj = BitLinear(self.state_dim, self.state_dim)
        self.proj_out = BitLinear(self.state_dim, self.hidden_size)
        
        # Learnable initial memory state
        self.memory = nn.Parameter(torch.zeros(1, self.state_dim))

    def forward(self, x, hidden_state=None):
        def run_block(input_tensor, h_init):
            batch_size, seq_len, _ = input_tensor.shape
            if h_init is None:
                h_init = self.memory.expand(batch_size, -1)
            
            # Parallel projection of entire sequence for speed
            projected = self.proj_in(input_tensor)
            x_projs, gate_projs = torch.chunk(projected, 2, dim=-1)
            
            curr_h = h_init
            state_w = self.state_proj.weight
            
            outputs_list = []
            # Recurrent evolution loop
            for t in range(seq_len):
                curr_h = liquid_step_kernel(x_projs[:, t, :], gate_projs[:, t, :], curr_h, state_w)
                outputs_list.append(self.proj_out(curr_h))
            
            outputs = torch.stack(outputs_list, dim=1)
            return outputs, curr_h

        # Support for Gradient Checkpointing to save RAM on edge devices
        if self.training and self.gradient_checkpointing:
            return checkpoint.checkpoint(run_block, x, hidden_state, use_reentrant=True)
        return run_block(x, hidden_state)
