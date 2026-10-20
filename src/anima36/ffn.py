"""SwiGLU feed-forward network with fused gate/up projection."""
import torch
from torch import nn
from torch.nn import functional as F

class SwiGLU(nn.Module):
    """SwiGLU using one 2H projection instead of two H projections.

    This reduces Linear launch count while keeping the same parameter count.
    Checkpoint conversion from separate gate_proj/up_proj weights must concatenate
    them in the output dimension.
    """
    def __init__(self, dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
