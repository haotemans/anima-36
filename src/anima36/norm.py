"""Minimal RMSNorm; verify epsilon and checkpoint compatibility during DS integration."""
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        y = (x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        if self.weight is not None:
            y = y * self.weight.to(dtype=y.dtype)
        return y
