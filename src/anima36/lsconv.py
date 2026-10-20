"""PyTorch LSConv: LSNet-style Large Kernel Perception + Small Kernel Aggregation.

The dynamic aggregation below is a portable PyTorch reference implementation.
It intentionally avoids the upstream Triton-only SKA kernel so CPU CI can test it.
Compare numerics against the official implementation before using pretrained LSNet weights.
"""
import torch
from torch import nn
from torch.nn import functional as F

from anima36.ska_triton import triton_ska, triton_ska_available

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )

class LargeKernelPerception(nn.Module):
    """Predict per-location 3x3 aggregation weights using a large-kernel path."""
    def __init__(self, dim: int, large_kernel: int = 7, small_kernel: int = 3, groups: int = 8):
        super().__init__()
        if dim % groups or dim % 2:
            raise ValueError("dim must be even and divisible by groups")
        if large_kernel % 2 != 1 or small_kernel % 2 != 1:
            raise ValueError("kernel sizes must be odd")
        self.dim, self.groups, self.small_kernel = dim, groups, small_kernel
        mid = dim // 2
        weight_channels = (dim // groups) * small_kernel * small_kernel
        self.cv1 = ConvBN(dim, mid)
        self.cv2 = ConvBN(mid, mid, large_kernel, (large_kernel - 1) // 2, groups=mid)
        self.cv3 = ConvBN(mid, mid)
        self.act = nn.ReLU()
        self.cv4 = nn.Conv2d(mid, weight_channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=weight_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.act(self.cv1(x))
        weights = self.act(self.cv3(self.cv2(weights)))
        weights = self.norm(self.cv4(weights))
        b, _, h, w = weights.shape
        return weights.reshape(b, self.dim // self.groups, self.small_kernel ** 2, h, w)

class SmallKernelAggregation(nn.Module):
    """Dynamic local aggregation; weights are shared across channel groups as in LSNet."""
    def __init__(self, groups: int = 8, kernel_size: int = 3):
        super().__init__()
        self.groups, self.kernel_size = groups, kernel_size

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k, groups = self.kernel_size, self.groups
        if c % groups:
            raise ValueError(f"channels ({c}) must be divisible by groups ({groups})")
        per_group = c // groups
        if weights.shape != (b, per_group, k * k, h, w):
            raise ValueError(f"unexpected dynamic weight shape: {tuple(weights.shape)}")
        if weights.device != x.device:
            raise ValueError("input and dynamic weights must be on the same device")
        if (x.is_cuda and triton_ska_available()
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and weights.dtype == x.dtype):
            return triton_ska(x, weights, groups, k)
        patches = F.unfold(x, kernel_size=k, padding=k // 2)
        patches = patches.reshape(b, groups, per_group, k * k, h, w)
        out = (patches * weights.unsqueeze(1)).sum(dim=3)
        return out.reshape(b, c, h, w)

class LSConv(nn.Module):
    """Residual LSConv with LKP-generated weights and SKA aggregation."""
    def __init__(self, dim: int, large_kernel: int = 7, small_kernel: int = 3, groups: int = 8):
        super().__init__()
        self.lkp = LargeKernelPerception(dim, large_kernel, small_kernel, groups)
        self.ska = SmallKernelAggregation(groups, small_kernel)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn(self.ska(x, self.lkp(x)))
