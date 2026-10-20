"""PyTorch attention primitives for the Anima-36 experimental design."""
import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # Compatibility with older supported PyTorch builds.
    SDPBackend = None
    sdpa_kernel = None


def _scaled_dot_product_attention(q, k, v, attn_mask=None):
    """Prefer FlashAttention, while retaining compatible SDPA fallbacks.

    PyTorch dispatches to FlashAttention only when device, dtype, shape, and
    mask constraints permit it; otherwise efficient attention or math is used.
    """
    if sdpa_kernel is None or SDPBackend is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    with sdpa_kernel([
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
from .lsconv import LSConv
from .norm import RMSNorm

class LSLinearAttention(nn.Module):
    """Full-resolution Q, spatially pooled K/V, positive-feature linear attention."""
    def __init__(self, dim: int, num_heads: int = 16, kv_stride: int = 2,
                 lsconv_groups: int = 8, eps: float = 1e-6):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if kv_stride < 1:
            raise ValueError("kv_stride must be >= 1")
        self.dim, self.num_heads = dim, num_heads
        self.head_dim, self.kv_stride, self.eps = dim // num_heads, kv_stride, eps
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim, eps), RMSNorm(self.head_dim, eps)
        self.local = LSConv(dim, groups=lsconv_groups)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = grid_size
        if h * w != n or d != self.dim:
            raise ValueError("grid_size must match token count and configured dim")
        image = x.transpose(1, 2).reshape(b, d, h, w)
        image = self.local(image)
        tokens = image.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        q = q.reshape(b, n, self.num_heads, self.head_dim)
        k = k.reshape(b, n, self.num_heads, self.head_dim)
        v = v.reshape(b, n, self.num_heads, self.head_dim)
        if self.kv_stride > 1 and h >= self.kv_stride and w >= self.kv_stride:
            def pool_heads(t):
                t = t.permute(0, 2, 3, 1).reshape(b, d, h, w)
                t = F.avg_pool2d(t, self.kv_stride, self.kv_stride, ceil_mode=True)
                return t.flatten(2).transpose(1, 2).reshape(b, -1, self.num_heads, self.head_dim)
            k, v = pool_heads(k), pool_heads(v)
        q = F.elu(self.q_norm(q).float()) + 1.0
        k = F.elu(self.k_norm(k).float()) + 1.0
        v = v.float()
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)
        numerator = torch.einsum("bhnd,bhde->bhne", q, kv)
        denominator = torch.einsum("bhnd,bhd->bhn", q, k_sum).unsqueeze(-1).clamp_min(self.eps)
        y = (numerator / denominator).transpose(1, 2).reshape(b, n, d).to(x.dtype)
        return self.out(y)

class StandardSelfAttention(nn.Module):
    """Standard full-resolution scaled dot-product self-attention."""
    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, grid_size=None) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).unbind(dim=2)
        q, k = self.q_norm(q), self.k_norm(k)
        y = _scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                      v.transpose(1, 2))
        return self.out(y.transpose(1, 2).reshape(b, n, d))

class TextCrossAttention(nn.Module):
    """Image queries attend to separately encoded text tokens; text never enters LSConv."""
    def __init__(self, dim: int, context_dim: int, num_heads: int = 16):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(context_dim, dim * 2)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        m = context.shape[1]
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim)
        k, v = self.kv(context).reshape(b, m, 2, self.num_heads, self.head_dim).unbind(dim=2)
        q, k = self.q_norm(q), self.k_norm(k)
        mask = None
        if context_mask is not None:
            if context_mask.shape != (b, m):
                raise ValueError("context_mask must have shape [batch, text_tokens]")
            mask = context_mask[:, None, None, :].to(torch.bool)
        y = _scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                      v.transpose(1, 2), attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, n, d))
