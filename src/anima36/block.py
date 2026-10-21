"""Timestep-conditioned hybrid Linear Attention + LSNet transformer block."""
import torch
from torch import nn
from .attnres import BlockAttnRes
from .attention import LSLinearAttention, TextCrossAttention
from .ffn import SwiGLU
from .norm import RMSNorm


class Anima36Block(nn.Module):
    """Every block combines global linear attention, local LSNet, text cross-attention and FFN.

    The timestep embedding is computed once by the model. This block maps that shared
    condition to independent AdaLN shift/scale/residual-gate parameters for each branch.
    """
    def __init__(self, dim: int, num_heads: int, context_dim: int,
                 block_index: int = 0, depth: int = 36, ffn_dim: int | None = None,
                 kv_stride: int = 2, attnres_sources: int = 4):
        super().__init__()
        # Kept as constructor metadata for call-site compatibility; there is no
        # longer a per-layer attention schedule: all blocks use the same hybrid.
        self.block_index, self.depth = block_index, depth
        self.kind = "hybrid_linear_lsnet"
        self.norm_attn = RMSNorm(dim)
        self.attn = LSLinearAttention(dim, num_heads, kv_stride)
        self.norm_text = RMSNorm(dim)
        self.text_attn = TextCrossAttention(dim, context_dim, num_heads)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim or dim * 4)
        # Three branches x (shift, scale, residual gate).
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 9))
        self.attnres = BlockAttnRes(dim, max_sources=attnres_sources)

    @staticmethod
    def _modulate(normed: torch.Tensor, shift: torch.Tensor,
                  scale: torch.Tensor) -> torch.Tensor:
        return normed * (1 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor,
                context: torch.Tensor, context_mask: torch.Tensor | None,
                grid_size: tuple[int, int], prior_states: list[torch.Tensor]):
        params = self.modulation(time_cond).chunk(9, dim=-1)
        shift_a, scale_a, gate_a, shift_t, scale_t, gate_t, shift_f, scale_f, gate_f = params
        attn_input = self._modulate(self.norm_attn(x), shift_a, scale_a)
        x = x + gate_a[:, None, :] * self.attn(attn_input, grid_size)
        text_input = self._modulate(self.norm_text(x), shift_t, scale_t)
        x = x + gate_t[:, None, :] * self.text_attn(text_input, context, context_mask)
        ffn_input = self._modulate(self.norm_ffn(x), shift_f, scale_f)
        x = x + gate_f[:, None, :] * self.ffn(ffn_input)
        return self.attnres(x, prior_states)
