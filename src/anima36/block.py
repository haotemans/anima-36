"""Pre-norm conditioned transformer block for Anima-36."""
import torch
from torch import nn
from .attnres import BlockAttnRes
from .attention import LSLinearAttention, StandardSelfAttention, TextCrossAttention
from .ffn import SwiGLU
from .norm import RMSNorm
from .schedule import attention_kind

class Anima36Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, context_dim: int,
                 block_index: int, depth: int = 36, ffn_dim: int | None = None,
                 kv_stride: int = 2, attnres_sources: int = 4):
        super().__init__()
        self.kind = attention_kind(block_index, depth)
        self.norm_attn = RMSNorm(dim)
        if self.kind == "ls_linear":
            self.attn = LSLinearAttention(dim, num_heads, kv_stride)
        else:
            self.attn = StandardSelfAttention(dim, num_heads)
        self.norm_text = RMSNorm(dim)
        self.text_attn = TextCrossAttention(dim, context_dim, num_heads)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim or dim * 4)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.attnres = BlockAttnRes(dim, max_sources=attnres_sources)

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor,
                context: torch.Tensor, context_mask: torch.Tensor | None,
                grid_size: tuple[int, int], prior_states: list[torch.Tensor]):
        shift_a, scale_a, gate_a, shift_t, scale_t, gate_t = self.modulation(time_cond).chunk(6, dim=-1)
        def modulate(normed, shift, scale):
            return normed * (1 + scale[:, None, :]) + shift[:, None, :]
        x = x + gate_a[:, None, :] * self.attn(modulate(self.norm_attn(x), shift_a, scale_a), grid_size)
        x = x + gate_t[:, None, :] * self.text_attn(modulate(self.norm_text(x), shift_t, scale_t), context, context_mask)
        x = x + self.ffn(self.norm_ffn(x))
        return self.attnres(x, prior_states)
