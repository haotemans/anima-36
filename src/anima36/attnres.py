"""Explicit learned aggregation over recent block states (Block AttnRes prototype).

This is a concrete prototype equation for experimentation, not a claim of exact parity
with any external Block AttnRes paper/implementation. It normalizes each source state,
learns softmax aggregation weights, projects the aggregate, and applies a gated residual.
"""
import torch
from torch import nn
from .norm import RMSNorm

class BlockAttnRes(nn.Module):
    def __init__(self, dim: int, max_sources: int = 4):
        super().__init__()
        if max_sources < 1:
            raise ValueError("max_sources must be positive")
        self.max_sources = max_sources
        self.norm = RMSNorm(dim)
        self.source_logits = nn.Parameter(torch.zeros(max_sources))
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor, states: list[torch.Tensor]) -> torch.Tensor:
        if not states:
            return x
        sources = states[-self.max_sources:]
        if any(s.shape != x.shape for s in sources):
            raise ValueError("all AttnRes source states must match x shape")
        logits = self.source_logits[-len(sources):].softmax(dim=0)
        # Stack once and reduce in one tensor op instead of launching one weighted
        # elementwise op per source. This trades a small temporary for fewer launches.
        normalized = torch.stack([self.norm(state) for state in sources], dim=0)
        aggregate = (normalized * logits.view(-1, 1, 1, 1)).sum(dim=0)
        return x + torch.sigmoid(self.gate) * self.proj(aggregate)
