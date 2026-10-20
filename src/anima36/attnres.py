"""Block AttnRes placeholder.

Do not implement with a guessed equation. First specify source-state selection,
normalization, aggregation weights, gating, and residual insertion semantics.
"""
from torch import nn

class BlockAttnRes(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Define and test the Block AttnRes equation first.")
