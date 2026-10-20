"""Layerwise text feature projection and staged routing for Qwen3.5-4B."""
import torch
from torch import nn
from .norm import RMSNorm

def stage_features_for_block(block_index: int):
    if not 0 <= block_index < 36:
        raise IndexError("block_index must be in [0, 36)")
    if block_index < 12:
        return ("middle", "deep")
    if block_index < 24:
        return ("shallow", "middle", "deep")
    return ("shallow", "deep")

class LayerwiseTextProjector(nn.Module):
    """Independent normalization/projection for selected shallow/middle/deep states."""
    def __init__(self, input_dim: int, output_dim: int = 2048):
        super().__init__()
        self.norms = nn.ModuleDict({name: RMSNorm(input_dim) for name in ("shallow", "middle", "deep")})
        self.projections = nn.ModuleDict({name: nn.Linear(input_dim, output_dim) for name in ("shallow", "middle", "deep")})

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"shallow", "middle", "deep"}
        if set(features) != required:
            raise ValueError(f"features must have exactly these keys: {sorted(required)}")
        shapes = {tuple(value.shape[:2]) for value in features.values()}
        if len(shapes) != 1:
            raise ValueError("all text feature tensors must share batch and token dimensions")
        return {name: self.projections[name](self.norms[name](value))
                for name, value in features.items()}
