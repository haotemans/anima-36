"""Qwen3.5-4B hidden-state routing interfaces."""
from torch import nn

class LayerwiseTextProjector(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Confirm model hidden-state semantics and dimensions first.")

def stage_features_for_block(block_index: int):
    if not 0 <= block_index < 36:
        raise IndexError("block_index must be in [0, 36)")
    if block_index < 12:
        return ("middle", "deep")
    if block_index < 24:
        return ("shallow", "middle", "deep")
    return ("shallow", "deep")
