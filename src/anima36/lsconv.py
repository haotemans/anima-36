"""LSConv placeholder for a faithful port of official THU-MIG/LSNet."""
from torch import nn

class LSConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "Port official Large Kernel Perception + Small Kernel Aggregation; do not replace with generic depthwise convolution."
        )
