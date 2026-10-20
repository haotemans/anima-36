"""Attention interfaces; numerical kernels are intentionally not implemented yet."""
from torch import nn

class LSLinearAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Specify linear kernel, K/V compression, masks, and RoPE handling.")

class StandardSelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Implement against verified DiffSynth-Studio attention conventions.")

class TextCrossAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Specify text dimensions, padding masks, and parameter-sharing policy.")
