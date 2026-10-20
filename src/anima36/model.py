"""Small, configurable PyTorch Anima-36 core; not yet a DS pipeline replacement."""
import math
import torch
from torch import nn
from .block import Anima36Block
from .schedule import build_attention_schedule

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(
        half, device=t.device, dtype=torch.float32) / max(half, 1))
    args = t.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    emb = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb

class Anima36Core(nn.Module):
    """Latent-in/latent-out transformer with 2x patching and text cross-attention.

    Defaults express the intended large configuration and require substantial accelerator
    memory. Use small dimensions for unit tests. VAE conventions and pipeline integration
    remain external responsibilities.
    """
    def __init__(self, latent_channels: int = 16, dim: int = 2048,
                 depth: int = 36, num_heads: int = 16, context_dim: int = 2048,
                 patch_size: int = 2, ffn_dim: int | None = None, kv_stride: int = 2):
        super().__init__()
        if depth % 4:
            raise ValueError("depth must be a positive multiple of four")
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.latent_channels, self.dim, self.depth = latent_channels, dim, depth
        self.patch_size = patch_size
        self.schedule = build_attention_schedule(depth)
        self.patch_embed = nn.Conv2d(latent_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.blocks = nn.ModuleList([
            Anima36Block(dim, num_heads, context_dim, i, depth, ffn_dim, kv_stride)
            for i in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.output = nn.ConvTranspose2d(dim, latent_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor,
                text_context: torch.Tensor, text_mask: torch.Tensor | None = None):
        b, _, h, w = latents.shape
        p = self.patch_size
        if h % p or w % p:
            raise ValueError("latent height and width must be divisible by patch_size")
        image = self.patch_embed(latents)
        gh, gw = image.shape[-2:]
        x = image.flatten(2).transpose(1, 2)
        time_cond = self.time_mlp(timestep_embedding(timesteps, self.dim).to(latents.dtype))
        prior_states = []
        for block in self.blocks:
            x = block(x, time_cond, text_context, text_mask, (gh, gw), prior_states)
            prior_states.append(x)
        shift, scale = self.final_modulation(time_cond).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        image = x.transpose(1, 2).reshape(b, self.dim, gh, gw)
        return self.output(image)
