"""Spatial positional encodings used by the Anima-36 experimental model."""
import torch


def _inverse_frequencies(pair_count: int, device: torch.device, base: float = 10000.0) -> torch.Tensor:
    if pair_count < 1:
        raise ValueError("dimensions must provide at least one frequency pair")
    indices = torch.arange(pair_count, device=device, dtype=torch.float32)
    return base ** (-indices / pair_count)


def _rotate_pairs(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs; x is [B, N, heads, axis_dim]."""
    original_dtype = x.dtype
    paired = x.float().reshape(*x.shape[:-1], -1, 2)
    angle = angles[None, :, None, :]
    cos, sin = angle.cos(), angle.sin()
    a, b = paired.unbind(dim=-1)
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
    return rotated.flatten(-2).to(original_dtype)


def apply_axial_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_size: tuple[int, int],
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply axial 2D RoPE to Q/K shaped [batch, tokens, heads, head_dim].

    Half of each head is assigned to row positions and half to column positions.
    This implementation requires head_dim divisible by four.
    """
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("q and k must have matching [batch, tokens, heads, head_dim] shapes")
    h, w = grid_size
    if h < 1 or w < 1 or h * w != q.shape[1]:
        raise ValueError("grid_size must be positive and match the token count")
    head_dim = q.shape[-1]
    if head_dim % 4:
        raise ValueError("2D RoPE requires head_dim divisible by 4")
    pair_count = head_dim // 4
    inv_freq = _inverse_frequencies(pair_count, q.device, base)
    rows = torch.arange(h, device=q.device, dtype=torch.float32).repeat_interleave(w)
    cols = torch.arange(w, device=q.device, dtype=torch.float32).repeat(h)
    row_angles = rows[:, None] * inv_freq[None, :]
    col_angles = cols[:, None] * inv_freq[None, :]
    q_y, q_x = q.split(head_dim // 2, dim=-1)
    k_y, k_x = k.split(head_dim // 2, dim=-1)
    return (
        torch.cat((_rotate_pairs(q_y, row_angles), _rotate_pairs(q_x, col_angles)), dim=-1),
        torch.cat((_rotate_pairs(k_y, row_angles), _rotate_pairs(k_x, col_angles)), dim=-1),
    )


def sinusoidal_2d_embedding(
    grid_size: tuple[int, int],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a deterministic absolute 2D sin/cos embedding [1, H*W, dim].

    Used by the positive-feature LS-Linear path. Applying RoPE to its positive
    kernel features would invalidate the nonnegative feature-map formulation.
    """
    h, w = grid_size
    if h < 1 or w < 1:
        raise ValueError("grid dimensions must be positive")
    if dim % 4:
        raise ValueError("2D sinusoidal embedding requires dim divisible by 4")
    pair_count = dim // 4
    inv_freq = _inverse_frequencies(pair_count, device)
    rows = torch.arange(h, device=device, dtype=torch.float32).repeat_interleave(w)
    cols = torch.arange(w, device=device, dtype=torch.float32).repeat(h)
    y_phase = rows[:, None] * inv_freq[None, :]
    x_phase = cols[:, None] * inv_freq[None, :]
    y_embed = torch.cat((y_phase.sin(), y_phase.cos()), dim=-1)
    x_embed = torch.cat((x_phase.sin(), x_phase.cos()), dim=-1)
    return torch.cat((y_embed, x_embed), dim=-1).unsqueeze(0).to(dtype=dtype)
