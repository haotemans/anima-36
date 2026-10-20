"""Optional fused Triton SKA kernels.

The PyTorch implementation in lsconv.py remains the portable reference. Triton is
an optional acceleration path and is never imported as a mandatory dependency.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs and minimal CI
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _ska_fwd(X, W, Y, C: tl.constexpr, H: tl.constexpr, WW: tl.constexpr,
                 G: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                 BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        xx = i % WW
        yy = (i // WW) % H
        c = (i // (WW * H)) % C
        b = i // (WW * H * C)
        pg = C // G
        p = c % pg
        acc = tl.full((BLOCK,), 0, tl.float32)
        for ky in range(K):
            iy = yy + ky - K // 2
            for kx in range(K):
                ix = xx + kx - K // 2
                inside = mask & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < WW)
                xi = ((b * C + c) * H + iy) * WW + ix
                wi = (((b * pg + p) * (K * K) + ky * K + kx) * H + yy) * WW + xx
                xv = tl.load(X + xi, inside, 0).to(tl.float32)
                wv = tl.load(W + wi, mask, 0).to(tl.float32)
                acc += xv * wv
        tl.store(Y + i, acc, mask)

    @triton.jit
    def _ska_dx(GRAD, W, DX, C: tl.constexpr, H: tl.constexpr, WW: tl.constexpr,
                G: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        xx = i % WW
        yy = (i // WW) % H
        c = (i // (WW * H)) % C
        b = i // (WW * H * C)
        pg = C // G
        p = c % pg
        acc = tl.full((BLOCK,), 0, tl.float32)
        for ky in range(K):
            oy = yy - (ky - K // 2)
            for kx in range(K):
                ox = xx - (kx - K // 2)
                inside = mask & (oy >= 0) & (oy < H) & (ox >= 0) & (ox < WW)
                gi = ((b * C + c) * H + oy) * WW + ox
                wi = (((b * pg + p) * (K * K) + ky * K + kx) * H + oy) * WW + ox
                gv = tl.load(GRAD + gi, inside, 0).to(tl.float32)
                wv = tl.load(W + wi, inside, 0).to(tl.float32)
                acc += gv * wv
        tl.store(DX + i, acc, mask)

    @triton.jit
    def _ska_dw(X, GRAD, DW, C: tl.constexpr, H: tl.constexpr, WW: tl.constexpr,
                G: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        xx = i % WW
        yy = (i // WW) % H
        q = i // (H * WW)
        koff = q % (K * K)
        p = (q // (K * K)) % (C // G)
        b = q // ((C // G) * (K * K))
        pg = C // G
        ky = koff // K
        kx = koff % K
        iy = yy + ky - K // 2
        ix = xx + kx - K // 2
        inside = mask & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < WW)
        total = tl.full((BLOCK,), 0, tl.float32)
        for group in range(G):
            c = group * pg + p
            xi = ((b * C + c) * H + iy) * WW + ix
            gi = ((b * C + c) * H + yy) * WW + xx
            xv = tl.load(X + xi, inside, 0).to(tl.float32)
            gv = tl.load(GRAD + gi, mask, 0).to(tl.float32)
            total += xv * gv
        tl.store(DW + i, total, mask)


class _TritonSKA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weights: torch.Tensor, groups: int, kernel_size: int):
        if triton is None:
            raise RuntimeError("Triton is not installed")
        b, c, h, w = x.shape
        k = kernel_size
        x = x.contiguous()
        weights = weights.contiguous()
        out = torch.empty_like(x)
        block = 128
        _ska_fwd[(triton.cdiv(x.numel(), block),)](
            x, weights, out, c, h, w, groups, k, x.numel(), block,
            num_warps=4,
        )
        ctx.save_for_backward(x, weights)
        ctx.groups, ctx.kernel_size = groups, k
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weights = ctx.saved_tensors
        b, c, h, w = x.shape
        groups, k = ctx.groups, ctx.kernel_size
        grad_output = grad_output.contiguous()
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.empty_like(weights) if ctx.needs_input_grad[1] else None
        block = 128
        if dx is not None:
            _ska_dx[(triton.cdiv(x.numel(), block),)](
                grad_output, weights, dx, c, h, w, groups, k, x.numel(), block,
                num_warps=4,
            )
        if dw is not None:
            _ska_dw[(triton.cdiv(weights.numel(), block),)](
                x, grad_output, dw, c, h, w, groups, k, weights.numel(), block,
                num_warps=4,
            )
        return dx, dw, None, None


def triton_ska_available() -> bool:
    """Whether the optional Triton package is importable."""
    return triton is not None


def triton_ska(x: torch.Tensor, weights: torch.Tensor, groups: int, kernel_size: int):
    """Run fused SKA; caller must ensure CUDA, contiguity-compatible shapes and Triton."""
    return _TritonSKA.apply(x, weights, groups, kernel_size)
