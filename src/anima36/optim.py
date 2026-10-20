"""Optional PyTorch compiler and fused-optimizer helpers."""
from __future__ import annotations
import inspect
import torch

def compile_model(model, *, mode: str = "default", dynamic: bool = False):
    """Wrap a model with torch.compile when available; leave eager as explicit fallback.

    Compilation is best tested on the target GPU and representative shapes. The
    first compiled steps incur compilation overhead; benchmark steady-state steps.
    """
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        raise RuntimeError("torch.compile requires a newer PyTorch build")
    return compiler(model, mode=mode, dynamic=dynamic)

def make_adamw(parameters, *, lr: float = 1e-4, weight_decay: float = 0.01,
               fused: bool | None = None, **kwargs):
    """Use fused AdamW on CUDA when supported; safely fall back to standard AdamW."""
    params = list(parameters)
    if fused is None:
        fused = any(p.is_cuda for p in params)
    signature = inspect.signature(torch.optim.AdamW)
    if "fused" in signature.parameters:
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                      fused=fused, **kwargs)
        except (RuntimeError, TypeError, ValueError):
            if fused:
                return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                         fused=False, **kwargs)
            raise
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)

def configure_cuda_performance():
    """Enable TF32 matmul/convolution on CUDA; this changes numerical precision slightly."""
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    return True
