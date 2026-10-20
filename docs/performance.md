# PyTorch performance notes

These are opt-in performance paths, not measured speedup claims. Benchmark on the target accelerator with the intended resolution, batch size, dtype, and training step.

## 1. Fused SwiGLU projection

`SwiGLU` uses `gate_up_proj: D -> 2F` and chunks the output into gate/up halves. This replaces two separate `D -> F` Linear launches with one `D -> 2F` launch. Parameter count is unchanged. When importing a checkpoint with separate gate/up projections, concatenate their weights and biases in the output dimension.

## 2. SDPA attention

The standard self-attention and text cross-attention call `torch.nn.functional.scaled_dot_product_attention`. PyTorch selects a compatible SDPA backend (Flash Attention, memory-efficient, or math) based on hardware, dtype, mask and shape. The LS-Linear kernel is a different linear-attention equation and does not use SDPA.

## 3. Activation checkpointing

Set `gradient_checkpointing=True` on `Anima36Core` during training to recompute block activations during backward and reduce activation memory. It can increase step time. It is disabled by default in the Python constructor; enable it deliberately after validating memory/throughput trade-offs.

## 4. Fused AdamW, TF32 and torch.compile

```python
import torch
from anima36.model import Anima36Core
from anima36.optim import make_adamw, configure_cuda_performance, compile_model

model = Anima36Core(gradient_checkpointing=True).cuda()
configure_cuda_performance()  # TF32: faster on supported NVIDIA GPUs, slightly different numerics.
optimizer = make_adamw(model.parameters(), lr=1e-4)  # fused AdamW auto-selects on CUDA.
# Optional: benchmark compile warm-up and steady-state separately.
model = compile_model(model, mode="default", dynamic=False)
```

For BF16, use autocast in the training loop only after validating the target GPU and all operations. Keep loss accumulation / sensitive reductions in FP32 where needed. `torch.compile` is opt-in because graph breaks, recompilation, compile overhead, or unsupported kernels can make it slower.

## 5. Current likely bottleneck: LSConv

The portable reference LSConv uses `unfold`, which materializes local patches and can consume substantial memory at high resolutions. It is intended as a correctness/reference implementation, not the final high-throughput kernel. The next optimization should replace Small Kernel Aggregation with a fused Triton/CUDA kernel and compare output and gradients to this reference across shapes and dtypes. Do not make that optional dependency mandatory for CPU CI.

## Benchmark checklist

- Record GPU model, PyTorch/CUDA versions, resolution, batch size, dtype, and attention backend.
- Exclude initial `torch.compile` warm-up; report steady-state steps/s and peak allocated memory.
- Compare eager vs compiled and checkpointing on/off using the same seeds and effective batch.
- Check finite loss, gradient norms, and output/gradient closeness against the eager reference.
- Do not infer speedups from CPU unit tests.
