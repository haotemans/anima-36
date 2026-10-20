# Anima-36

Experimental PyTorch diffusion-transformer prototype targeting integration with [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

**Status: core prototype implemented; CPU CI tests pass. Not trained, not yet a complete DiffSynth-Studio pipeline, and not validated against model weights.**

## Performance work
- SwiGLU gate/up projections are fused into one Linear call, reducing kernel launches without changing parameter count. Checkpoint conversion must concatenate old gate/up weights.
- Standard self-attention and text cross-attention use PyTorch SDPA, allowing CUDA Flash/memory-efficient attention kernels when hardware, dtype, masks, and shapes support them.
- Optional non-reentrant activation checkpointing trades recompute for lower activation memory.
- `anima36.optim.make_adamw` enables fused AdamW on CUDA when supported; `compile_model` wraps `torch.compile`; `configure_cuda_performance` enables TF32 for CUDA matmul/conv.
- Measure on target hardware: compile warmup, dynamic shapes, LSConv unfold memory, and fallback SDPA backends can change performance. No speedup claim without representative GPU benchmarks.

## Target design
- 36 DiT blocks: repeat 3 LSConv-enhanced linear-attention blocks, then 1 standard Softmax self-attention block (27 + 9).
- LSNet-style Large Kernel Perception + Small Kernel Aggregation in a portable PyTorch reference; not yet numerically matched to official LSNet Triton.
- RMSNorm, timestep modulation, SwiGLU, and an experimental learned Block AttnRes prototype.
- Qwen3.5-4B hidden-state routing target and Qwen-Image-2.1 VAE target; exact configs and end-to-end integration remain to be verified.

## Layout
- `configs/anima36.yaml`: initial config.
- `docs/architecture.md`: architecture and integration contracts.
- `src/anima36/`: PyTorch modules, core model, optimization helpers.
- `tests/`: schedule, module forward/backward, and small model tests.
- `.github/workflows/pytorch-tests.yml`: CPU PyTorch CI.

## CI
[View GitHub Actions](https://github.com/haotemans/anima-36/actions/workflows/pytorch-tests.yml).

## Integration references
- https://github.com/modelscope/DiffSynth-Studio
- https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_training/train.py
- https://github.com/THU-MIG/lsnet

Before training, verify LSConv against official LSNet, validate VAE conventions, implement RoPE, and integrate text encoder/pipeline APIs against an exact DS revision.

## License
No license selected yet. Review upstream licenses before copying or redistributing code.
