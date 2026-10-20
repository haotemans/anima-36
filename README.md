# Anima-36

Experimental PyTorch diffusion-transformer prototype targeting integration with [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

**Status: core prototype implemented; CPU CI tests pass. Not trained, not yet a complete DiffSynth-Studio pipeline, and not validated against model weights.** No weights, tokenizer, or VAE assets are included.

## Target design
- 36 DiT blocks: repeat 3 LSConv-enhanced linear-attention blocks, then 1 standard Softmax self-attention block (27 + 9).
- LSNet-style Large Kernel Perception + Small Kernel Aggregation in a portable PyTorch implementation; numerics have not yet been compared against the official [THU-MIG/LSNet](https://github.com/THU-MIG/lsnet) Triton kernel.
- SwiGLU FFN, RMSNorm pre-norm and Q/K RMSNorm, timestep/AdaLN-style modulation.
- Explicit learned Block AttnRes prototype: softmax-weighted aggregation of recent normalized states with a gated residual. It is an experimental equation, not claimed paper parity.
- Qwen3.5-4B multi-layer hidden states, normalized/projected independently and fused through dedicated text cross-attention. The legacy LLMAdapter path is not present in this new standalone core.
- Qwen-Image-2.1 VAE target; latent channels, spatial scale, and scaling/shift convention still require verification.

## Layout
- `configs/anima36.yaml`: initial config.
- `docs/architecture.md`: architecture and integration contracts.
- `src/anima36/`: PyTorch modules and configurable `Anima36Core`.
- `tests/`: schedule, module forward/backward, and small end-to-end core tests.
- `.github/workflows/pytorch-tests.yml`: CPU PyTorch CI.

## CI
GitHub Actions: [view test runs](https://github.com/haotemans/anima-36/actions/workflows/pytorch-tests.yml). The latest verified run reported **9 passed**.

## DS integration references
- https://github.com/modelscope/DiffSynth-Studio
- https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_training/train.py
- https://github.com/THU-MIG/lsnet

Before real training, verify LSConv against official LSNet, validate VAE conventions, implement/validate RoPE and checkpoint conversion, and integrate the text encoder and pipeline APIs against an exact DS revision.

## License
No license selected yet. Review upstream licenses before copying or redistributing code.
