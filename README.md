# Anima-36

Experimental Anima-inspired diffusion-transformer architecture scaffold for integration with [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

**Status: scaffold only, not yet a runnable or trained model.** No model weights or tokenizer/VAE assets are included.

## Target design
- 36 DiT blocks: repeat 3 LSConv-enhanced linear-attention blocks, then 1 standard Softmax self-attention block (27 + 9).
- LSConv faithful to the official [THU-MIG/LSNet](https://github.com/THU-MIG/lsnet) Large Kernel Perception + Small Kernel Aggregation design, not a generic depthwise convolution.
- SwiGLU in every FFN; RMSNorm pre-norm and Q/K RMSNorm; retain timestep/AdaLN-style conditioning.
- Explicit Block AttnRes; not equivalent to an ordinary residual addition.
- Qwen3.5-4B multi-layer hidden states, normalized/projected independently and fused through dedicated text cross-attention. Remove the legacy LLMAdapter path in the eventual DS integration.
- Qwen-Image-2.1 VAE; verify latent channels, spatial scale, and scaling/shift convention against the exact target config.

## Layout
- `configs/anima36.yaml`: initial config.
- `docs/architecture.md`: architecture and implementation contracts.
- `src/anima36/`: standalone modules and explicit placeholders.
- `tests/`: schedule contract tests.

## DS integration references
- https://github.com/modelscope/DiffSynth-Studio
- https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_training/train.py
- https://github.com/THU-MIG/lsnet

Check the exact upstream revision and source APIs before integration. No end-to-end inference or training should be claimed until tests pass.

## License
No license selected yet. Review upstream licenses before copying or redistributing code.
