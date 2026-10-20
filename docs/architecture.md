# Anima-36 Architecture Specification

## Status
This document specifies the target design, not a claim that a runnable model exists. Verify exact upstream APIs, model configs, VAE conventions, and LSConv implementation before integration.

## Topology
- 36 blocks; hidden size 2048 and 16 heads (head dimension 128) are initial assumptions.
- Blocks 1–3 LS-Linear, block 4 Standard; repeat every four blocks through block 36.
- Total: 27 LS-Linear blocks and 9 Standard blocks.

## LS-Linear contract
1. Preserve image token grid metadata for reshape to/from `[B, C, H, W]`.
2. Apply pre-norm and timestep/AdaLN-style conditioning.
3. Project Q/K/V and apply Q/K RMSNorm.
4. Port official LSNet LSConv (Large Kernel Perception + Small Kernel Aggregation); do not substitute generic depthwise convolution.
5. Keep Q full-resolution and spatially compress K/V.
6. Specify the linear-attention equation, normalization, masking, and RoPE treatment for compressed K; test rather than assume.
7. Apply output projection.
8. Fuse text through separate cross-attention; never send text tokens through image LSConv or image K/V compression.
9. Apply RMSNorm + SwiGLU FFN.
10. Apply explicitly defined Block AttnRes aggregation.

## Standard block contract
Use standard Softmax image self-attention, consistent RMSNorm/QK normalization, RoPE, timestep conditioning, dedicated text cross-attention, SwiGLU, and the shared Block AttnRes contract.

## Text conditioning
- Target encoder: `Qwen/Qwen3.5-4B`.
- Encode once and expose selected hidden states.
- Normalize and project shallow/middle/deep features independently.
- Initial mapping: blocks 1–12 use middle+deep; 13–24 use shallow+middle+deep; 25–36 use shallow+deep.
- Use dedicated cross-attention and propagate padding masks.
- Remove legacy LLMAdapter module and all call paths in the integrated DS implementation.
- Confirm tokenizer, hidden-state indexing, dimensions, dtype, and model license/config.

This layer mapping is a starting hypothesis, not a validated optimum.

## Block AttnRes
Initial grouping boundary: four blocks. Specify source-state eligibility, normalization, aggregation-weight computation, gating, insertion point, checkpointing, and memory retention. A plain skip addition is not Block AttnRes.

## Normalization and conditioning
Use RMSNorm for main pre-norm and Q/K paths. Keep timestep/AdaLN conditioning; RMSNorm does not replace conditional modulation. Preserve VAE-native normalization. Report missing, unexpected, and shape-mismatched tensors during checkpoint conversion.

## VAE
Target Qwen-Image-2.1 VAE. Verify latent channels, spatial downsampling factor, latent scaling/shift, and exact implementation before updating DiT input/output layers and pipeline encode/decode. Do not infer compatibility from a filename.

## Rough parameter estimate
With D=2048, SwiGLU width F=8192, four D×D attention projections and three SwiGLU projections:
- attention weights ≈ 4D² = 16.78M/block;
- SwiGLU weights ≈ 3DF = 50.33M/block;
- 36-block baseline ≈ 2.42B weights.

This excludes LSConv, compressed K/V projections, cross-attention, AttnRes, input/output heads and biases. It is a rough baseline, not the final count.

## DiffSynth-Studio integration
Inspect the exact target revision of `diffsynth/models/anima_dit.py`, `diffsynth/pipelines/anima_image.py`, model loading/registration, and `examples/anima/model_training/`. Keep modules isolated until interfaces are verified.

## Acceptance criteria
- Exact 36-position attention schedule test.
- Shape tests across supported resolutions and batch sizes.
- LSConv output/gradient/dtype tests.
- Attention finite-output/gradient and mask tests.
- Text layer routing and padding-mask tests.
- VAE latent convention tests.
- Explicit checkpoint mismatch report.
- End-to-end pipeline test before claiming inference support.
