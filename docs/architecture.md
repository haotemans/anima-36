# Anima-36 Architecture Specification

## Status
This document specifies the target design, not a claim that a runnable model exists. Verify exact upstream APIs, model configs, VAE conventions, and LSConv implementation before integration.

## Topology
- 36 blocks; hidden size 2048 and 16 heads (head dimension 128) are initial assumptions.
- Alternate LS-Linear and Standard Self-Attention throughout the network.
- Total: 18 LS-Linear blocks and 18 Standard blocks.
- Rationale: distribute global token mixing across the full depth instead of placing it only once every four blocks. This is a testable hypothesis, not a proven quality improvement; compare against the previous 27/9 schedule under matched training conditions.

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


## Feasibility audit (2026-10)

### Important target distinction

There are two different goals and they must not be conflated:

1. **DiffSynth-Studio Anima compatibility.** The current upstream `diffsynth/models/anima_dit.py` implementation defines `AnimaDiT` with 28 transformer blocks, 16 input/output latent channels, spatial patch size 2, model width 2048, 16 heads, cross-attention context width 1024, 3D RoPE, AdaLN-LoRA, and an LLMAdapter. Its current `AnimaImagePipeline` uses a `ZImageTextEncoder`, a separate T5 tokenizer/IDs path, and `WanVideoVAE`. The upstream pipeline uses a FlowMatch scheduler and predicts a denoising/flow target.
2. **A new Anima-inspired 36-block model.** This repository's 36-block schedule, LS-Linear layers, Block AttnRes prototype, Qwen3.5-4B target, and Qwen-Image-2.1 VAE target are a distinct proposed architecture. They are not drop-in compatible with upstream Anima weights or pipeline unless an explicit conversion/adaptation is designed and validated.

Upstream references (verify against a pinned commit before implementation):
- https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/anima_dit.py
- https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/anima_image.py
- https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_training/train.py

### Dataflow implemented by this repository

1. Input latents `[B, C_latent, H, W]` enter a stride-`patch_size` Conv2d and become `[B, N, D]`, where `N=(H/patch_size)(W/patch_size)`.
2. A scalar timestep per batch item is converted to a sinusoidal embedding and projected to `[B,D]`.
3. Each block applies timestep-modulated image self-attention, separate text cross-attention, an unmodulated pre-norm SwiGLU FFN, and learned aggregation from recent block outputs.
4. Final timestep modulation and a transposed convolution map tokens back to `[B,C_latent,H,W]`.

This describes the current code path, not the full diffusion system. Text encoding/tokenization, VAE encode/decode and latent scaling, noise/sigma schedule, target parameterization, loss, classifier-free guidance, sampler, checkpoint conversion, and an end-to-end pipeline are not implemented here.

### Blocking feasibility issues

- **Text routing is not connected.** `LayerwiseTextProjector` and `stage_features_for_block` are standalone utilities. `Anima36Core.forward` accepts one `text_context` tensor and passes the same context to every block; it does not select shallow/middle/deep hidden states per block. The current staged-routing claim is therefore a design intention, not implemented behavior.
- **Position information is missing.** The config explicitly sets `rope: false`, and neither standard attention nor LS-Linear attention applies RoPE or another explicit spatial position embedding. LSConv provides local inductive bias but does not replace a complete positional encoding contract for global attention. This is a blocker for target-model parity and a major risk for spatial composition.
- **LS-Linear positional semantics are undefined.** K/V are average-pooled spatially, then used in positive-feature linear attention. The code does not define a positional encoding treatment for compressed K or demonstrate that this operator preserves the required spatial relationships.
- **Conditioning is inconsistent across sublayers.** Self-attention and text cross-attention receive timestep modulation; the FFN only receives RMSNorm and has no timestep modulation. This may be a valid custom design, but it differs from the stated target block contract and needs an explicit decision.
- **Latent contract is unknown.** The config leaves latent channels, spatial scale, scaling factor, and shift factor null. The model defaults to 16 channels, but a default is not proof that the selected VAE emits compatible latents.
- **Text encoder contract is unknown.** The code assumes one `context_dim` and one mask. Qwen3.5-4B hidden-state indices, feature dimensions, tokenizer behavior, model output API, and license have not been verified; a separate T5 path in upstream Anima is not represented.
- **Training objective and sampler are absent.** Without a specified noise/sigma schedule, prediction target (epsilon, velocity, or flow), loss weighting, and matching sampler, a forward pass alone cannot establish diffusion-model feasibility.
- **Checkpoint compatibility is not established.** The 36-vs-28 depth, attention operators, positional encoding, FFN, normalization, conditioning, and text adapter differ from upstream. Weight loading requires an explicit mapping and mismatch report; shape-compatible loading alone is insufficient.
- **AttnRes is a prototype.** Its learned source weights are shared across tokens and use recent states; this is not validated as equivalent to the intended reference mechanism. The implementation also retains a growing list of all block outputs, so activation-memory impact must be measured.
- **LSConv equivalence is unverified.** The local implementation's padding, normalization, weight layout and group sharing must be checked against the exact official LSNet implementation. The optional Triton path needs actual CUDA compilation and forward/backward parity before being treated as usable.

### Recommended acceptance gates

**Gate A — architecture specification (before performance tuning)**
- Choose whether the goal is upstream checkpoint/pipeline compatibility or a new 36-block model.
- Pin exact upstream revisions and model IDs.
- Specify tensor shapes, spatial position encoding, timestep modulation, text features/masks, latent scaling, prediction target, and scheduler/loss.
- Resolve the latent channel and text-context dimensions from actual model configs, not assumptions.

**Gate B — tiny mathematical prototype**
- Test multiple non-square grids, patch sizes, batches, and dtypes.
- Add positional encoding and test translation/spatial sensitivity.
- Test text padding invariance and prove staged hidden-state routing actually changes the per-block context.
- Compare checkpointed/eager outputs and all parameter gradients.
- Check finite outputs/gradients under BF16 and FP32.

**Gate C — reference parity**
- Compare LSConv output and gradients with the pinned official LSNet kernel.
- Compare VAE encode/decode latent conventions and text encoder outputs with the selected upstream components.
- Load a checkpoint with a complete missing/unexpected/shape-mismatch report.

**Gate D — end-to-end diffusion**
- Implement training data flow, noise/sigma scheduler, prediction parameterization, loss, sampler, and decode.
- Overfit a tiny dataset or synthetic batch as a basic learnability check before scaling.
- Only after these gates pass should performance benchmarking be treated as meaningful.

### Current conclusion

The repository has a runnable small-shape transformer scaffold, but **the target architecture is not yet fully specified or validated as a diffusion model**. The highest-priority blockers are the unresolved target (upstream-compatible vs new architecture), missing positional encoding, disconnected layerwise text routing, undefined latent/text contracts, and missing training/sampling objective. Kernel optimization should wait until these are resolved.


## Position encoding implementation (2026-10)

- Standard self-attention applies axial 2D RoPE to Q/K using the patch grid passed by the core model. The head dimension must be divisible by four.
- LS-Linear uses deterministic 2D sinusoidal absolute embeddings on token features before Q/K/V projection. RoPE is intentionally not applied to its positive ELU+1 kernel features, because arbitrary rotations would break the nonnegative feature-map formulation.
- Text cross-attention remains without image-side RoPE in this first implementation.
- This is a custom architecture choice, not checkpoint parity with upstream Anima. Weights trained without these encodings are not expected to be behaviorally compatible without retraining or explicit validation.
- Unit tests cover shape, finite values, gradients, and mismatched grid sizes. Run the full CPU suite and accelerator tests before treating this as validated; unit tests do not establish end-to-end image quality.
