# Anima-200M Pixel Art Experiment

## Goal

A roughly 200M-parameter, 512x512 text-to-image research experiment trained on locally sourced Danbooru posts tagged `pixel_art`. This is a new configuration alongside the existing Anima-36-scale configuration; it does not replace `configs/anima36.yaml`.

## Model configuration

- 14 identical hybrid blocks, width 640, 10 heads (64 dimensions per head).
- Each block: positive-feature Linear Attention + LSNet-style local branch, separate text cross-attention, SwiGLU FFN, and the repository's experimental AttnRes.
- The timestep embedding is computed once; each block predicts independent shift/scale/residual-gate values for its three sublayers.
- 16 latent channels, patch size 2, FFN hidden width 2560, text-context projection width 1024.
- Parameter budget estimate: about 197M parameters under the current module definitions. This is a formula-based estimate, not yet a measured instantiation count. Run `python scripts/count_parameters.py --config configs/anima200m.yaml` before training and adjust depth/width if module definitions have changed.

## Qwen-Image VAE compatibility gate

The candidate is `Qwen/Qwen-Image`, specifically its `vae/config.json`. The public config identifies `AutoencoderKLQwenImage` and `z_dim=16`; the config also carries 16-channel `latents_mean` and `latents_std`. The Qwen-Image pipeline uses a VAE spatial scale of 8 and then a 2x2 latent patch packing convention. For this repository's convolutional patch embed, 512x512 RGB images should therefore yield a 16x64x64 latent tensor and a 32x32 transformer token grid if the selected VAE implementation matches that pipeline revision.

**Not yet proven compatible.** Before training, run an integration test against the exact Diffusers/Qwen-Image revision and verify:
1. encode/decode input and output shapes for 512x512 images;
2. the latent distribution choice (sample vs mode);
3. per-channel normalization using the VAE config's means and standard deviations;
4. numerical finiteness and a decode/reconstruction sanity check;
5. text encoder feature width, mask convention, and projection to 1024;
6. the training scheduler's target convention. The `flow` value in YAML is provisional, not a guarantee that a generic scheduler can be used with this VAE.

Do not treat the fact that both systems use 16 latent channels as sufficient compatibility evidence. Qwen-Image is not automatically a drop-in VAE for a DiffSynth-Studio Anima checkpoint.

## Dataset policy and manifest

Use only local files/metadata that you are permitted to process. The manifest builder accepts CSV or JSONL with a post ID, a local image path, tags, and an optional rating. It selects the exact `pixel_art` tag and defaults to safe-rated posts only. The optional style tags in YAML are audit/analysis hints, not mandatory filters.

- Exclude deleted, inaccessible, or missing image files.
- Deduplicate by post ID and SHA-256; keep a stable train/validation split by post ID.
- Use nearest-neighbor resizing and padding to square rather than antialiased resampling, which can blur hard pixel boundaries.
- Keep original attribution and source IDs in the manifest. Review Danbooru terms, creator-specific licensing, opt-outs, and applicable copyright/privacy requirements. Public availability is not permission to redistribute or train on every image.
- Inspect a random sample and tag precision before large-scale download or training.

Example:
```bash
python scripts/build_danbooru_pixel_art_manifest.py \
  --input /data/danbooru/posts.jsonl \
  --image-root /data/danbooru/images \
  --output /data/manifests/pixel_art.jsonl
```

Input fields accepted by the builder: `id` or `post_id`; `file_path` or `path`; `tags` as a space/comma-separated string or a JSON list; optional `rating`. It writes only manifest metadata, not copied images.

## First acceptance gates

1. Instantiate the configured model and record the actual parameter count.
2. Run unit tests and forward/backward smoke tests at a 32x32 token grid.
3. Verify all 14 blocks use the hybrid operator.
4. Run the exact VAE and text-encoder integration tests above.
5. Overfit a small, rights-cleared subset before a longer run.
6. Record GPU model, peak memory, throughput, loss curve, sample grids, and checkpoints. Do not claim image quality or training stability before those results exist.
