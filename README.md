# Anima-36

Experimental PyTorch diffusion-transformer prototype targeting integration with [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

**Status: experimental core only.** The architecture and data utilities are not a complete, end-to-end training pipeline. No model has been trained and no image-quality claim is made.

## Current architecture

- Every transformer block combines positive-feature Linear Attention with an LSNet-style local branch; blocks no longer alternate between hybrid and full-softmax attention.
- Separate text cross-attention injects text tokens; text is not sent through LSConv.
- The timestep embedding is computed once per model and each block predicts independent AdaLN shift/scale/residual gates for hybrid attention, text cross-attention, and SwiGLU FFN.
- LSConv is a portable PyTorch LKP + SKA reference implementation, not yet numerically matched to official LSNet Triton.
- Block AttnRes is an experimental learned aggregation prototype, not a claim of exact parity with an external implementation.

## ~200M pixel-art experiment

- `configs/anima200m.yaml`: 14 blocks, width 640, 10 heads, SwiGLU width 2560, 512x512 training target, and a formula-based estimate of about 197M parameters.
- `docs/anima200m-pixel-art.md`: architecture budget, Qwen-Image VAE compatibility checks, dataset policy, and acceptance gates.
- `scripts/count_parameters.py --config configs/anima200m.yaml`: instantiate the actual module definitions and print the measured parameter count.
- `scripts/build_danbooru_pixel_art_manifest.py`: filter local CSV/JSONL metadata to exact `pixel_art` tag matches, deduplicate local files, and create a stable train/validation manifest. It does not download images or resize them.

The selected VAE target is `Qwen/Qwen-Image-2.1` (`AutoencoderKLQwenImage21`): its published config specifies 64 latent channels and 16x spatial compression. It is not yet integration-verified; validate its 64-channel per-channel mean/std normalization, encode/decode behavior, and text-conditioning interface against a pinned Diffusers revision before training.

## Layout

- `configs/anima36.yaml`: larger 36-block baseline configuration.
- `configs/anima200m.yaml`: ~200M pixel-art experiment.
- `docs/architecture.md`: architecture specification and integration contracts.
- `docs/anima200m-pixel-art.md`: experiment-specific plan and compatibility gates.
- `src/anima36/`: PyTorch modules and core model.
- `scripts/`: parameter-count and local dataset-manifest utilities.
- `tests/`: CPU module and model tests.
- `.github/workflows/pytorch-tests.yml`: CPU PyTorch CI.

## Validation status

The configuration and utilities have been committed, but this change has not been run through pytest, a model instantiation, a VAE round-trip, or a training smoke test in this session. Run the parameter-count script and test suite in the repository environment before starting a training job.

## Integration references

- https://github.com/modelscope/DiffSynth-Studio
- https://github.com/THU-MIG/lsnet

Before training, verify LSConv against official LSNet, validate VAE conventions, and integrate text encoder/pipeline APIs against an exact DiffSynth-Studio revision.

## License

No license selected yet. Review upstream licenses and artwork rights before copying, training on, or redistributing code and data.
