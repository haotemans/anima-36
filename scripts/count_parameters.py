#!/usr/bin/env python3
"""Instantiate an Anima experiment config and report its exact parameter count."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from anima36.model import Anima36Core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model_cfg = cfg["model"]
    model = Anima36Core(
        latent_channels=int(model_cfg["latent_channels"]),
        dim=int(model_cfg["hidden_size"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        context_dim=int(model_cfg["context_dim"]),
        patch_size=int(model_cfg["patch_size"]),
        ffn_dim=int(model_cfg["ffn_hidden_size"]),
        kv_stride=int(model_cfg.get("kv_stride", 2)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    target = int(model_cfg.get("target_parameters", 0))
    print(f"config: {args.config}")
    print(f"total_parameters: {total:,} ({total / 1e6:.3f}M)")
    print(f"trainable_parameters: {trainable:,} ({trainable / 1e6:.3f}M)")
    if target:
        print(f"target_parameters: {target:,} ({target / 1e6:.3f}M)")
        print(f"relative_error: {(total - target) / target:+.2%}")


if __name__ == "__main__":
    main()
