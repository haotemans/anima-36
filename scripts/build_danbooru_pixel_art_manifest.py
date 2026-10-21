#!/usr/bin/env python3
"""Build a deterministic metadata manifest for Danbooru pixel-art training.

Accepted input: CSV or JSONL. Required columns/keys: id or post_id, file_path
or path, tags. Optional: rating. This script never downloads images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
                if isinstance(item, dict):
                    yield item
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)


def tags_of(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(tag).strip().lower() for tag in value if str(tag).strip()}
    if value is None:
        return set()
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return {str(tag).strip().lower() for tag in parsed if str(tag).strip()}
        except json.JSONDecodeError:
            pass
    return {tag.lower() for tag in text.replace(",", " ").split() if tag}


def first(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_split(post_id: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{post_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV or JSONL metadata")
    parser.add_argument("--image-root", required=True, type=Path, help="Root for local image paths")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL manifest")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--allowed-ratings", nargs="+", default=["s"])
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    counts = {"input": 0, "missing_fields": 0, "wrong_tag": 0, "wrong_rating": 0,
              "missing_file": 0, "duplicate_id": 0, "duplicate_hash": 0, "written": 0}
    with args.output.open("w", encoding="utf-8") as output:
        for row in records(args.input):
            counts["input"] += 1
            post_id = first(row, "post_id", "id")
            relative_path = first(row, "file_path", "path")
            tag_value = row.get("tags")
            if not post_id or not relative_path or tag_value is None:
                counts["missing_fields"] += 1
                continue
            tags = tags_of(tag_value)
            if "pixel_art" not in tags:
                counts["wrong_tag"] += 1
                continue
            rating = first(row, "rating").lower()
            if rating not in {r.lower() for r in args.allowed_ratings}:
                counts["wrong_rating"] += 1
                continue
            image_path = Path(relative_path)
            if not image_path.is_absolute():
                image_path = args.image_root / image_path
            if not image_path.is_file():
                counts["missing_file"] += 1
                continue
            if post_id in seen_ids:
                counts["duplicate_id"] += 1
                continue
            digest = sha256_file(image_path)
            if digest in seen_hashes:
                counts["duplicate_hash"] += 1
                continue
            seen_ids.add(post_id)
            seen_hashes.add(digest)
            record = {
                "post_id": post_id,
                "file_path": str(image_path.resolve()),
                "sha256": digest,
                "rating": rating,
                "tags": sorted(tags),
                "split": stable_split(post_id, args.seed, args.validation_fraction),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["written"] += 1
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
