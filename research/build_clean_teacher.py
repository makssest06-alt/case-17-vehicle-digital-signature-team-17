#!/usr/bin/env python3
"""Build a train-only weighted teacher embedding ensemble."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TRAINING = HERE / "finals_handoff" / "training"
sys.path.insert(0, str(TRAINING))

from extract_emb import embed_rows, load_model  # noqa: E402
from train_reid import build_transforms  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    rows = read_rows(args.rows)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = np.zeros((len(rows), 512), dtype=np.float64)
    for item in config:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, spec, _ = load_model(item["checkpoint"], device)
        transform = build_transforms(spec, item["tf_mode"], train=False)
        embedding = embed_rows(
            model, transform, rows, args.images_dir, 0.10, device,
            tta=bool(item.get("tta", True)), bs=args.batch_size,
        )
        result += float(item["weight"]) * l2(embedding.astype(np.float64))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"added {item['name']}", flush=True)
    result = l2(result).astype("float32")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, result)
    print(f"saved {args.out}: {result.shape}")


if __name__ == "__main__":
    main()
