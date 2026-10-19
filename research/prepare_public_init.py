#!/usr/bin/env python3
"""Convert public timm state_dict into the local trainer checkpoint format."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tf-mode", choices=("timm", "squash"), required=True)
    parser.add_argument("--input-size", type=int, default=224,
                        help="родное разрешение бэкбона (siglip_large — 256)")
    args = parser.parse_args()

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    source = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(source, dict):
        raise ValueError("source checkpoint must contain a state dict")
    model = {f"encoder.{key}": value for key, value in source.items()}
    metadata = {
        "backbone": args.backbone,
        "out_dim": 512,
        "hidden": 1024,
        "dropout": 0.3,
        "loss": "arcface",
        "seed": args.seed,
        "tf_mode": args.tf_mode,
        "input_size": args.input_size,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model, "args": metadata}, args.out)
    print(f"saved {args.out} with {len(model)} encoder tensors")


if __name__ == "__main__":
    main()
