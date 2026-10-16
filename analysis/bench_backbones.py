#!/usr/bin/env python3
"""Benchmark candidate SMALL backbones on CPU (2 threads) vs SigLIP2 B/16."""
import os
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(2)

CANDIDATES = [
    ("siglip2_b16", "vit_base_patch16_siglip_224.v2_webli", 224),
    ("dinov2_s14", "vit_small_patch14_dinov2.lvd142m", 224),
    ("resnet18", "resnet18.a1_in1k", 224),
    ("resnet34", "resnet34.a1_in1k", 224),
    ("mobilenetv3", "mobilenetv3_large_100.ra_in1k", 224),
    ("efficientnet_b0", "efficientnet_b0.ra_in1k", 224),
]


def main():
    import timm
    print(f"{'name':16s} {'params M':>9s} {'ms/img':>8s} {'FPS':>6s}")
    for name, model_name, size in CANDIDATES:
        try:
            m = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=size)
            m.eval()
            params = sum(p.numel() for p in m.parameters()) / 1e6
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                m(x)
                best = 1e9
                for _ in range(3):
                    t0 = time.perf_counter()
                    for _ in range(5):
                        m(x)
                    best = min(best, (time.perf_counter() - t0) / 5 * 1000)
            print(f"{name:16s} {params:9.1f} {best:8.1f} {1000/best:6.1f}", flush=True)
        except Exception as e:
            print(f"{name:16s} FAIL {str(e)[:60]}")


if __name__ == "__main__":
    main()
