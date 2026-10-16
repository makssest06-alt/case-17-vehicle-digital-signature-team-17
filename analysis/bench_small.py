#!/usr/bin/env python3
"""Benchmark small CNN/ViT backbones on CPU (2 threads)."""
import os
import time

import torch

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(2)

CANDS = [
    ("resnet18", "resnet18.a1_in1k", 224),
    ("resnet34", "resnet34.a1_in1k", 224),
    ("mobilenetv3", "mobilenetv3_large_100.ra_in1k", 224),
    ("efficientnet_b0", "efficientnet_b0.ra_in1k", 224),
    ("dinov2_s14_336", "vit_small_patch14_dinov2.lvd142m", 336),
]


def main():
    import timm
    print(f"{'name':18s} {'paramsM':>8s} {'ms/img':>8s} {'FPS':>6s}")
    for name, mn, sz in CANDS:
        try:
            try:
                m = timm.create_model(mn, pretrained=False, num_classes=0, img_size=sz)
            except TypeError:
                m = timm.create_model(mn, pretrained=False, num_classes=0)
            m.eval()
            p = sum(x.numel() for x in m.parameters()) / 1e6
            x = torch.randn(1, 3, sz, sz)
            with torch.no_grad():
                m(x)
                best = 1e9
                for _ in range(3):
                    t0 = time.perf_counter()
                    for _ in range(5):
                        m(x)
                    best = min(best, (time.perf_counter() - t0) / 5 * 1000)
            print(f"{name:18s} {p:8.1f} {best:8.1f} {1000/best:6.1f}", flush=True)
        except Exception as e:
            print(f"{name:18s} FAIL {str(e)[:70]}")


if __name__ == "__main__":
    main()
