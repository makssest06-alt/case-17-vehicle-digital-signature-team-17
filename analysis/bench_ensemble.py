#!/usr/bin/env python3
"""Benchmark embedding inference: single vs 3-model vs 4-model ensemble.

CPU-only, 2 threads (to mimic the organizers' 2-vCPU stand).
Measures per-image latency, throughput (FPS) at batch 1/8/16, peak RSS.
"""
import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from extract_emb import crop_image, load_model
from train_reid import build_transforms

HERE = Path(__file__).parent
IMAGES = Path(r"D:\bm\dataset_extracted\images")
TESTQ = Path(r"D:\bm\dataset_extracted\test_query.csv")
MODELS = Path(r"D:\bm_improve\models")
EXTRA = Path(r"D:\bm\improve_models")


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        return float("nan")


def load_imgs(n):
    rows = list(csv.DictReader(TESTQ.open(encoding="utf-8")))[:n]
    return [crop_image(IMAGES / f"{r['image_id']}.jpg", r, 0.10) for r in rows]


@torch.no_grad()
def embed_batch(loaded, imgs, device, tta):
    """loaded = list of (model, weight, tf). Returns weighted-sum features."""
    feats = None
    for model, w, tf in loaded:
        x = torch.stack([tf(im) for im in imgs]).to(device)
        _, fb = model(x)
        f = fb.float()
        if tta:
            xf = torch.stack([tf(im.transpose(1)) for im in imgs]).to(device)
            _, fb2 = model(xf)
            f = f + fb2.float()
        f = w * f
        feats = f if feats is None else feats + f
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    device = torch.device("cpu")

    ck = {
        "siglip2": MODELS / "siglip2_reid_scratch30_ep28.pt",
        "sq30": MODELS / "siglip_reid_scratch30_ep26.pt",
        "ep29": MODELS / "siglip_vitb16_reid_ep29.pt",
        "clip": EXTRA / "clip_reid_scratch30_ep28.pt",
    }
    TFM = {"siglip2": "squash", "sq30": "squash", "ep29": "timm", "clip": "squash"}

    configs = {
        "single_siglip2": (["siglip2"], [1.0], False),
        "team3": (["siglip2", "sq30", "ep29"], [0.45, 0.25, 0.30], False),
        "ens4": (["siglip2", "sq30", "ep29", "clip"], [0.364, 0.232, 0.182, 0.222], False),
        "ens4_tta": (["siglip2", "sq30", "ep29", "clip"], [0.364, 0.232, 0.182, 0.222], True),
    }

    imgs = load_imgs(a.n)
    print(f"threads={a.threads} images={len(imgs)}", flush=True)

    results = {}
    for name, (keys, weights, tta) in configs.items():
        gc.collect()
        rss0 = rss_mb()
        t0 = time.perf_counter()
        loaded = []
        for k, w in zip(keys, weights):
            m, spec, _ = load_model(ck[k], device)
            loaded.append((m, w, build_transforms(spec, TFM[k], train=False)))
        load_s = time.perf_counter() - t0
        rss1 = rss_mb()

        embed_batch(loaded, imgs[:2], device, tta)  # warmup

        lat = []
        for im in imgs:
            t = time.perf_counter()
            embed_batch(loaded, [im], device, tta)
            lat.append((time.perf_counter() - t) * 1000)
        lat = np.array(lat)

        fps = {}
        for bs in (1, 8, 16):
            chunk = imgs[:bs]
            t = time.perf_counter()
            embed_batch(loaded, chunk, device, tta)
            fps[bs] = round(len(chunk) / (time.perf_counter() - t), 2)

        rss2 = rss_mb()
        results[name] = dict(load_s=round(load_s, 1), p50=round(float(np.median(lat)), 1),
                             p95=round(float(np.percentile(lat, 95)), 1),
                             fps1=fps[1], fps8=fps[8], fps16=fps[16],
                             rss_start=round(rss0), rss_load=round(rss1), rss_peak=round(rss2))
        print(f"{name:16s} load {load_s:5.1f}s  p50 {results[name]['p50']:7.1f}ms  "
              f"p95 {results[name]['p95']:7.1f}ms  FPS(1/8/16) "
              f"{fps[1]}/{fps[8]}/{fps[16]}  RSS {rss0:.0f}->{rss1:.0f}->{rss2:.0f} MB",
              flush=True)
        del loaded
        gc.collect()

    (HERE / "bench_ensemble.json").write_text(json.dumps(results, indent=2))
    print("saved bench_ensemble.json")


if __name__ == "__main__":
    main()
