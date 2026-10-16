#!/usr/bin/env python3
"""Benchmark the distilled student vs baselines on CPU (2 threads, like service)."""
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

IMAGES = Path(r"D:\bm\dataset_extracted\images")
TESTQ = Path(r"D:\bm\dataset_extracted\test_query.csv")

torch.set_num_threads(2)
device = torch.device("cpu")

CK = {
    "single_siglip2": (r"D:\bm_improve\models\siglip2_reid_scratch30_ep28.pt", "squash"),
    "student_a25": (r"D:\bm\improve_models\siglip2_distill_a25_ep26.pt", "squash"),
    "student_a12": (r"D:\bm\improve_models\siglip2_distill_a12_ep26.pt", "squash"),
}
N = 15


def rss():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


rows = list(csv.DictReader(TESTQ.open(encoding="utf-8")))[:N]
imgs = [crop_image(IMAGES / f"{r['image_id']}.jpg", r, 0.10) for r in rows]

res = {}
for name, (path, tfm) in CK.items():
    gc.collect()
    m, spec, _ = load_model(path, device)
    tf = build_transforms(spec, tfm, train=False)
    rss_load = rss()
    with torch.no_grad():
        m(torch.stack([tf(imgs[0])]))  # warmup
        # per-image (batch=1), no TTA
        t0 = time.perf_counter()
        for im in imgs:
            m(torch.stack([tf(im)]))
        per1 = (time.perf_counter() - t0) / len(imgs) * 1000
        # batch=8
        t0 = time.perf_counter()
        m(torch.stack([tf(im) for im in imgs[:8]]))
        fps8 = 8 / (time.perf_counter() - t0)
    res[name] = dict(per_image_ms=round(per1, 1), fps_b1=round(1000 / per1, 2),
                     fps_b8=round(fps8, 2), rss_mb=round(rss_load))
    print(f"{name:16s} {per1:7.1f} ms/img  {1000/per1:5.2f} FPS (b=1)  "
          f"{fps8:5.2f} FPS (b=8)  RAM {rss_load:.0f} MB", flush=True)
    del m
    gc.collect()

Path(r"D:\bm_improve\analysis\bench_student.json").write_text(json.dumps(res, indent=2))
print("saved")
