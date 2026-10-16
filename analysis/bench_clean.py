#!/usr/bin/env python3
"""Clean end-to-end benchmark: total time for N images, per-image, per config."""
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
MODELS = Path(r"D:\bm_improve\models")
EXTRA = Path(r"D:\bm\improve_models")

torch.set_num_threads(2)
device = torch.device("cpu")

CK = {
    "siglip2": (MODELS / "siglip2_reid_scratch30_ep28.pt", "squash"),
    "sq30": (MODELS / "siglip_reid_scratch30_ep26.pt", "squash"),
    "ep29": (MODELS / "siglip_vitb16_reid_ep29.pt", "timm"),
    "clip": (EXTRA / "clip_reid_scratch30_ep28.pt", "squash"),
}
CONFIGS = {
    "single_siglip2": (["siglip2"], False),
    "team3": (["siglip2", "sq30", "ep29"], False),
    "ens4": (["siglip2", "sq30", "ep29", "clip"], False),
    "ens4_tta": (["siglip2", "sq30", "ep29", "clip"], True),
}
N = 15


def rss():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


rows = list(csv.DictReader(TESTQ.open(encoding="utf-8")))[:N]
imgs = [crop_image(IMAGES / f"{r['image_id']}.jpg", r, 0.10) for r in rows]

res = {}
for name, (keys, tta) in CONFIGS.items():
    gc.collect()
    loaded = []
    for k in keys:
        path, tfm = CK[k]
        m, spec, _ = load_model(path, device)
        loaded.append((m, build_transforms(spec, tfm, train=False)))
    rss_load = rss()
    # warmup
    for m, tf in loaded:
        with torch.no_grad():
            m(torch.stack([tf(imgs[0])]))
    per_model = []
    total0 = time.perf_counter()
    with torch.no_grad():
        for im in imgs:
            for m, tf in loaded:
                x = torch.stack([tf(im)])
                m(x)
                if tta:
                    m(torch.stack([tf(im.transpose(1))]))
    total = time.perf_counter() - total0
    per_img = total / len(imgs) * 1000
    res[name] = dict(per_image_ms=round(per_img, 1), fps=round(len(imgs) / total, 2),
                     rss_load_mb=round(rss_load))
    print(f"{name:16s} {per_img:8.1f} ms/img  {len(imgs)/total:5.2f} FPS  RSS {rss_load:.0f} MB",
          flush=True)
    del loaded
    gc.collect()

Path(r"D:\bm_improve\analysis\bench_clean.json").write_text(json.dumps(res, indent=2))
print("saved")
