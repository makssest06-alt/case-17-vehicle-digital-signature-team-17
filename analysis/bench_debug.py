#!/usr/bin/env python3
"""Debug: time each model's forward separately on CPU, 2 threads."""
import csv
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

rows = list(csv.DictReader(TESTQ.open(encoding="utf-8")))[:20]
imgs = [crop_image(IMAGES / f"{r['image_id']}.jpg", r, 0.10) for r in rows]

CK = {
    "siglip2": (MODELS / "siglip2_reid_scratch30_ep28.pt", "squash"),
    "sq30": (MODELS / "siglip_reid_scratch30_ep26.pt", "squash"),
    "ep29": (MODELS / "siglip_vitb16_reid_ep29.pt", "timm"),
    "clip": (EXTRA / "clip_reid_scratch30_ep28.pt", "squash"),
}

with torch.no_grad():
    for name, (path, tfm) in CK.items():
        model, spec, _ = load_model(path, device)
        tf = build_transforms(spec, tfm, train=False)
        x = torch.stack([tf(im) for im in imgs[:2]])
        model(x)  # warmup
        t0 = time.perf_counter()
        for im in imgs:
            xi = torch.stack([tf(im)])
            model(xi)
        dt = (time.perf_counter() - t0) / len(imgs) * 1000
        print(f"{name:8s} avg forward per image = {dt:7.1f} ms", flush=True)
        del model
