#!/usr/bin/env python3
"""Clean repeated CPU benchmark (min over repeats) for same-arch models."""
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
}
N, REPS = 10, 3

rows = list(csv.DictReader(TESTQ.open(encoding="utf-8")))[:N]
imgs = [crop_image(IMAGES / f"{r['image_id']}.jpg", r, 0.10) for r in rows]

res = {}
for name, (path, tfm) in CK.items():
    gc.collect()
    m, spec, _ = load_model(path, device)
    tf = build_transforms(spec, tfm, train=False)
    xs = [torch.stack([tf(im)]) for im in imgs]
    with torch.no_grad():
        m(xs[0])
        best = 1e9
        for _ in range(REPS):
            t0 = time.perf_counter()
            for x in xs:
                m(x)
            best = min(best, (time.perf_counter() - t0) / len(xs) * 1000)
    res[name] = dict(per_image_ms=round(best, 1), fps_b1=round(1000 / best, 2))
    print(f"{name:16s} {best:7.1f} ms/img  {1000/best:5.2f} FPS (batch=1)", flush=True)
    del m
    gc.collect()

Path(r"D:\bm_improve\analysis\bench_student.json").write_text(json.dumps(res, indent=2))
print("saved")
