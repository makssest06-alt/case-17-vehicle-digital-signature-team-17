#!/usr/bin/env python3
"""Clean CPU benchmark (2 threads): production vs students, TTA/no-TTA."""
import csv
import gc
import json
import os
import time
from pathlib import Path

import torch

from extract_emb import crop_image, load_model
from train_reid import build_transforms

IMAGES = Path(r"D:\bm\dataset_extracted\images")
TESTQ = Path(r"D:\bm\dataset_extracted\test_query.csv")
torch.set_num_threads(2)
device = torch.device("cpu")

CK = {
    "prod_geom_ep30": (r"D:\bm_improve\models\siglip_vitb16_reid_geom_ep30.pt", "timm"),
    "student_a25_ep28": (r"D:\bm\improve_models\siglip2_distill_a25_ep28.pt", "squash"),
    "student_dv2s": (r"D:\bm\improve_models\distill_dv2s_ep30.pt", "squash"),
}
N, REPS = 10, 3

rows = list(csv.DictReader(TESTQ.open(encoding="utf-8")))[:N]
imgs = [crop_image(IMAGES / f"{r['image_id']}.jpg", r, 0.10) for r in rows]


def rss():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


res = {}
for name, (path, tfm) in CK.items():
    for tta in (True, False):
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
                    if tta:
                        m(torch.flip(x, dims=[3]))
                best = min(best, (time.perf_counter() - t0) / len(xs) * 1000)
        key = f"{name}{'_tta' if tta else ''}"
        res[key] = dict(per_image_ms=round(best, 1), fps=round(1000 / best, 2),
                        rss_mb=round(rss()))
        print(f"{key:26s} {best:7.1f} ms  {1000/best:5.2f} FPS  RAM {rss():.0f} MB", flush=True)
        del m
        gc.collect()

Path(r"D:\bm_improve\analysis\bench_final.json").write_text(json.dumps(res, indent=2))
print("saved")
