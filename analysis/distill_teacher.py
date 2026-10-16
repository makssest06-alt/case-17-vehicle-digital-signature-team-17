#!/usr/bin/env python3
"""Compute TEACHER (4-model ensemble) embeddings for the full train split.

Student will be trained to mimic these embeddings (feature distillation).
Rows = train.csv minus val vehicles (same split the teachers were trained on).
Saves: teacher_emb.npy (L2-normalised, weighted ensemble), distill_train_rows.csv
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch

from extract_emb import embed_rows, load_model
from train_reid import build_transforms
from reid_postprocess import l2

HERE = Path(__file__).parent
IMAGES = Path(r"D:\bm\dataset_extracted\images")
TRAIN = Path(r"D:\bm\dataset_extracted\train.csv")
VAL_IDS = Path(r"D:\bm\improve_val_ids.txt")
OUT = Path(r"D:\bm\improve_models")

CONFIG = [
    (r"D:\bm_improve\models\siglip2_reid_scratch30_ep28.pt", "squash", 0.364),
    (r"D:\bm_improve\models\siglip_reid_scratch30_ep26.pt", "squash", 0.232),
    (r"D:\bm_improve\models\siglip_vitb16_reid_ep29.pt", "timm", 0.182),
    (r"D:\bm\improve_models\clip_reid_scratch30_ep28.pt", "squash", 0.222),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    val_ids = {l.strip() for l in VAL_IDS.read_text().split() if l.strip()}
    rows = [r for r in csv.DictReader(TRAIN.open(encoding="utf-8"))
            if r["vehicle_id"] not in val_ids]
    print(f"train rows {len(rows)}", flush=True)

    q = np.zeros((len(rows), 512), dtype="float64")
    t0 = time.perf_counter()
    for path, tfm, w in CONFIG:
        tm = time.perf_counter()
        model, spec, _ = load_model(path, device)
        tf = build_transforms(spec, tfm, train=False)
        e = embed_rows(model, tf, rows, IMAGES, 0.10, device, tta=True, bs=24)
        q += w * l2(e.astype("float64"))
        print(f"  {Path(path).name}: {time.perf_counter()-tm:.1f}s", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    q = l2(q)
    np.save(OUT / "teacher_emb.npy", q.astype("float32"))
    with (OUT / "distill_train_rows.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["image_id", "x", "y", "w", "h", "vehicle_id"])
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in wr.fieldnames})
    print(f"teacher_emb {q.shape} in {time.perf_counter()-t0:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
