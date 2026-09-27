#!/usr/bin/env python3
"""Признаки валидационных кадров при нескольких масштабах кропа (multi-crop TTA).

    python extract_multicrop.py --scales 0.85 0.70

Кропы на диске уже сделаны с запасом 10% вокруг bbox, поэтому расширить рамку
нечем — исходных кадров на сервере нет. Значит доступны только более тесные
масштабы: центральный кроп 85% и 70% от готового кропа. Это честное ограничение,
а не выбор: при пересборке кропов из исходников стоит добавить и масштаб больше 1.

Пишет /opt/case17/data/features/val_siglip_base_s<масштаб>_orig.npy и общий
индекс val_multicrop_index.csv. Считаются только 912 кадров валидации, а не весь
train: multi-crop проверяется на инференсе, обучать под него нечего.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import backbones

CROPS = Path("/opt/case17/data/crops")
FEAT = Path("/opt/case17/data/features")
HERE = Path(__file__).parent


def center_crop(im: Image.Image, frac: float) -> Image.Image:
    w, h = im.size
    nw, nh = int(round(w * frac)), int(round(h * frac))
    left, top = (w - nw) // 2, (h - nh) // 2
    return im.crop((left, top, left + nw, top + nh))


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="siglip_base")
    ap.add_argument("--scales", nargs="+", type=float, default=[0.85, 0.70])
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    torch.set_num_threads(2)
    spec = backbones.get(a.backbone)
    forward, tf, _ = backbones.build_encoder(spec)

    val_ids = {l.strip() for l in (HERE / "val_ids.txt").read_text().split() if l.strip()}
    rows = [r for r in csv.DictReader((FEAT / "train_index.csv").open())
            if r["vehicle_id"] in val_ids]
    ids = [r["image_id"] for r in rows]
    n = len(ids)
    print(f"кадров валидации: {n}, масштабы: {a.scales}", flush=True)

    with (FEAT / "val_multicrop_index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "vehicle_id", "camera_id"])
        for r in rows:
            w.writerow([r["image_id"], r["vehicle_id"], r["camera_id"]])

    for frac in a.scales:
        tag = f"{frac:.2f}"
        out = FEAT / f"val_{spec.key}_s{tag}_orig.npy"
        if out.exists():
            print(f"  масштаб {tag}: уже посчитан", flush=True)
            continue
        arr = np.zeros((n, spec.dim), dtype="float32")
        t0 = time.perf_counter()
        for i in range(0, n, a.batch):
            chunk = ids[i : i + a.batch]
            imgs = [center_crop(Image.open(CROPS / "train" / f"{c}.jpg").convert("RGB"), frac)
                    for c in chunk]
            arr[i : i + len(chunk)] = forward(torch.stack([tf(im) for im in imgs])).numpy()
            if i and i % (a.batch * 10) == 0:
                done = i + len(chunk)
                print(f"    {done}/{n}  {done / (time.perf_counter() - t0):.2f} кадров/с",
                      flush=True)
        np.save(out, arr)
        print(f"  масштаб {tag}: готов за {time.perf_counter() - t0:.0f} с -> {out.name}",
              flush=True)

    print("MULTICROP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
