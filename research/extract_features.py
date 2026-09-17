#!/usr/bin/env python3
"""Признаки CLIP ViT-B/16 для всех кропов: оригинал + горизонтальное отражение.

Пишет по одному .npy на сплит плюс индексный csv. Считает батчами и
сохраняет прогресс: если процесс упал, следующий запуск продолжит с места.
Результат в /opt/case17/data/features — в git не попадает.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

CROPS = Path("/opt/case17/data/crops")
OUT = Path("/opt/case17/data/features")
MODEL_NAME = "vit_base_patch16_clip_224.openai"
DIM = 768
SPLITS = {"train": "crops_train.csv", "query": "crops_query.csv", "gallery": "crops_gallery.csv"}


def build():
    import timm

    m = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    m.eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    cfg["input_size"] = (3, 224, 224)
    return m, timm.data.create_transform(**cfg, is_training=False)


@torch.inference_mode()
def run_split(model, tf, split: str, csv_name: str, batch: int) -> None:
    rows = list(csv.DictReader((CROPS / csv_name).open()))
    ids = [r["image_id"] for r in rows]
    n = len(ids)

    idx_path = OUT / f"{split}_index.csv"
    with idx_path.open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["image_id", "vehicle_id", "camera_id"] if split == "train" else ["image_id"]
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if c in r else "" for c in cols])

    # memmap: переживает падение процесса, прогресс хранится отдельным файлом
    for kind in ("orig", "flip"):
        path = OUT / f"{split}_{kind}.npy"
        if not path.exists():
            np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(n, DIM)).flush()

    done_path = OUT / f"{split}_done.txt"
    done = int(done_path.read_text()) if done_path.exists() else 0
    if done >= n:
        print(f"  {split}: уже посчитан ({n})", flush=True)
        return

    a_orig = np.lib.format.open_memmap(OUT / f"{split}_orig.npy", mode="r+")
    a_flip = np.lib.format.open_memmap(OUT / f"{split}_flip.npy", mode="r+")
    folder = CROPS / split
    t0, start = time.perf_counter(), done
    print(f"  {split}: продолжаю с {done}/{n}", flush=True)

    for i in range(done, n, batch):
        chunk = ids[i : i + batch]
        imgs = [Image.open(folder / f"{c}.jpg").convert("RGB") for c in chunk]
        x = torch.stack([tf(im) for im in imgs])
        xf = torch.stack([tf(ImageOps.mirror(im)) for im in imgs])
        fo = torch.nn.functional.normalize(model(x), p=2, dim=1).numpy()
        ff = torch.nn.functional.normalize(model(xf), p=2, dim=1).numpy()
        a_orig[i : i + len(chunk)] = fo
        a_flip[i : i + len(chunk)] = ff
        a_orig.flush(); a_flip.flush()
        done = i + len(chunk)
        done_path.write_text(str(done))
        if done % (batch * 10) == 0 or done == n:
            sp = (done - start) / (time.perf_counter() - t0)
            eta = (n - done) / sp / 60 if sp > 0 else 0
            print(f"    {done}/{n}  {sp:.2f} кадров/с  осталось ~{eta:.0f} мин", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--splits", default="train,query,gallery")
    a = ap.parse_args()

    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    model, tf = build()
    print(f"модель {MODEL_NAME} загружена", flush=True)
    for s in a.splits.split(","):
        run_split(model, tf, s, SPLITS[s], a.batch)
    print("ГОТОВО", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
