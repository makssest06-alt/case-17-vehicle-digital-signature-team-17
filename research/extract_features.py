#!/usr/bin/env python3
"""Признаки выбранного бэкбона для всех кропов: оригинал + горизонтальное отражение.

Бэкбон выбирается параметром --backbone из реестра backbones.py.
Файлы: /opt/case17/data/features/{split}_{backbone}_{orig|flip}.npy + индексный csv.
Считает батчами и сохраняет прогресс: если процесс упал, следующий запуск
продолжит с места. В git не попадает.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import backbones

CROPS = Path("/opt/case17/data/crops")
OUT = Path("/opt/case17/data/features")
SPLITS = {"train": "crops_train.csv", "query": "crops_query.csv", "gallery": "crops_gallery.csv"}


@torch.inference_mode()
def run_split(forward, tf, spec, split: str, csv_name: str, batch: int) -> None:
    rows = list(csv.DictReader((CROPS / csv_name).open()))
    ids = [r["image_id"] for r in rows]
    n = len(ids)
    pref = f"{split}_{spec.key}"

    with (OUT / f"{split}_index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["image_id", "vehicle_id", "camera_id"] if split == "train" else ["image_id"]
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])

    for kind in ("orig", "flip"):
        p = OUT / f"{pref}_{kind}.npy"
        if not p.exists():
            np.lib.format.open_memmap(p, mode="w+", dtype="float32", shape=(n, spec.dim)).flush()

    done_path = OUT / f"{pref}_done.txt"
    done = int(done_path.read_text()) if done_path.exists() else 0
    if done >= n:
        print(f"  {split}: уже посчитан ({n})", flush=True)
        return

    a_o = np.lib.format.open_memmap(OUT / f"{pref}_orig.npy", mode="r+")
    a_f = np.lib.format.open_memmap(OUT / f"{pref}_flip.npy", mode="r+")
    folder = CROPS / split
    t0, start = time.perf_counter(), done
    print(f"  {split}: продолжаю с {done}/{n}", flush=True)

    for i in range(done, n, batch):
        chunk = ids[i : i + batch]
        imgs = [Image.open(folder / f"{c}.jpg").convert("RGB") for c in chunk]
        x = torch.stack([tf(im) for im in imgs])
        xf = torch.stack([tf(ImageOps.mirror(im)) for im in imgs])
        fo = torch.nn.functional.normalize(forward(x), p=2, dim=1).numpy()
        ff = torch.nn.functional.normalize(forward(xf), p=2, dim=1).numpy()
        a_o[i : i + len(chunk)] = fo
        a_f[i : i + len(chunk)] = ff
        a_o.flush(); a_f.flush()
        done = i + len(chunk)
        done_path.write_text(str(done))
        if done % (batch * 10) == 0 or done == n:
            sp = (done - start) / (time.perf_counter() - t0)
            print(f"    {done}/{n}  {sp:.2f} кадров/с  осталось ~{(n-done)/sp/60:.0f} мин", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip_vitb16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--splits", default="train,query,gallery")
    a = ap.parse_args()

    spec = backbones.get(a.backbone)
    if spec.key in backbones.RESEARCH_ONLY:
        print(f"ВНИМАНИЕ: {spec.key} — {spec.license_note}.", flush=True)
        print("Использовать только как исследовательский ориентир, см. research/LICENSES.md.", flush=True)

    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    forward, tf, _ = backbones.build_encoder(spec)
    print(f"бэкбон {spec.key} ({spec.dim}D, вход {spec.input_size}) загружен", flush=True)
    for s in a.splits.split(","):
        run_split(forward, tf, spec, s, SPLITS[s], a.batch)
    print("ГОТОВО", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
