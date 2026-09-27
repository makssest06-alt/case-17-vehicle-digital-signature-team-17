#!/usr/bin/env python3
"""Валидационный сплит из train: 150 машин, gallery — по одному кадру с каждой камеры.

Такой сплит повторяет условия жюри: query и gallery сняты РАЗНЫМИ камерами,
поэтому метрика не завышается за счёт почти одинаковых соседних кадров.
"""
import csv
import random
import argparse
from collections import defaultdict
from pathlib import Path

SEED = 42
N_VEHICLES = 150
OUT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="/opt/case17/data/crops/crops_train.csv")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(a.train_csv).open(encoding="utf-8")))
    by_vehicle = defaultdict(list)
    for r in rows:
        by_vehicle[r["vehicle_id"]].append(r)

    vehicles = sorted(by_vehicle)
    picked = sorted(random.Random(SEED).sample(vehicles, N_VEHICLES), key=int)
    (out / "val_ids.txt").write_text("\n".join(picked) + "\n")

    split = []
    for vid in picked:
        by_cam = defaultdict(list)
        for r in by_vehicle[vid]:
            by_cam[r["camera_id"]].append(r)
        for cam in sorted(by_cam, key=int):
            items = sorted(by_cam[cam], key=lambda r: r["image_id"])
            split.append((items[0], "gallery"))          # по одному кадру с каждой камеры
            split.extend((r, "query") for r in items[1:])  # остальные — запросы

    with (out / "val_split.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "vehicle_id", "camera_id", "role"])
        for r, role in split:
            w.writerow([r["image_id"], r["vehicle_id"], r["camera_id"], role])

    n_g = sum(1 for _, role in split if role == "gallery")
    n_q = len(split) - n_g
    cams = {(r["vehicle_id"], r["camera_id"]) for r, _ in split}
    print(f"машин: {len(picked)}")
    print(f"gallery: {n_g}, query: {n_q}, всего кадров: {len(split)}")
    print(f"пар (машина, камера): {len(cams)}")
    # сколько query вообще имеют шанс найтись: нужен тот же vehicle_id с ДРУГОЙ камеры
    gal = defaultdict(set)
    for r, role in split:
        if role == "gallery":
            gal[r["vehicle_id"]].add(r["camera_id"])
    usable = sum(1 for r, role in split if role == "query" and len(gal[r["vehicle_id"]] - {r["camera_id"]}) > 0)
    print(f"query с хотя бы одним релевантным кандидатом с другой камеры: {usable}")


if __name__ == "__main__":
    main()
