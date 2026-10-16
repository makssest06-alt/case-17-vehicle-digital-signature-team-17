#!/usr/bin/env python3
"""Сквозная проверка новой смеси через официальный evaluate.py.

Сравнивает на val-сплите:
  C) прежнюю лучшую тройку (sq30 0.6 + ep29-base 0.2 + geom-base 0.2)
  D) новую (siglip2 0.45 + sq30 0.25 + ep29-base 0.30)

Скрипт организаторов — тот же, что в ТЗ; берём его без правок, поэтому
числа сопоставимы с прошлыми прогонами. Артефакты кладутся в
`official_run4/` в формате организаторов.
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from postproc_v2 import pipe
from rerank_test import l2, load_rows

HERE = Path(__file__).parent
OUT = HERE / "official_run4"
EVAL = Path(r"C:\Users\user\Downloads\evaluate.py")


def load(tag):
    return (l2(np.load(HERE / f"emb_{tag}_qry.npy").astype("float64")),
            l2(np.load(HERE / f"emb_{tag}_gal.npy").astype("float64")))


def main():
    OUT.mkdir(exist_ok=True)
    qry, gal = load_rows()
    box = {r["image_id"]: r for r in
           csv.DictReader((HERE / "train.csv").open(encoding="utf-8"))}
    for r in qry + gal:
        b = box[r["image_id"]]
        r["x"], r["y"], r["w"], r["h"] = (int(b[k]) for k in ("x", "y", "w", "h"))

    with (OUT / "gt.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "vehicle_id", "camera_id", "split"])
        for r in qry:
            w.writerow([r["image_id"], r["vehicle_id"], r["camera_id"], "query"])
        for r in gal:
            w.writerow([r["image_id"], r["vehicle_id"], r["camera_id"], "gallery"])

    for name, rows in (("test_query.csv", qry), ("test_gallery.csv", gal)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "x", "y", "w", "h"])
            for r in rows:
                w.writerow([r["image_id"], r["x"], r["y"], r["w"], r["h"]])

    configs = {
        "C_prev_3way": [("sq30_ep26", 0.6), ("ep29new_bs", 0.2), ("repro_tta", 0.2)],
        "D_new_3model": [("siglip2_ep28", 0.45), ("sq30_ep26", 0.25),
                         ("ep29new_bs", 0.30)],
    }

    for name, parts in configs.items():
        sets = [load(t) for t, _ in parts]
        ws = [w for _, w in parts]
        q = l2(sum(w * s[0] for w, s in zip(ws, sets)))
        g = l2(sum(w * s[1] for w, s in zip(ws, sets)))

        np.save(OUT / f"embeddings_{name}.npy", np.vstack([q, g]).astype("float32"))
        S = pipe(q, g)
        order = np.argsort(-S, axis=1)

        with (OUT / f"submission_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, 11)])
            for qi, qr in enumerate(qry):
                w.writerow([qr["image_id"]] + [gal[j]["image_id"] for j in order[qi, :10]])

        with (OUT / f"candidates_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "gallery_id", "confidence"])
            for qi, qr in enumerate(qry):
                j = int(order[qi, 0])
                w.writerow([qr["image_id"], gal[j]["image_id"], round(float(S[qi, j]), 4)])

        cmd = [
            sys.executable, str(EVAL),
            "--gt", str(OUT / "gt.csv"),
            "--submission", str(OUT / f"submission_{name}.csv"),
            "--candidates", str(OUT / f"candidates_{name}.csv"),
            "--embeddings", str(OUT / f"embeddings_{name}.npy"),
            "--query", str(OUT / "test_query.csv"),
            "--gallery", str(OUT / "test_gallery.csv"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print("=" * 70)
        print(f"### {name}")
        print(r.stdout[-1200:] if r.stdout else r.stderr[-1200:])


if __name__ == "__main__":
    main()
