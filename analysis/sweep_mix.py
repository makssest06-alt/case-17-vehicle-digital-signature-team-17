#!/usr/bin/env python3
"""Подбор веса ансамбля двух наборов эмбеддингов по 5 разбиениям.

Один сплит (551 запрос) даёт разброс ±1.5 mAP, поэтому вес выбирается по
среднему на 5 разбиениях, а не по максимуму на одном.

    python sweep_mix.py --a repro_tta --b repro_squash
"""
import argparse
from pathlib import Path

import numpy as np

from official_protocol import official_metrics
from postproc_v2 import csls, qq_expand
from rerank_test import k_reciprocal_rerank, l2, load_rows

HERE = Path(__file__).parent


def pipe(q, g):
    return csls(-k_reciprocal_rerank(qq_expand(q, g, 1, 1.0), g, 20, 6, 0.7), 10)


def load(tag):
    q = l2(np.load(HERE / f"emb_{tag}_qry.npy").astype("float64"))
    g = l2(np.load(HERE / f"emb_{tag}_gal.npy").astype("float64"))
    return q, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="тег первой модели (вес 1-w)")
    ap.add_argument("--b", required=True, help="тег второй модели (вес w)")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()

    qry, gal = load_rows()
    qa, ga = load(a.a)
    qb, gb = load(a.b)

    vids = sorted({r["vehicle_id"] for r in qry})
    folds = []
    for seed in range(a.splits):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        half = set(order[: len(order) // 2])
        idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
        folds.append((idx, [qry[i] for i in idx]))

    print(f"запросов {len(qry)}, разбиений {a.splits}")
    print(f"{'w(' + a.b + ')':>10s} {'все запросы':>12s} {'среднее 5 разб.':>16s} {'мин':>7s} {'макс':>7s}")
    print("-" * 58)

    ws = np.arange(0.0, 1.0001, a.step)
    best = None
    for w in ws:
        q = l2((1 - w) * qa + w * qb)
        g = l2((1 - w) * ga + w * gb)
        S = pipe(q, g)
        full = official_metrics(S, qry, gal)["mAP@10"]
        vals = [official_metrics(S[idx], rows, gal)["mAP@10"] for idx, rows in folds]
        mean = float(np.mean(vals))
        print(f"{w:10.1f} {full:12.2f} {mean:16.2f} {min(vals):7.2f} {max(vals):7.2f}")
        if best is None or mean > best[1]:
            best = (w, mean)

    print(f"\nлучший вес по среднему: w={best[0]:.1f} -> {best[1]:.2f} mAP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
