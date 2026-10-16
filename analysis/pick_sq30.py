#!/usr/bin/env python3
"""Честный выбор эпохи/набора эмбеддингов по многим разбиениям.

Максимум на одном сплите (551 запрос) — это подгонка под шум: SE mAP ~1.5.
Здесь каждый кандидат считается на N разбиениях по машинам, и решение
принимается по среднему, а не по максимуму.

    python pick_sq30.py --tags sq30_ep17,sq30_ep21,sq30_ep25,sq30_ep26,sq30_ep30
"""
import argparse
from pathlib import Path

import numpy as np

from official_protocol import official_metrics
from postproc_v2 import pipe
from rerank_test import l2, load_rows

HERE = Path(__file__).parent


def load(tag):
    return (l2(np.load(HERE / f"emb_{tag}_qry.npy").astype("float64")),
            l2(np.load(HERE / f"emb_{tag}_gal.npy").astype("float64")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    qry, gal = load_rows()
    S = {t: load(t) for t in tags}

    vids = sorted({r["vehicle_id"] for r in qry})
    folds = []
    for seed in range(a.splits):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        half = set(order[: len(order) // 2])
        idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
        folds.append((idx, [qry[i] for i in idx]))

    def score(name, q, g):
        sim = pipe(q, g)
        full = official_metrics(sim, qry, gal)
        vals = np.array([official_metrics(sim[i], rows, gal)["mAP@10"]
                         for i, rows in folds])
        print(f"{name:34s} {full['mAP@10']:7.2f} {full['Rank-1']:7.2f} "
              f"{vals.mean():8.2f} {vals.std():5.2f} {vals.min():6.2f} "
              f"{vals.max():6.2f}", flush=True)
        return vals

    print(f"запросов {len(qry)}, разбиений {a.splits}")
    print(f"{'набор':34s} {'mAP':>7s} {'R1':>7s} {'ср.':>8s} {'±':>5s} "
          f"{'мин':>6s} {'макс':>6s}")
    print("-" * 78)

    per = {}
    for t in tags:
        per[t] = score(t, *S[t])

    # усреднение всех кандидатов (сглаживает выбор одной эпохи)
    q_all = l2(sum(S[t][0] for t in tags) / len(tags))
    g_all = l2(sum(S[t][1] for t in tags) / len(tags))
    score(f"среднее всех ({len(tags)})", q_all, g_all)

    # усреднение топ-3 по среднему
    top3 = sorted(tags, key=lambda t: -per[t].mean())[:3]
    q3 = l2(sum(S[t][0] for t in top3) / 3)
    g3 = l2(sum(S[t][1] for t in top3) / 3)
    print(f"топ-3: {', '.join(top3)}")
    score("среднее топ-3", q3, g3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
