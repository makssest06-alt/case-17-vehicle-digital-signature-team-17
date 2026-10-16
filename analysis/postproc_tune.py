#!/usr/bin/env python3
"""Подбор параметров постобработки под текущую смесь эмбеддингов.

Прежние значения (k1=20, k2=6, lambda=0.7, CSLS k=10, qq k=1) выбирались под
эмбеддинги geom-модели. Смесь трёх моделей имеет другую структуру сходств,
поэтому имеет смысл перепроверить.

Чтобы не подгоняться под один сплит, решение принимается по среднему на
20 разбиениях по машинам, а не по максимуму на полном сплите.

    python postproc_tune.py --splits 20

Веса смеси задаются флагами, чтобы не править файл под каждую конфигурацию:

    python postproc_tune.py --tags siglip2_ep28,sq30_ep26,ep29new_bs \
        --weights 0.45,0.25,0.30
"""
import argparse
from pathlib import Path

import numpy as np

from official_protocol import official_metrics
from rerank_test import k_reciprocal_rerank, l2, load_rows

HERE = Path(__file__).parent

DEFAULT_TAGS = "siglip2_ep28,sq30_ep26,ep29new_bs"
DEFAULT_WEIGHTS = "0.45,0.25,0.30"


def load(tag):
    return (l2(np.load(HERE / f"emb_{tag}_qry.npy").astype("float64")),
            l2(np.load(HERE / f"emb_{tag}_gal.npy").astype("float64")))


def qq_expand(q, k, alpha):
    sim = q @ q.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argsort(-sim, axis=1)[:, :k]
    w = np.clip(sim[np.arange(len(q))[:, None], idx], 0, None) ** alpha
    return l2(q + np.einsum("nk,nkd->nd", w, q[idx]))


def csls(S, k):
    rq = np.sort(S, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    rg = np.sort(S, axis=0)[-k:, :].mean(axis=0, keepdims=True)
    return 2 * S - rq - rg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--tags", default=DEFAULT_TAGS)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    a = ap.parse_args()

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    weights = [float(x) for x in a.weights.split(",")]
    if len(tags) != len(weights):
        raise SystemExit("число весов не совпадает с числом моделей")
    WEIGHTS = dict(zip(tags, weights))
    print(f"смесь: " + " + ".join(f"{w:.2f}*{t}" for t, w in WEIGHTS.items()), flush=True)

    qry, gal = load_rows()
    parts = [load(t) for t in WEIGHTS]
    q = l2(sum(w * p[0] for w, p in zip(WEIGHTS.values(), parts)))
    g = l2(sum(w * p[1] for w, p in zip(WEIGHTS.values(), parts)))

    vids = sorted({r["vehicle_id"] for r in qry})
    folds = []
    for seed in range(a.splits):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        half = set(order[: len(order) // 2])
        idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
        folds.append((idx, [qry[i] for i in idx]))

    def evaluate(qk, qa, k1, k2, lam, ck, verbose=False):
        qe = qq_expand(q, qk, qa) if qk > 0 else q
        S = csls(-k_reciprocal_rerank(qe, g, k1, k2, lam), ck)
        full = official_metrics(S, qry, gal)["mAP@10"]
        vals = np.array([official_metrics(S[i], rows, gal)["mAP@10"]
                         for i, rows in folds])
        if verbose:
            print(f"  qq k={qk} a={qa} | rr k1={k1} k2={k2} lam={lam} | csls k={ck}"
                  f"  ->  mAP {full:6.2f}  ср. {vals.mean():6.2f}", flush=True)
        return vals.mean()

    cur = dict(qk=1, qa=1.0, k1=20, k2=6, lam=0.7, ck=10)
    base = evaluate(**cur, verbose=True)
    print(f"исходные параметры: ср. {base:.2f}\n")

    grid = {
        "qk": [0, 1, 2, 3],
        "qa": [1.0, 2.0, 3.0],
        "k1": [10, 20, 30, 40],
        "k2": [3, 6, 10],
        "lam": [0.3, 0.5, 0.7],
        "ck": [5, 10, 20, 30],
    }
    for _ in range(2):
        for key, values in grid.items():
            best_v, best_s = cur[key], evaluate(**cur)
            for v in values:
                trial = dict(cur)
                trial[key] = v
                s = evaluate(**trial)
                if s > best_s:
                    best_v, best_s = v, s
            cur[key] = best_v
            print(f"{key:4s} -> {best_v}  (ср. {best_s:.2f})", flush=True)

    print(f"\nитоговые параметры: {cur}")
    print(f"ср. по {a.splits} разбиениям: {evaluate(**cur):.2f} "
          f"(было {base:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
