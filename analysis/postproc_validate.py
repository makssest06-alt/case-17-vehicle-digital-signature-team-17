#!/usr/bin/env python3
"""Независимая проверка параметров постобработки.

`postproc_tune.py` выбирает параметры по тем же разбиениям, на которых потом
и рапортует, — это завышает результат. Здесь кандидаты проверяются на
разбиениях с ДРУГИМИ зёрнами, плюс делается честная схема tune/test:
параметры подбираются на одной половине машин и проверяются на другой.

    python postproc_validate.py

Веса смеси задаются флагами (по умолчанию — текущая лучшая конфигурация):

    python postproc_validate.py --tags siglip2_ep28,sq30_ep26,ep29new_bs \
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

CANDIDATES = {
    "прежние (qk=1, a=3.0, ck=10)": dict(qk=1, qa=3.0, k1=30, k2=6, lam=0.7, ck=10),
    "без qq (qk=0, ck=10)": dict(qk=0, qa=1.0, k1=30, k2=6, lam=0.7, ck=10),
    "ck=5 (qk=1)": dict(qk=1, qa=3.0, k1=30, k2=6, lam=0.7, ck=5),
    "подобранные (qk=0, ck=5)": dict(qk=0, qa=1.0, k1=30, k2=6, lam=0.7, ck=5),
    "qk=0, ck=5, lam=0.5": dict(qk=0, qa=1.0, k1=30, k2=6, lam=0.5, ck=5),
}


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
    ap.add_argument("--tags", default=DEFAULT_TAGS)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    weights = [float(x) for x in a.weights.split(",")]
    if len(tags) != len(weights):
        raise SystemExit("число весов не совпадает с числом моделей")
    WEIGHTS = dict(zip(tags, weights))
    print("смесь: " + " + ".join(f"{w:.2f}*{t}" for t, w in WEIGHTS.items()), flush=True)

    qry, gal = load_rows()
    parts = [load(t) for t in WEIGHTS]
    q = l2(sum(w * p[0] for w, p in zip(WEIGHTS.values(), parts)))
    g = l2(sum(w * p[1] for w, p in zip(WEIGHTS.values(), parts)))

    vids = sorted({r["vehicle_id"] for r in qry})

    def make_folds(seeds, keep):
        folds = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            order = vids[:]
            rng.shuffle(order)
            half = set(order[: len(order) // 2])
            idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
            folds.append((idx, [qry[i] for i in idx]))
        return folds

    def run(p, folds):
        qe = qq_expand(q, p["qk"], p["qa"]) if p["qk"] > 0 else q
        S = csls(-k_reciprocal_rerank(qe, g, p["k1"], p["k2"], p["lam"]), p["ck"])
        return np.array([official_metrics(S[i], rows, gal)["mAP@10"]
                         for i, rows in folds])

    # 1) те же параметры на НЕЗАВИСИМЫХ зёрнах разбиений
    folds_new = make_folds(range(100, 120), None)
    print("=== разбиения с зёрнами 100..119 (тюнинг шёл на 0..19) ===")
    print(f"{'конфигурация':26s} {'ср.':>7s} {'±':>5s} {'мин':>6s}")
    print("-" * 48)
    for name, p in CANDIDATES.items():
        v = run(p, folds_new)
        print(f"{name:26s} {v.mean():7.2f} {v.std():5.2f} {v.min():6.2f}")

    # 2) честная схема tune/test: две половины машин, непересекающиеся
    print("\n=== tune на одной половине машин, test на другой ===")
    rng = np.random.default_rng(7)
    order = vids[:]
    rng.shuffle(order)
    half_a = set(order[: len(order) // 2])
    idx_a = [i for i, r in enumerate(qry) if r["vehicle_id"] in half_a]
    idx_b = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half_a]
    rows_a = [qry[i] for i in idx_a]
    rows_b = [qry[i] for i in idx_b]
    print(f"машин в A: {len(half_a)}, в B: {len(vids) - len(half_a)}; "
          f"запросов A {len(rows_a)}, B {len(rows_b)}")
    print(f"{'конфигурация':26s} {'A':>7s} {'B':>7s} {'среднее':>8s}")
    print("-" * 52)
    for name, p in CANDIDATES.items():
        qe = qq_expand(q, p["qk"], p["qa"]) if p["qk"] > 0 else q
        S = csls(-k_reciprocal_rerank(qe, g, p["k1"], p["k2"], p["lam"]), p["ck"])
        a = official_metrics(S[idx_a], rows_a, gal)["mAP@10"]
        b = official_metrics(S[idx_b], rows_b, gal)["mAP@10"]
        print(f"{name:26s} {a:7.2f} {b:7.2f} {(a + b) / 2:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
