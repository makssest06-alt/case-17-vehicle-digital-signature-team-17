#!/usr/bin/env python3
"""Поиск финальной смеси среди сильных моделей (без слабых компонент).

Идея: squash-нативная (sq30) и base-нативная (ep29bs, repro_tta) модели видят
разные кропы, поэтому их ошибки частично независимы. Слабую geom-squash
не подмешиваем — она тянет вниз (проверено в mix_sq30.py).

    python mix_final.py --new-tag sq30_ep26 --splits 20
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
    ap.add_argument("--new-tag", required=True)
    ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()

    qry, gal = load_rows()
    S = {t: load(t) for t in (a.new_tag, "ep29new_bs", "repro_tta", "repro_squash")}

    vids = sorted({r["vehicle_id"] for r in qry})
    folds = []
    for seed in range(a.splits):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        half = set(order[: len(order) // 2])
        idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
        folds.append((idx, [qry[i] for i in idx]))

    results = {}

    def run(name, w):
        q = l2(sum(x * S[t][0] for t, x in w.items()))
        g = l2(sum(x * S[t][1] for t, x in w.items()))
        sim = pipe(q, g)
        full = official_metrics(sim, qry, gal)
        vals = np.array([official_metrics(sim[i], rows, gal)["mAP@10"]
                         for i, rows in folds])
        print(f"{name:44s} {full['mAP@10']:7.2f} {full['Rank-1']:7.2f} "
              f"{vals.mean():8.2f} {vals.std():5.2f} {(vals > 0).sum():3d}/{len(folds)}",
              flush=True)
        results[name] = (full["mAP@10"], vals.mean())
        return vals

    n = a.new_tag
    print(f"запросов {len(qry)}, разбиений {a.splits}")
    print(f"{'конфигурация':44s} {'mAP':>7s} {'R1':>7s} {'ср.':>8s} {'±':>5s} {'>0':>6s}")
    print("-" * 84)

    run(f"{n} одна", {n: 1.0})
    run("ep29bs одна", {"ep29new_bs": 1.0})
    run("repro_tta одна", {"repro_tta": 1.0})
    print("-" * 84)
    for w in (0.5, 0.6, 0.7, 0.8):
        run(f"{n}{w:.1f} + ep29bs{1 - w:.1f}", {n: w, "ep29new_bs": 1 - w})
    for w in (0.6, 0.7, 0.8):
        run(f"{n}{w:.1f} + repro_tta{1 - w:.1f}", {n: w, "repro_tta": 1 - w})
    print("-" * 84)
    for w_n, w_e in ((0.6, 0.2), (0.7, 0.15), (0.5, 0.25)):
        rest = 1 - w_n - w_e
        if rest <= 0.05:
            continue
        run(f"{n}{w_n:.2f} + ep29bs{w_e:.2f} + tta{rest:.2f}",
            {n: w_n, "ep29new_bs": w_e, "repro_tta": rest})
    run(f"{n}0.5 + ep29bs0.25 + tta0.25",
        {n: 0.5, "ep29new_bs": 0.25, "repro_tta": 0.25})

    best = max(results.items(), key=lambda kv: kv[1][1])
    print(f"\nлучшее по среднему разбиений: {best[0]} -> "
          f"mAP {best[1][0]:.2f}, ср. {best[1][1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
