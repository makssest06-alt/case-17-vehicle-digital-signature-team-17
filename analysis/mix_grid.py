#!/usr/bin/env python3
"""Проверка устойчивости весов 3-way смеси, а не поиск максимума.

Погоня за максимумом на 551 запросе — подгонка под шум. Здесь проверяем, что
выбранные веса лежат на плато и не зависят от конкретной эпохи squash-модели.

    python mix_grid.py --tags sq30_ep26,sq30_ep21 --splits 20
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
    ap.add_argument("--tags", required=True, help="squash-модели через запятую")
    ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()

    qry, gal = load_rows()
    base = {t: load(t) for t in ("ep29new_bs", "repro_tta")}
    news = {t: load(t) for t in a.tags.split(",") if t.strip()}

    vids = sorted({r["vehicle_id"] for r in qry})
    folds = []
    for seed in range(a.splits):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        half = set(order[: len(order) // 2])
        idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
        folds.append((idx, [qry[i] for i in idx]))

    def run(name, w):
        q = l2(sum(x * base[t][0] for t, x in w.items() if t in base) +
               sum(x * news[t][0] for t, x in w.items() if t in news))
        g = l2(sum(x * base[t][1] for t, x in w.items() if t in base) +
               sum(x * news[t][1] for t, x in w.items() if t in news))
        sim = pipe(q, g)
        full = official_metrics(sim, qry, gal)
        vals = np.array([official_metrics(sim[i], rows, gal)["mAP@10"]
                         for i, rows in folds])
        print(f"{name:42s} {full['mAP@10']:7.2f} {full['Rank-1']:7.2f} "
              f"{vals.mean():8.2f} {vals.std():5.2f}", flush=True)

    print(f"{'конфигурация':42s} {'mAP':>7s} {'R1':>7s} {'ср.':>8s} {'±':>5s}")
    print("-" * 76)

    # веса вокруг выбранной точки 0.6 / 0.2 / 0.2
    grid = [(0.55, 0.225, 0.225), (0.60, 0.20, 0.20), (0.65, 0.175, 0.175),
            (0.60, 0.25, 0.15), (0.60, 0.15, 0.25), (0.65, 0.20, 0.15),
            (0.70, 0.15, 0.15)]
    for tag in news:
        print(f"--- {tag} ---")
        for w_n, w_e, w_t in grid:
            run(f"{tag}{w_n:.2f} + ep29bs{w_e:.3f} + tta{w_t:.3f}",
                {tag: w_n, "ep29new_bs": w_e, "repro_tta": w_t})

    if len(news) > 1:
        print("--- среднее двух squash-моделей + base ---")
        tags = list(news)
        q_new = l2(sum(news[t][0] for t in tags) / len(tags))
        g_new = l2(sum(news[t][1] for t in tags) / len(tags))
        for w_n, w_e, w_t in ((0.6, 0.2, 0.2), (0.65, 0.175, 0.175)):
            q = l2(w_n * q_new + w_e * base["ep29new_bs"][0] + w_t * base["repro_tta"][0])
            g = l2(w_n * g_new + w_e * base["ep29new_bs"][1] + w_t * base["repro_tta"][1])
            sim = pipe(q, g)
            full = official_metrics(sim, qry, gal)
            vals = np.array([official_metrics(sim[i], rows, gal)["mAP@10"]
                             for i, rows in folds])
            label = "avg2%.2f + ep29bs%.3f + tta%.3f" % (w_n, w_e, w_t)
            print(f"{label:42s} "
                  f"{full['mAP@10']:7.2f} {full['Rank-1']:7.2f} "
                  f"{vals.mean():8.2f} {vals.std():5.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
