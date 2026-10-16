#!/usr/bin/env python3
"""Смесь geom-squash с новой моделью (sq_scratch30) на holdout-сплите.

Единый конвейер постобработки и TTA, оценка по многим разбиениям — как в
`mix_ep29.py`, чтобы числа были сравнимы с прежними экспериментами.

Сначала извлечь эмбеддинги лучшей эпохи:

    python extract_emb.py --ckpt gpu_models/sq_scratch30_ep28.pt \
        --tag sq30 --tf-mode squash --tta

Затем подобрать вес:

    python mix_sq30.py --new-tag sq30
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
    ap.add_argument("--new-tag", required=True, help="тег новой модели")
    ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()

    qry, gal = load_rows()
    tags = ("repro_squash", "repro_tta", "ep29new_bs", a.new_tag)
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

    def run(name, w):
        q = l2(sum(x * S[t][0] for t, x in w.items()))
        g = l2(sum(x * S[t][1] for t, x in w.items()))
        sim = pipe(q, g)
        full = official_metrics(sim, qry, gal)
        vals = np.array([official_metrics(sim[i], rows, gal)["mAP@10"]
                         for i, rows in folds])
        print(f"{name:50s} {full['mAP@10']:7.2f} {full['Rank-1']:7.2f} "
              f"{vals.mean():8.2f} {vals.std():6.2f} {(vals > 0).sum():3d}/{len(folds)}",
              flush=True)
        return vals

    print(f"запросов {len(qry)}, разбиений {a.splits}")
    print(f"{'конфигурация':50s} {'mAP':>7s} {'R1':>7s} {'ср.':>8s} {'±':>6s} {'>0':>6s}")
    print("-" * 90)
    run("текущий лучший: gs0.8 + ep29bs0.2", {"repro_squash": 0.8, "ep29new_bs": 0.2})
    run("geom-squash одна", {"repro_squash": 1.0})
    run(f"{a.new_tag} одна", {a.new_tag: 1.0})
    for w_n in (0.2, 0.3, 0.4, 0.5):
        run(f"gs{1 - w_n:.1f} + new({w_n:.1f})",
            {"repro_squash": 1 - w_n, a.new_tag: w_n})
    for w_n in (0.2, 0.3):
        for w_e in (0.1, 0.2):
            base = 1 - w_n - w_e
            if base <= 0.05:
                continue
            run(f"gs{base:.1f} + new({w_n:.1f}) + ep29bs({w_e:.1f})",
                {"repro_squash": base, a.new_tag: w_n, "ep29new_bs": w_e})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
