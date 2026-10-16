#!/usr/bin/env python3
"""Coordinate ascent over an arbitrary set of embedding sets.

Usage: python optimize_n.py tag1,tag2,...
Objective = mean mAP over tune folds (seeds 0..19); validate on 100..139.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from validate_ensemble import load_split, official_metrics  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)


def main():
    tags = sys.argv[1].split(",")
    qry, gal = load_split()
    vids = sorted({r["vehicle_id"] for r in qry})
    q = [np.load(HERE / f"{t}_qry.npy").astype("float64") for t in tags]
    g = [np.load(HERE / f"{t}_gal.npy").astype("float64") for t in tags]
    n = len(tags)

    def sim(w):
        s, _, _ = rp.rank(q, g, list(w), **D)
        return s

    def folds(seeds):
        out = []
        for s in seeds:
            rng = np.random.default_rng(s)
            order = vids[:]
            rng.shuffle(order)
            half = set(order[: len(order) // 2])
            idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
            out.append((idx, [qry[i] for i in idx]))
        return out

    fT, fV = folds(range(0, 20)), folds(range(100, 140))

    def obj(w, fs):
        s = sim(w)
        return float(np.mean([official_metrics(s[i], rows, gal)["mAP"] for i, rows in fs]))

    def norm(x):
        x = np.clip(np.asarray(x, float), 0, None)
        return x / x.sum()

    w = np.ones(n) / n
    if n >= 4:
        w = norm([0.36, 0.23, 0.19, 0.22] + [0.0] * (n - 4))
    bestT = obj(w, fT)
    print(f"start tune {bestT:.2f} w={[round(x,3) for x in w]}", flush=True)
    for step in (0.08, 0.04, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for sgn in (+1, -1):
                    c = norm(w + sgn * step * np.eye(n)[i])
                    v = obj(c, fT)
                    if v > bestT + 1e-9:
                        w, bestT = c, v
                        improved = True
                        print(f"  step {step}: tune {bestT:.2f} "
                              f"{dict(zip(tags, [round(x,3) for x in w]))}", flush=True)
    print(f"\nOPT tune {bestT:.2f}")
    print(f"w = {dict(zip(tags, [round(x,4) for x in w]))}")
    s_opt = sim(list(w))
    m = official_metrics(s_opt, qry, gal)
    print(f"full mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f} R5 {m['R5']:6.2f}")
    d = np.array([official_metrics(s_opt[i], rows, gal)["mAP"] -
                  official_metrics(sim([0.45, 0.25, 0.30] + [0.0] * (n - 3))[i], rows, gal)["mAP"]
                  for i, rows in fV])
    print(f"valid(100..139) vs team3: {d.mean():+.2f} ± {d.std():.2f}, "
          f"better {(d>0).sum()}/{len(d)}")
    np.save(HERE / "weights_n_opt.npy", w)
    (HERE / "weights_n_names.txt").write_text("\n".join(tags))


if __name__ == "__main__":
    main()
