#!/usr/bin/env python3
"""Correct 4-model weight optimization: [siglip2, sq30, ep29, clip].

Coordinate ascent on tune folds (seeds 0..19); validate on 100..139.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from validate_ensemble import load_split, official_metrics  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)
NAMES = ["emb_siglip2_ep28", "emb_sq30_ep26", "emb_ep29_fresh", "emb_clip_ep28"]


def main():
    qry, gal = load_split()
    vids = sorted({r["vehicle_id"] for r in qry})
    q = [np.load(HERE / f"{p}_qry.npy").astype("float64") for p in NAMES]
    g = [np.load(HERE / f"{p}_gal.npy").astype("float64") for p in NAMES]

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

    fT = folds(range(0, 20))
    fV = folds(range(100, 140))

    def obj(w, fs):
        s = sim(w)
        return float(np.mean([official_metrics(s[i], rows, gal)["mAP"] for i, rows in fs]))

    def norm(x):
        x = np.clip(np.asarray(x, dtype=float), 0, None)
        return x / x.sum()

    # start from teammate weights + small clip
    w = np.array([0.40, 0.22, 0.27, 0.11])
    bestT = obj(w, fT)
    print(f"start tune {bestT:.2f} w={[round(x,3) for x in w]}", flush=True)
    for step in (0.08, 0.04, 0.02):
        improved = True
        while improved:
            improved = False
            for i in range(len(w)):
                for sgn in (+1, -1):
                    c = w.copy()
                    c[i] += sgn * step
                    c = norm(c)
                    v = obj(c, fT)
                    if v > bestT + 1e-9:
                        w, bestT = c, v
                        improved = True
                        print(f"  step {step}: tune {bestT:.2f} w={[round(x,3) for x in w]}",
                              flush=True)
    print(f"\nOPT w={[round(x,3) for x in w]}  tune {bestT:.2f}", flush=True)

    # validate vs teammate
    s_team = sim([0.45, 0.25, 0.30, 0.0])
    s_opt = sim(list(w))
    mt = official_metrics(s_team, qry, gal)
    mo = official_metrics(s_opt, qry, gal)
    print(f"team3  full mAP {mt['mAP']:6.2f} R1 {mt['R1']:6.2f}")
    print(f"opt4   full mAP {mo['mAP']:6.2f} R1 {mo['R1']:6.2f}")
    d = np.array([official_metrics(s_opt[i], rows, gal)["mAP"] -
                  official_metrics(s_team[i], rows, gal)["mAP"] for i, rows in fV])
    print(f"valid(100..139): delta {d.mean():+.2f} ± {d.std():.2f}, "
          f"better {(d>0).sum()}/{len(d)}")
    np.save(HERE / "weights4_opt.npy", w)


if __name__ == "__main__":
    main()
