#!/usr/bin/env python3
"""Optimize weights for student-based small ensembles (no-TTA) + validate."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(r"D:\bm\research\ensemble")
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from official_protocol import load_split, official_ranking  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)
NO = {
    "stu12": "emb_distill_a12_ep26_noTTA",
    "sq30": "emb_sq30_ep26_noTTA",
    "ep29": "emb_ep29_noTTA",
    "clip": "emb_clip_ep28_noTTA",
    "siglip2": "emb_siglip2_ep28_noTTA",
}


def L(p):
    return (np.load(HERE / f"{p}_qry.npy").astype("float64"),
            np.load(HERE / f"{p}_gal.npy").astype("float64"))


def main():
    qry, gal = load_split()
    vids = sorted({r["vehicle_id"] for r in qry})

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

    def optimize(keys, label):
        q = [L(NO[k])[0] for k in keys]
        g = [L(NO[k])[1] for k in keys]
        n = len(keys)

        def sim(w):
            s, _, _ = rp.rank(q, g, list(w), **D)
            return s

        def obj(w, fs):
            s = sim(w)
            return float(np.mean([official_ranking(s[i], rows, gal)["mAP@10"] for i, rows in fs]))

        def norm(x):
            x = np.clip(np.asarray(x, float), 0, None)
            return x / x.sum()

        w = np.ones(n) / n
        best = obj(w, fT)
        for step in (0.1, 0.05, 0.025):
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    for sgn in (+1, -1):
                        c = norm(w + sgn * step * np.eye(n)[i])
                        v = obj(c, fT)
                        if v > best + 1e-9:
                            w, best = c, v
                            improved = True
        s = sim(list(w))
        full = official_ranking(s, qry, gal)
        d = np.array([official_ranking(s[i], rows, gal)["mAP@10"] -
                      official_ranking(sim([1.0 / n] * n)[i], rows, gal)["mAP@10"]
                      for i, rows in fV])
        print(f"{label:26s} w={[round(x,3) for x in w]}  mAP@10 {full['mAP@10']:6.2f} "
              f"R1 {full['Rank-1']:6.2f} | tune {best:.2f}")
        return w, full

    print("=== optimizing ===")
    optimize(["stu12", "sq30"], "stu+sq30")
    optimize(["stu12", "sq30", "ep29"], "stu+sq30+ep29")
    optimize(["stu12", "clip"], "stu+clip")
    optimize(["stu12", "clip", "ep29"], "stu+clip+ep29")
    optimize(["stu12", "siglip2", "sq30", "ep29", "clip"], "stu+all4")
    optimize(["stu12", "sq30", "clip"], "stu+sq30+clip")


if __name__ == "__main__":
    main()
