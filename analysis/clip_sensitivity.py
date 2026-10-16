#!/usr/bin/env python3
"""Sensitivity of the CLIP weight: mean delta vs teammate over many half-splits."""
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
    nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    qry, gal = load_split()
    vids = sorted({r["vehicle_id"] for r in qry})
    q = [np.load(HERE / f"{p}_qry.npy").astype("float64") for p in NAMES]
    g = [np.load(HERE / f"{p}_gal.npy").astype("float64") for p in NAMES]

    def sim(w):
        s, _, _ = rp.rank(q, g, list(w), **D)
        return s

    xs = [0.0, 0.10, 0.15, 0.20, 0.224, 0.25, 0.30]
    cfgs = {}
    for x in xs:
        r = 1 - x
        cfgs[x] = sim([0.45 * r, 0.25 * r, 0.30 * r, x])

    for x in xs:
        m = official_metrics(cfgs[x], qry, gal)
        print(f"x={x:<5} full mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f}")

    base = cfgs[0.0]
    print(f"\n=== mean delta vs x=0 over {nseed} half-splits ===")
    for x in xs[1:]:
        d = []
        for seed in range(nseed):
            rng = np.random.default_rng(seed)
            order = vids[:]
            rng.shuffle(order)
            half = set(order[: len(order) // 2])
            idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
            rows = [qry[i] for i in idx]
            a = official_metrics(base[idx], rows, gal)["mAP"]
            b = official_metrics(cfgs[x][idx], rows, gal)["mAP"]
            d.append(b - a)
        d = np.array(d)
        print(f"x={x:<5} delta {d.mean():+.2f} ± {d.std():.2f}, "
              f"better {(d>0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
