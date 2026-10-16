#!/usr/bin/env python3
"""Validate the optimized 4-model weights (with CLIP) vs teammate 3-model."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from validate_ensemble import load_split, official_metrics  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)
NAMES = ["emb_siglip2_ep28", "emb_sq30_ep26", "emb_ep29_fresh", "emb_clip_ep28"]
TEAM = [0.45, 0.25, 0.30, 0.0]
OPT = [0.358, 0.233, 0.185, 0.224]


def main():
    lo = int(sys.argv[1]); hi = int(sys.argv[2])
    qry, gal = load_split()
    vids = sorted({r["vehicle_id"] for r in qry})
    q = [np.load(HERE / f"{p}_qry.npy").astype("float64") for p in NAMES]
    g = [np.load(HERE / f"{p}_gal.npy").astype("float64") for p in NAMES]

    def sim(w):
        s, _, _ = rp.rank(q, g, list(w), **D)
        return s

    S = {"team3": sim(TEAM), "opt4": sim(OPT)}
    for k, s in S.items():
        m = official_metrics(s, qry, gal)
        print(f"[full] {k:6s} mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f} R5 {m['R5']:6.2f}")

    d = []
    for seed in range(lo, hi):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        half = set(order[: len(order) // 2])
        idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
        rows = [qry[i] for i in idx]
        a = official_metrics(S["team3"][idx], rows, gal)["mAP"]
        b = official_metrics(S["opt4"][idx], rows, gal)["mAP"]
        d.append(b - a)
    d = np.array(d)
    print(f"seeds {lo}..{hi}: delta {d.mean():+.2f} ± {d.std():.2f}, "
          f"better {(d>0).sum()}/{len(d)}, worst {d.min():+.2f}")


if __name__ == "__main__":
    main()
