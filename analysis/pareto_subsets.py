#!/usr/bin/env python3
"""Pareto: official mAP vs #models (compute) over all subsets, no-TTA and TTA."""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(r"D:\bm\research\ensemble")
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from official_protocol import load_split, official_ranking  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)
FILES = {
    "siglip2": ("emb_siglip2_ep28", "emb_siglip2_ep28_noTTA"),
    "sq30": ("emb_sq30_ep26", "emb_sq30_ep26_noTTA"),
    "ep29": ("emb_ep29_fresh", "emb_ep29_noTTA"),
    "clip": ("emb_clip_ep28", "emb_clip_ep28_noTTA"),
}
TEAMW = {"siglip2": 0.45, "sq30": 0.25, "ep29": 0.30}


def L(prefix):
    return (np.load(HERE / f"{prefix}_qry.npy").astype("float64"),
            np.load(HERE / f"{prefix}_gal.npy").astype("float64"))


def main():
    qry, gal = load_split()
    keys_all = list(FILES)
    out = []
    for r in range(1, len(keys_all) + 1):
        for combo in itertools.combinations(keys_all, r):
            # weight: team weights if subset of team, else equal
            if set(combo) <= set(TEAMW):
                w = [TEAMW[k] for k in combo]
                w = [x / sum(w) for x in w]
            else:
                w = [1.0 / len(combo)] * len(combo)
            for tta in (False, True):
                q = [L(FILES[k][0 if tta else 1])[0] for k in combo]
                g = [L(FILES[k][0 if tta else 1])[1] for k in combo]
                sim, _, _ = rp.rank(q, g, w, **D)
                m = official_ranking(sim, qry, gal)
                compute = len(combo) * (2 if tta else 1)
                out.append((m["mAP@10"], compute, tta, combo, m["Rank-1"]))
    out.sort(key=lambda t: (-t[0]))
    print(f"{'mAP@10':>7s} {'R1':>6s} {'#m':>3s} {'TTA':>4s} {'compute':>7s}  models")
    for mapv, compute, tta, combo, r1 in out[:20]:
        print(f"{mapv:7.2f} {r1:6.2f} {len(combo):3d} {str(tta):5s} {compute:7d}  {combo}")


if __name__ == "__main__":
    main()
