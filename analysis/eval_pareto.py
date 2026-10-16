#!/usr/bin/env python3
"""Pareto: official mAP vs compute for single/team3/ens4, with and without TTA."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(r"D:\bm\research\ensemble")
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from official_protocol import load_split, official_ranking  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)

# tag -> (tta_file_prefix, notta_file_prefix)
FILES = {
    "siglip2": ("emb_siglip2_ep28", "emb_siglip2_ep28_noTTA"),
    "sq30": ("emb_sq30_ep26", "emb_sq30_ep26_noTTA"),
    "ep29": ("emb_ep29_fresh", "emb_ep29_noTTA"),
    "clip": ("emb_clip_ep28", "emb_clip_ep28_noTTA"),
}
WEIGHTS = {"siglip2": 0.364, "sq30": 0.232, "ep29": 0.182, "clip": 0.222}


def L(prefix):
    return (np.load(HERE / f"{prefix}_qry.npy").astype("float64"),
            np.load(HERE / f"{prefix}_gal.npy").astype("float64"))


def run(keys, tta, w):
    q = [L(FILES[k][0 if tta else 1])[0] for k in keys]
    g = [L(FILES[k][0 if tta else 1])[1] for k in keys]
    sim, _, _ = rp.rank(q, g, list(w), **D)
    return sim


def main():
    qry, gal = load_split()
    configs = [
        ("single_siglip2", ["siglip2"], [1.0]),
        ("single_sq30", ["sq30"], [1.0]),
        ("single_clip", ["clip"], [1.0]),
        ("team3 (A.)", ["siglip2", "sq30", "ep29"], [0.45, 0.25, 0.30]),
        ("ens4 (+clip)", ["siglip2", "sq30", "ep29", "clip"], [0.364, 0.232, 0.182, 0.222]),
    ]
    print(f"{'config':16s} {'TTA':5s} {'mAP@10':>7s} {'R1':>6s} {'R5':>6s}")
    for name, keys, w in configs:
        for tta in (True, False):
            m = official_ranking(run(keys, tta, w), qry, gal)
            print(f"{name:16s} {str(tta):5s} {m['mAP@10']:7.2f} "
                  f"{m['Rank-1']:6.2f} {m['Rank-5']:6.2f}")


if __name__ == "__main__":
    main()
