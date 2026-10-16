#!/usr/bin/env python3
"""Combos with the distilled student (no-TTA), official protocol."""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(r"D:\bm\research\ensemble")
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from official_protocol import load_split, official_ranking  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)

# all no-TTA sets
NO = {
    "siglip2": "emb_siglip2_ep28_noTTA",
    "sq30": "emb_sq30_ep26_noTTA",
    "ep29": "emb_ep29_noTTA",
    "clip": "emb_clip_ep28_noTTA",
    "stu12": "emb_distill_a12_ep26_noTTA",
    "stu6": "emb_distill_a6_ep26_noTTA",
    "stu2": "emb_distill_a2_ep26_noTTA",
}


def L(p):
    return (np.load(HERE / f"{p}_qry.npy").astype("float64"),
            np.load(HERE / f"{p}_gal.npy").astype("float64"))


def ev(combo, w=None):
    q = [L(NO[k])[0] for k in combo]
    g = [L(NO[k])[1] for k in combo]
    if w is None:
        w = [1.0 / len(combo)] * len(combo)
    sim, _, _ = rp.rank(q, g, w, **D)
    return official_ranking(sim, *load_split())


def main():
    print("=== singles (no-TTA) ===")
    for k in NO:
        m = ev([k])
        print(f"  {k:8s} mAP@10 {m['mAP@10']:6.2f} R1 {m['Rank-1']:6.2f}")

    print("\n=== student + others (equal weights, no-TTA) ===")
    for combo in [("stu12", "clip"), ("stu12", "ep29"), ("stu12", "sq30"),
                  ("stu12", "clip", "ep29"), ("stu12", "siglip2"),
                  ("stu12", "sq30", "ep29", "clip"),
                  ("stu12", "siglip2", "sq30", "ep29", "clip")]:
        m = ev(combo)
        print(f"  {'+'.join(combo):34s} mAP@10 {m['mAP@10']:6.2f} R1 {m['Rank-1']:6.2f}")


if __name__ == "__main__":
    main()
