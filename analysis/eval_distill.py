#!/usr/bin/env python3
"""Evaluate distilled students (official protocol) vs baselines."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(r"D:\bm\research\ensemble")
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from official_protocol import load_split, official_ranking  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)


def L(p):
    return (np.load(HERE / f"{p}_qry.npy").astype("float64"),
            np.load(HERE / f"{p}_gal.npy").astype("float64"))


def main():
    qry, gal = load_split()
    tags = sys.argv[1:] if len(sys.argv) > 1 else [
        "emb_siglip2_ep28", "emb_distill_a2_ep26", "emb_distill_a2_ep28", "emb_distill_a2_ep30"]
    for t in tags:
        q, g = L(t)
        sim, _, _ = rp.rank([q], [g], [1.0], **D)
        m = official_ranking(sim, qry, gal)
        print(f"{t:26s} mAP@10 {m['mAP@10']:6.2f}  R1 {m['Rank-1']:6.2f}  R5 {m['Rank-5']:6.2f}")


if __name__ == "__main__":
    main()
