#!/usr/bin/env python3
"""Final metrics: full AP and official mAP@10 for team3 vs opt4 (with CLIP)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from validate_ensemble import load_split, official_metrics  # noqa: E402
from eval10 import map10  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)
NAMES = ["emb_siglip2_ep28", "emb_sq30_ep26", "emb_ep29_fresh", "emb_clip_ep28"]


def main():
    qry, gal = load_split()
    q = [np.load(HERE / f"{n}_qry.npy").astype("float64") for n in NAMES]
    g = [np.load(HERE / f"{n}_gal.npy").astype("float64") for n in NAMES]

    def sim(w):
        s, _, _ = rp.rank(q, g, list(w), **D)
        return s

    for name, w in (("team3", [0.45, 0.25, 0.30, 0.0]),
                    ("opt4+clip", [0.364, 0.232, 0.182, 0.222])):
        s = sim(w)
        m = official_metrics(s, qry, gal)
        m10, r1, _ = map10(s, qry, gal)
        print(f"{name:10s} fullAP mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f} "
              f"R5 {m['R5']:6.2f} | mAP@10 {m10:6.2f} R1 {r1:6.2f}")


if __name__ == "__main__":
    main()
