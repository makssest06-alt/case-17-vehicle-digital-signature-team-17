#!/usr/bin/env python3
"""Evaluate CLIP (4th model) alone and in the ensemble; full 6-set search."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from validate_ensemble import load_split, official_metrics  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)


def L(p):
    return (np.load(HERE / f"{p}_qry.npy").astype("float64"),
            np.load(HERE / f"{p}_gal.npy").astype("float64"))


def main():
    qry, gal = load_split()

    def run(pres, w):
        qs = [L(p)[0] for p in pres]
        gs = [L(p)[1] for p in pres]
        sim, _, _ = rp.rank(qs, gs, list(w), **D)
        return official_metrics(sim, qry, gal)

    print("=== CLIP single ===")
    for p in ("emb_clip_ep26", "emb_clip_ep28", "emb_clip_ep30"):
        m = run([p], [1.0])
        print(f"  {p:18s} mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f}")

    base = ["emb_siglip2_ep28", "emb_sq30_ep26", "emb_ep29_fresh"]
    print("\n=== teammate 3-model ===", run(base, [0.45, 0.25, 0.30]))

    print("\n=== 4-model: + CLIP x ===")
    for x in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        rest = 1 - x
        w = [0.45 * rest, x, 0.25 * rest, 0.30 * rest]
        m = run([base[0], "emb_clip_ep28", base[1], base[2]], w)
        print(f"  x={x:<4} mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f}")

    print("\n=== 5-model: + CLIP + best-other ===")
    for extra in ("emb_siglip2_s123_ep26", "emb_siglip2_ng_ep30", "emb_dinov2_ep26"):
        for x in (0.1, 0.2):
            rest = 1 - x
            # base3 scaled to (1-x) and extra gets x
            w = [0.45 * rest, 0.25 * rest, 0.30 * rest, x]
            m = run(base + [extra], w)
            print(f"  +{extra:22s} x={x}: mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f}")


if __name__ == "__main__":
    main()
