#!/usr/bin/env python3
"""Validate student-based configs (fixed weights) on held-out halves."""
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
CFGS = {
    "team3 noTTA": (["siglip2", "sq30", "ep29"], [0.45, 0.25, 0.30]),
    "ens4 noTTA": (["siglip2", "sq30", "ep29", "clip"], [0.364, 0.232, 0.182, 0.222]),
    "stu12 alone": (["stu12"], [1.0]),
    "stu+sq30": (["stu12", "sq30"], [0.457, 0.543]),
    "stu+sq30+ep29": (["stu12", "sq30", "ep29"], [0.394, 0.303, 0.303]),
    "stu+all4": (["stu12", "siglip2", "sq30", "ep29", "clip"],
                 [0.296, 0.143, 0.200, 0.165, 0.197]),
}


def main():
    lo = int(sys.argv[1]); hi = int(sys.argv[2])
    qry, gal = load_split()
    vids = sorted({r["vehicle_id"] for r in qry})

    def sim(keys, w):
        q = [np.load(HERE / f"{NO[k]}_qry.npy").astype("float64") for k in keys]
        g = [np.load(HERE / f"{NO[k]}_gal.npy").astype("float64") for k in keys]
        s, _, _ = rp.rank(q, g, list(w), **D)
        return s

    S = {name: sim(keys, w) for name, (keys, w) in CFGS.items()}
    for name, s in S.items():
        m = official_ranking(s, qry, gal)
        print(f"[full] {name:16s} mAP@10 {m['mAP@10']:6.2f} R1 {m['Rank-1']:6.2f}")

    base = S["team3 noTTA"]
    print(f"\n=== held-out halves seeds {lo}..{hi}, delta vs team3-noTTA ===")
    for name, s in S.items():
        if name == "team3 noTTA":
            continue
        d = []
        for seed in range(lo, hi):
            rng = np.random.default_rng(seed)
            order = vids[:]
            rng.shuffle(order)
            half = set(order[: len(order) // 2])
            idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
            rows = [qry[i] for i in idx]
            d.append(official_ranking(s[idx], rows, gal)["mAP@10"] -
                     official_ranking(base[idx], rows, gal)["mAP@10"])
        d = np.array(d)
        print(f"{name:16s} delta {d.mean():+.2f} ± {d.std():.2f}, better {(d>0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
