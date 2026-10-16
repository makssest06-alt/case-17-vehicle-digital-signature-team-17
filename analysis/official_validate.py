#!/usr/bin/env python3
"""CLIP gain under the OFFICIAL protocol, on held-out vehicle halves."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from official_protocol import load_split, official_ranking  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)


def main():
    lo = int(sys.argv[1]); hi = int(sys.argv[2])
    qry, gal = load_split()
    names = ["siglip2_ep28", "sq30_ep26", "ep29", "clip"]
    pre = {"siglip2_ep28": "emb_siglip2_ep28", "sq30_ep26": "emb_sq30_ep26",
           "ep29": "emb_ep29_fresh", "clip": "emb_clip_ep28"}
    q = [np.load(HERE / f"{pre[n]}_qry.npy").astype("float64") for n in names]
    g = [np.load(HERE / f"{pre[n]}_gal.npy").astype("float64") for n in names]

    def sim(w):
        s, _, _ = rp.rank(q, g, list(w), **D)
        return s

    s3 = sim([0.45, 0.25, 0.30, 0.0])
    s4 = sim([0.364, 0.232, 0.182, 0.222])
    print("full official team3:", official_ranking(s3, qry, gal))
    print("full official ens4 :", official_ranking(s4, qry, gal))

    vids = sorted({r["vehicle_id"] for r in qry})
    d = []
    for seed in range(lo, hi):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        half = set(order[: len(order) // 2])
        idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
        rq = [qry[i] for i in idx]
        a = official_ranking(s3[idx], rq, gal)["mAP@10"]
        b = official_ranking(s4[idx], rq, gal)["mAP@10"]
        d.append(b - a)
    d = np.array(d)
    print(f"seeds {lo}..{hi}: CLIP delta (official) {d.mean():+.2f} ± {d.std():.2f}, "
          f"better {(d>0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
