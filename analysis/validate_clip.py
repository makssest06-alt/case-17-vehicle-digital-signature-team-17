#!/usr/bin/env python3
"""Honest validation of adding CLIP as a 4th model.

- fixed configs on held-out vehicle halves (multiple seed ranges)
- tune weights on seeds 0..19, validate on 100..139
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402
from validate_ensemble import load_split, official_metrics  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)
BASE = ["emb_siglip2_ep28", "emb_sq30_ep26", "emb_ep29_fresh"]


def L(p):
    return (np.load(HERE / f"{p}_qry.npy").astype("float64"),
            np.load(HERE / f"{p}_gal.npy").astype("float64"))


def w4(x):
    r = 1 - x
    return [0.45 * r, x, 0.25 * r, 0.30 * r]


def main():
    qry, gal = load_split()
    vids = sorted({r["vehicle_id"] for r in qry})

    # precompute similarity for each config (base3, base3+clip at several x)
    cfgs = {"team3": (BASE, [0.45, 0.25, 0.30])}
    for x in (0.10, 0.15, 0.20, 0.25):
        cfgs[f"+clip{x:.2f}"] = (BASE + ["emb_clip_ep28"], w4(x))

    S = {}
    for name, (pres, w) in cfgs.items():
        qs = [L(p)[0] for p in pres]
        gs = [L(p)[1] for p in pres]
        sim, _, _ = rp.rank(qs, gs, list(w), **D)
        S[name] = sim
        m = official_metrics(sim, qry, gal)
        print(f"[full] {name:12s} mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f} R5 {m['R5']:6.2f}",
              flush=True)

    def halves(lo, hi):
        out = []
        for seed in range(lo, hi):
            rng = np.random.default_rng(seed)
            order = vids[:]
            rng.shuffle(order)
            half = set(order[: len(order) // 2])
            idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
            out.append((idx, [qry[i] for i in idx]))
        return out

    for (lo, hi) in ((0, 40), (200, 240), (500, 540)):
        fs = halves(lo, hi)
        base = np.array([official_metrics(S["team3"][i], rows, gal)["mAP"]
                         for i, rows in fs])
        print(f"\n=== held-out halves seeds {lo}..{hi} (base {base.mean():.2f}) ===")
        for name in cfgs:
            if name == "team3":
                continue
            v = np.array([official_metrics(S[name][i], rows, gal)["mAP"]
                          for i, rows in fs])
            d = v - base
            print(f"  {name:12s} {v.mean():6.2f}  delta {d.mean():+.2f} ± {d.std():.2f}, "
                  f"better {(d>0).sum()}/{len(d)}", flush=True)


if __name__ == "__main__":
    main()
