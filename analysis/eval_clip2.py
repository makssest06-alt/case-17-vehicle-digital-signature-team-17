#!/usr/bin/env python3
"""Add the 2nd CLIP (seed21) to the ensemble and check held-out halves."""
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
    vids = sorted({r["vehicle_id"] for r in qry})

    def run(pres, w):
        qs = [L(p)[0] for p in pres]
        gs = [L(p)[1] for p in pres]
        s, _, _ = rp.rank(qs, gs, list(w), **D)
        return s

    base3 = ["emb_siglip2_ep28", "emb_sq30_ep26", "emb_ep29_fresh"]
    s_team = run(base3, [0.45, 0.25, 0.30])
    s_opt4 = run(base3 + ["emb_clip_ep28"], [0.364, 0.232, 0.182, 0.222])

    # clip21 single
    for e in (26, 28, 30):
        s = run([f"emb_clip21_ep{e}"], [1.0])
        m = official_metrics(s, qry, gal)
        print(f"clip21 ep{e} single: mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f}")

    cfgs = {
        "team3": s_team,
        "opt4(clip)": s_opt4,
        "opt4+clip21 .08": run(base3 + ["emb_clip_ep28", "emb_clip21_ep28"],
                               [0.335, 0.214, 0.167, 0.204, 0.08]),
        "opt4+clip21 .12": run(base3 + ["emb_clip_ep28", "emb_clip21_ep28"],
                               [0.32, 0.205, 0.16, 0.195, 0.12]),
    }
    # clip average (clip + clip21)/2 as single 4th
    qca = 0.5 * (L("emb_clip_ep28")[0] + L("emb_clip21_ep28")[0])
    gca = 0.5 * (L("emb_clip_ep28")[1] + L("emb_clip21_ep28")[1])
    s = run(["q"], [1.0]) if False else None
    # build via rank on the 3 base + averaged clip
    q = 0.364 * L("emb_siglip2_ep28")[0] + 0.232 * L("emb_sq30_ep26")[0] \
        + 0.182 * L("emb_ep29_fresh")[0] + 0.222 * qca
    g = 0.364 * L("emb_siglip2_ep28")[1] + 0.232 * L("emb_sq30_ep26")[1] \
        + 0.182 * L("emb_ep29_fresh")[1] + 0.222 * gca
    sim, _, _ = rp.rank([q], [g], [1.0], **D)
    cfgs["opt4 clip-avg(clip,clip21)"] = sim

    for name, s in cfgs.items():
        m = official_metrics(s, qry, gal)
        line = f"{name:28s} mAP {m['mAP']:6.2f} R1 {m['R1']:6.2f}"
        d = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            order = vids[:]
            rng.shuffle(order)
            half = set(order[: len(order) // 2])
            idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
            rows = [qry[i] for i in idx]
            d.append(official_metrics(s[idx], rows, gal)["mAP"] -
                     official_metrics(s_team[idx], rows, gal)["mAP"])
        d = np.array(d)
        line += f" | delta {d.mean():+.2f}±{d.std():.2f} better {(d>0).sum()}/200"
        print(line, flush=True)


if __name__ == "__main__":
    main()
