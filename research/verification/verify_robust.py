#!/usr/bin/env python3
"""Robustness check of the ensemble claim across many val splits.

Reuses embeddings already computed by verify_team17.py (saved as .npy) and
re-partitions the same 150 identities / 912 frames into new splits: for every
(vehicle, camera) group a random frame becomes gallery, the rest become queries.
No re-embedding needed.
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import verify_team17 as V

HERE = Path(__file__).parent
SETS = {
    "geom_squash": "v17_geom_ep30_squash",
    "geom_timm": "v17_geom_ep30_timm",
    "ep29_timm": "v17_ep29_timm",
}


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def load_embeddings(rows):
    """Return {tag: (emb_all, index_by_image_id)} in the row order of `rows`."""
    order = [r["image_id"] for r in rows if r["role"] == "query"] + [r["image_id"] for r in rows if r["role"] == "gallery"]
    out = {}
    for tag, stem in SETS.items():
        q = np.load(HERE / f"{stem}_qry.npy")
        g = np.load(HERE / f"{stem}_gal.npy")
        emb = l2(np.vstack([q, g]).astype("float64"))
        out[tag] = {iid: emb[i] for i, iid in enumerate(order)}
    return out


def make_split(rows, seed):
    by_vc = defaultdict(list)
    for r in rows:
        by_vc[(r["vehicle_id"], r["camera_id"])].append(r)
    rng = random.Random(seed)
    gal, qry = [], []
    for key in sorted(by_vc, key=lambda k: (int(k[0]), int(k[1]))):
        items = sorted(by_vc[key], key=lambda r: r["image_id"])
        gi = rng.randrange(len(items))
        gal.append(items[gi])
        qry.extend(items[j] for j in range(len(items)) if j != gi)
    return qry, gal


def main():
    rows = list(csv.DictReader((HERE / "val_split.csv").open(encoding="utf-8")))
    emb = load_embeddings(rows)
    tags = list(SETS)

    # config: (label, [(tag, weight)...], postprocess?)
    cfgs = [
        ("base geom_timm (no pp)", [("geom_timm", 1.0)], False),
        ("geom_squash +pp", [("geom_squash", 1.0)], True),
        ("geom 0.8 squash + 0.2 base +pp", [("geom_squash", 0.8), ("geom_timm", 0.2)], True),
        ("geom-sq0.8 + ep29-base0.2 +pp", [("geom_squash", 0.8), ("ep29_timm", 0.2)], True),
        ("geom-sq0.7 + ep29-base0.3 +pp", [("geom_squash", 0.7), ("ep29_timm", 0.3)], True),
    ]

    acc = {c[0]: [] for c in cfgs}
    wins = {c[0]: 0 for c in cfgs}
    n = 20
    for s in range(n):
        qry, gal = make_split(rows, 1000 + s)
        res = {}
        for label, parts, pp in cfgs:
            q = l2(sum(w * np.array([emb[t][r["image_id"]] for r in qry]) for t, w in parts))
            g = l2(sum(w * np.array([emb[t][r["image_id"]] for r in gal]) for t, w in parts))
            if pp:
                sim = V.pipeline([q], [g], [1.0])
            else:
                sim = q @ g.T
            res[label] = V.metrics(sim, qry, gal, "ours")["mAP"]
        for label, _, _ in cfgs:
            acc[label].append(res[label])
        best = max(res, key=res.get)
        for label in res:
            if res[label] == res[best]:
                wins[label] += 1
        if s == 0:
            print("split #1 mAP:", {k: round(v, 2) for k, v in res.items()}, flush=True)

    print(f"\n{'config':34s} {'mean mAP':>9s} {'min':>7s} {'max':>7s} {'wins/20':>8s}")
    print("-" * 72)
    for label, _, _ in cfgs:
        a = np.array(acc[label])
        print(f"{label:34s} {a.mean():9.2f} {a.min():7.2f} {a.max():7.2f} {wins[label]:8d}")

    # paired gain of the final ensemble over the baseline
    base = np.array(acc["base geom_timm (no pp)"])
    for label in ("geom_squash +pp", "geom-sq0.8 + ep29-base0.2 +pp",
                  "geom-sq0.7 + ep29-base0.3 +pp"):
        d = np.array(acc[label]) - base
        print(f"gain over baseline [{label}]: mean {d.mean():+.2f}  "
              f"min {d.min():+.2f}  max {d.max():+.2f}  positive {int((d > 0).sum())}/20")
    prev = np.array(acc["geom_squash +pp"])
    for label in ("geom-sq0.8 + ep29-base0.2 +pp", "geom-sq0.7 + ep29-base0.3 +pp"):
        d = np.array(acc[label]) - prev
        print(f"gain over prev best [{label}]: mean {d.mean():+.2f}  "
              f"min {d.min():+.2f}  max {d.max():+.2f}  positive {int((d > 0).sum())}/20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


