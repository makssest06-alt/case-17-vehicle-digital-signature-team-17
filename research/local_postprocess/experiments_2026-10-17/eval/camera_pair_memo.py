"""Does A10 rely on memorised camera pairs?

For every holdout (query, cross-camera positive) pair, count how many TRAIN vehicles were seen on
both cameras (camera-pair frequency in training). If the model learned camera-pair-specific
transformations, positives on rarely/never co-observed camera pairs should rank much worse.
Raw A10 cosine, holdout gallery 361 and 1111 (+ test gallery distractors), junk removed.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}


def rows(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    train = rows(D / "train.csv")
    split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
    hold_v = {r["vehicle_id"] for r in split}
    cams_of = defaultdict(set)
    for r in train:
        if r["vehicle_id"] not in hold_v:
            cams_of[r["vehicle_id"]].add(r["camera_id"])
    pair_freq: Counter = Counter()
    for cams in cams_of.values():
        pair_freq.update(frozenset(p) for p in combinations(sorted(cams), 2))

    seen, hg, hq = set(), [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        (hq if key in seen else hg).append(r)
        seen.add(key)
    z = np.load(D / "a10_clean_expanded.npz"); t = np.load(D / "a10_test.npz")
    emb = {str(i): e for i, e in zip(z["ids"].astype(str), z["emb"])}
    q = l2(np.stack([emb[r["image_id"]] for r in hq]))
    for name, extra in (("361", None), ("1111", l2(t["emb"][1110:]))):
        g = l2(np.stack([emb[r["image_id"]] for r in hg]))
        gv = [r["vehicle_id"] for r in hg]; gc = [r["camera_id"] for r in hg]
        if extra is not None:
            g = np.vstack([g, extra]); gv += ["x"] * len(extra); gc += ["x"] * len(extra)
        gv, gc = np.array(gv), np.array(gc)
        s = q @ g.T
        recs = []  # (train pair frequency, rank of this positive among non-junk)
        for i, r in enumerate(hq):
            junk = (gv == r["vehicle_id"]) & (gc == r["camera_id"])
            order = np.argsort(-np.where(junk, -np.inf, s[i]), kind="stable")
            rank_of = np.empty(len(order), int); rank_of[order] = np.arange(1, len(order) + 1)
            for j in np.flatnonzero((gv == r["vehicle_id"]) & (gc != r["camera_id"])):
                recs.append((pair_freq[frozenset((r["camera_id"], gc[j]))], rank_of[j]))
        a = np.array(recs)
        print(f"\n== gallery {name}: {len(a)} (query, positive) pairs")
        print(f"  {'train vehicles on this camera pair':36s} {'pairs':>6s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'median rank':>11s}")
        for lo, hi in ((0, 0), (1, 2), (3, 9), (10, 10**9)):
            m = (a[:, 0] >= lo) & (a[:, 0] <= hi)
            if m.any():
                rk = a[m, 1]
                label = f"{lo}..{hi}" if hi < 10**9 else f"{lo}+"
                print(f"  {label:36s} {m.sum():6d} {np.mean(rk <= 1):6.3f} {np.mean(rk <= 5):6.3f} "
                      f"{np.mean(rk <= 10):6.3f} {np.median(rk):11.1f}")
        print(f"  corr(log1p(pair freq), -log(rank)) = {np.corrcoef(np.log1p(a[:, 0]), -np.log(a[:, 1]))[0, 1]:.3f}")
    print(f"\ntrain: {len(cams_of)} vehicles, {len(pair_freq)} distinct camera pairs; "
          f"pair frequency quantiles 25/50/75/95: {np.percentile(list(pair_freq.values()), [25, 50, 75, 95])}")


if __name__ == "__main__":
    main()
