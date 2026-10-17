"""Diagnostic for local_rerank.py: inside the raw top-20 (holdout 361 gallery, head tokens),
compare global and local (maxsim) scores of cross-camera positives vs same-camera negatives
vs other-camera negatives, and the pairwise AUC positive-vs-negative of each score."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

import local_rerank as lr


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    z = np.load(lr.HERE / "tokens.npz")
    ids = z["ids"].astype(str).tolist(); pos = {i: k for k, i in enumerate(ids)}
    train = {r["image_id"]: r for r in lr.rows(lr.D / "train.csv")}
    split = [r for r in lr.rows(lr.D / "val_split.csv") if r["image_id"] not in lr.REMOVED]
    seen, hold_g, hold_q = set(), [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        (hold_q if key in seen else hold_g).append(r)
        seen.add(key)
    qi = [pos[r["image_id"]] for r in hold_q]; gi = [pos[r["image_id"]] for r in hold_g]
    qv = np.array([r["vehicle_id"] for r in hold_q]); qc = np.array([r["camera_id"] for r in hold_q])
    gv = np.array([r["vehicle_id"] for r in hold_g]); gc = np.array([r["camera_id"] for r in hold_g])
    glob = z["glob"]; raw = glob[qi] @ glob[gi].T
    cand = np.argsort(-raw, axis=1, kind="stable")[:, :lr.TOP_K]
    tok = torch.from_numpy(z["head_tok"]).to(lr.DEV)
    loc = lr.local_scores(tok[qi], tok[gi], cand, torch.from_numpy(glob[qi]).to(lr.DEV), lr.inner_mask().to(lr.DEV))["maxsim"]
    gsc = np.take_along_axis(raw, cand, axis=1)
    same_v = gv[cand] == qv[:, None]; same_c = gc[cand] == qc[:, None]
    groups = {"positive (same car, other cam)": same_v & ~same_c, "junk (same car, same cam)": same_v & same_c,
              "negative, same cam": ~same_v & same_c, "negative, other cam": ~same_v & ~same_c}
    # z-score within each query's top-K so scales are comparable
    zs = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
    gz, lz = zs(gsc), zs(loc)
    print(f"{'group in raw top-20':32s} {'count':>6s} {'global z':>9s} {'local z':>9s}")
    for k, m in groups.items():
        print(f"{k:32s} {int(m.sum()):6d} {gz[m].mean():9.3f} {lz[m].mean():9.3f}")
    p, n = groups["positive (same car, other cam)"], groups["negative, other cam"] | groups["negative, same cam"]
    print(f"\nAUC positive vs negative: global {auc(gz[p], gz[n]):.3f}  local {auc(lz[p], lz[n]):.3f}")
    ns = groups["negative, same cam"]
    print(f"AUC positive vs same-cam negative: global {auc(gz[p], gz[ns]):.3f}  local {auc(lz[p], lz[ns]):.3f}")
    print(f"corr(global z, local z) over all top-20 pairs: {np.corrcoef(gz.ravel(), lz.ravel())[0, 1]:.3f}")


if __name__ == "__main__":
    main()
