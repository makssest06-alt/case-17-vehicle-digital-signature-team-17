#!/usr/bin/env python3
"""Re-implementation of the ORGANIZERS' evaluate.py ranking metrics.

Protocol (from the official evaluate.py):
  - remove junk = gallery items with SAME vehicle_id AND SAME camera_id as query
    (same-camera DIFFERENT-vehicle items stay as valid negatives)
  - open-set queries (no valid positive after junk removal) excluded from mAP/Rank
  - AP@10 normalised by min(n_pos, 10)
  - ties resolved by submission order

Compare against the team's "whole-camera" protocol.
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reid_postprocess_andrey as rp  # noqa: E402

D = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)
TOP_K = 10


def load_split():
    rows = list(csv.DictReader((HERE / "val_split.csv").open(encoding="utf-8")))
    gal = [r for r in rows if r["role"] == "gallery"]
    qry = [r for r in rows if r["role"] == "query"]
    return qry, gal


def official_ranking(sim, qry, gal, top_k=TOP_K):
    """Exactly the organizers' ranking_metrics."""
    g_vid = [r["vehicle_id"] for r in gal]
    g_cam = [r["camera_id"] for r in gal]
    aps, hits = [], {1: [], 5: []}
    n_openset, n_missing = 0, 0
    for i, q in enumerate(qry):
        # valid positives: same vid, NOT same cam
        n_pos = sum(1 for j in range(len(gal))
                    if g_vid[j] == q["vehicle_id"] and g_cam[j] != q["camera_id"])
        if n_pos == 0:
            n_openset += 1
            continue
        # rank all gallery by sim, then strip junk (same vid AND same cam), then top_k
        order = np.argsort(-sim[i], kind="stable")
        clean = [j for j in order
                 if not (g_vid[j] == q["vehicle_id"] and g_cam[j] == q["camera_id"])][:top_k]
        rel = np.array([g_vid[j] == q["vehicle_id"] for j in clean], dtype=bool)
        if rel.any():
            cum = np.cumsum(rel)
            prec = cum / (np.arange(len(rel)) + 1)
            aps.append(float((prec * rel).sum() / min(n_pos, top_k)))
        else:
            aps.append(0.0)
        for k in (1, 5):
            hits[k].append(bool(rel[:k].any()))
    return {
        "mAP@10": round(float(np.mean(aps)) * 100, 2) if aps else 0.0,
        "Rank-1": round(float(np.mean(hits[1])) * 100, 2) if hits[1] else 0.0,
        "Rank-5": round(float(np.mean(hits[5])) * 100, 2) if hits[5] else 0.0,
        "n_scored": len(aps), "n_openset": n_openset,
    }


def wholecam_ranking(sim, qry, gal):
    """Team's protocol: drop the WHOLE query camera, full AP."""
    g_vid = np.array([r["vehicle_id"] for r in gal])
    g_cam = np.array([r["camera_id"] for r in gal])
    aps, r1, r5, used = [], 0, 0, 0
    for i, q in enumerate(qry):
        keep = g_cam != q["camera_id"]
        s = sim[i][keep]
        rel = (g_vid[keep] == q["vehicle_id"])
        if not rel.any():
            continue
        used += 1
        order = np.argsort(-s)
        rel_s = rel[order]
        hits = np.flatnonzero(rel_s)
        aps.append(float(((np.arange(len(hits)) + 1) / (hits + 1)).mean()))
        r1 += int(rel_s[0]); r5 += int(rel_s[:5].any())
    return {"mAP": round(float(np.mean(aps)) * 100, 2),
            "R1": round(r1 / used * 100, 2), "R5": round(r5 / used * 100, 2), "n": used}


def main():
    qry, gal = load_split()
    sets = {n: (np.load(HERE / f"emb_{n}_qry.npy").astype("float64"),
                np.load(HERE / f"emb_{n}_gal.npy").astype("float64"))
            for n in ("siglip2_ep28", "sq30_ep26")}
    sets["ep29"] = (np.load(HERE / "emb_ep29_fresh_qry.npy").astype("float64"),
                    np.load(HERE / "emb_ep29_fresh_gal.npy").astype("float64"))
    sets["clip"] = (np.load(HERE / "emb_clip_ep28_qry.npy").astype("float64"),
                    np.load(HERE / "emb_clip_ep28_gal.npy").astype("float64"))

    configs = {
        "team3": (["siglip2_ep28", "sq30_ep26", "ep29"], [0.45, 0.25, 0.30]),
        "ens4": (["siglip2_ep28", "sq30_ep26", "ep29", "clip"], [0.364, 0.232, 0.182, 0.222]),
    }
    for name, (names, w) in configs.items():
        q = [sets[n][0] for n in names]
        g = [sets[n][1] for n in names]
        sim, _, _ = rp.rank(q, g, list(w), **D)
        off = official_ranking(sim, qry, gal)
        wc = wholecam_ranking(sim, qry, gal)
        print(f"{name}:")
        print(f"  OFFICIAL (junk-only): mAP@10 {off['mAP@10']:6.2f}  R1 {off['Rank-1']:6.2f}  "
              f"R5 {off['Rank-5']:6.2f}  (scored {off['n_scored']}, openset {off['n_openset']})")
        print(f"  team (whole-camera) : mAP {wc['mAP']:6.2f}  R1 {wc['R1']:6.2f}  "
              f"R5 {wc['R5']:6.2f}  (n {wc['n']})")


if __name__ == "__main__":
    main()
