"""Follow-up to oracle_ceiling.py: hubness beta stability and hub + soft pseudo-camera combos,
only on galleries of unseen vehicles (holdout 361, holdout + test gallery 1111)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import oracle_ceiling as oc


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", oc.EVALUATOR)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    train = {r["image_id"]: r for r in oc.rows(oc.DATA / "train.csv")}
    pseudo = {r["image_id"]: r["pseudo_cam"] for r in oc.rows(oc.PSEUDO)}
    split = [r for r in oc.rows(oc.DATA / "val_split.csv") if r["image_id"] not in oc.REMOVED]
    seen, hold_g, hold_q = set(), [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        (hold_q if key in seen else hold_g).append(r)
        seen.add(key)
    clean = np.load(oc.DATA / "a10_clean_expanded.npz"); test = np.load(oc.DATA / "a10_test.npz")
    emb = {str(i): e for i, e in zip(clean["ids"].astype(str), clean["emb"])}
    qids = [r["image_id"] for r in hold_q]; hg = [r["image_id"] for r in hold_g]
    tg = test["ids"].astype(str)[1110:].tolist()
    q = oc.norm(np.stack([emb[i] for i in qids]))
    hgv = oc.norm(np.stack([emb[i] for i in hg])); tgv = oc.norm(test["emb"][1110:])
    qdf = pd.DataFrame(hold_q).set_index("image_id")
    qc = np.array([pseudo[i] for i in qids])
    for name, g, gids in (("holdout_361", hgv, hg), ("holdout+test_1111", np.vstack([hgv, tgv]), hg + tg)):
        recs = [{"image_id": i, "vehicle_id": train[i]["vehicle_id"], "camera_id": train[i]["camera_id"]} if i in train
                else {"image_id": i, "vehicle_id": "x_" + i, "camera_id": "x_" + i} for i in gids]
        gdf = pd.DataFrame(recs).set_index("image_id")
        gc = np.array([pseudo[i] for i in gids])
        same = qc[:, None] == gc[None, :]
        raw = q @ g.T
        v = {"raw": raw}
        for k in (3, 5, 10):
            h = oc.hub(g, k)
            for b in (0.5, 1.0, 1.5):
                v[f"hub_k{k}_b{b}"] = raw - b * h[None, :]
        h5 = oc.hub(g, 5)[None, :]
        for p in (0.03, 0.05, 0.1):
            v[f"hub_k5_b1+campen{p}[pseudo]"] = raw - h5 - p * same
        gd = oc.dba(g, 1)
        v["dba1+hub_k5"] = q @ gd.T - oc.hub(gd, 5)[None, :]
        v["dba1+hub_k5+campen0.05"] = v["dba1+hub_k5"] - 0.05 * same
        base = None
        print(f"\n== {name}")
        for k, s in v.items():
            order = np.argsort(-s, axis=1, kind="stable")[:, :10]
            m = ev.ranking_metrics(qdf, gdf, {qid: [gids[j] for j in order[i]] for i, qid in enumerate(qids)}, top_k=10)
            base = m["mAP@10"] if base is None else base
            print(f"  {k:30s} mAP@10 {m['mAP@10']:.4f} ({m['mAP@10'] - base:+.4f})  R1 {m['Rank-1']:.4f}  R5 {m['Rank-5']:.4f}")
        # where do raw positives sit: how much is left on the table for any reranker of the top-10?
        gv = gdf.vehicle_id.to_numpy(); gcam = gdf.camera_id.astype(str).to_numpy()
        order = np.argsort(-raw, axis=1, kind="stable")
        first_pos = []
        for i, qid in enumerate(qids):
            row = qdf.loc[qid]
            pos = (gv == row.vehicle_id) & (gcam != str(row.camera_id))
            ranks = np.flatnonzero(pos[order[i]])
            first_pos.append(ranks[0] + 1 if len(ranks) else 10**6)
        fp = np.array(first_pos)
        print(f"  first true positive rank (raw): r1 {np.mean(fp == 1):.3f}  2-10 {np.mean((fp > 1) & (fp <= 10)):.3f}"
              f"  11-50 {np.mean((fp > 10) & (fp <= 50)):.3f}  >50 {np.mean(fp > 50):.3f}")


if __name__ == "__main__":
    main()
