"""Refusal working point for the organizer formula: score = 0.7 * F1 + 0.3 * TNR (FAQ Q15),
20 % open-set queries (FAQ Q17). Reuses head_to_head.py data and Ilya's functions.

Protocols
- 'as test'  : open-set vehicles keep only the gallery frame of one camera, so their queries from
               that camera see only a same-camera frame of themselves (closed test has the
               organizers' structure: one gallery frame per (vehicle, camera)).
- 'clean'    : additionally every query's same-vehicle-same-camera frame is hidden (the experts'
               'clean sample without irrelevant pairs').
5 rotations x 20 % open-set vehicles; threshold chosen out-of-fold (2 folds by vehicle) to
maximise 0.7*F1 + 0.3*TNR; we report the held-out objective and the chosen thresholds.
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import pandas as pd

import head_to_head as hh

ROT = 5


def objective(acc, has, correct) -> tuple[float, float, float]:
    m = hh.decision_metrics(acc, has, correct)
    tnr = m["TNR"] if np.isfinite(m["TNR"]) else 0.0
    return 0.7 * m["F1"] + 0.3 * tnr, m["F1"], tnr


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    train = {r["image_id"]: r for r in hh.rows(hh.D / "train.csv")}
    split = [r for r in hh.rows(hh.D / "val_split.csv") if r["image_id"] not in hh.REMOVED]
    seen, hold_g, hold_q = set(), [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        (hold_q if key in seen else hold_g).append(r)
        seen.add(key)
    tgal = hh.rows(hh.D / "test_gallery.csv")
    meta = {**{r["image_id"]: r for r in tgal}, **train}
    clean = np.load(hh.D / "a10_clean_expanded.npz"); test = np.load(hh.D / "a10_test.npz")
    emb = {str(i): e for i, e in zip(clean["ids"].astype(str), clean["emb"])}
    emb.update({str(i): e for i, e in zip(test["ids"].astype(str), test["emb"])})
    bg = hh.bg_cache([])
    qids = [r["image_id"] for r in hold_q]; hg = [r["image_id"] for r in hold_g]
    tg = [r["image_id"] for r in tgal]
    qdf = pd.DataFrame(hold_q).set_index("image_id")
    qv = qdf.vehicle_id.to_numpy(); qc = qdf.camera_id.to_numpy()
    q = hh.l2(np.stack([emb[i] for i in qids])).astype("float32")
    qbg = np.stack([bg[i] for i in qids])
    qbox = np.array([[float(meta[i][k]) for k in "xywh"] for i in qids])
    vehicles = sorted(set(qv), key=int)
    rng = np.random.default_rng(7)
    rot_of = {v: n % ROT for n, v in enumerate(rng.permutation(vehicles))}
    fold_of = {v: n % 2 for n, v in enumerate(rng.permutation(vehicles))}
    keep_cam = {v: Counter(qc[qv == v]).most_common(1)[0][0] for v in vehicles}
    qfold = np.array([fold_of[v] for v in qv])

    for gname, gids in (("361", hg), ("1111", hg + tg)):
        g = hh.l2(np.stack([emb[i] for i in gids])).astype("float32")
        gbg = np.stack([bg[i] for i in gids])
        gbox = np.array([[float(meta[i][k]) for k in "xywh"] for i in gids])
        gv = np.array([train[i]["vehicle_id"] if i in train else "x_" + i for i in gids])
        gc = np.array([train[i]["camera_id"] if i in train else "x_" + i for i in gids])
        raw = q @ g.T
        s_dba = q @ hh.dba(g, k=1, weights="cos").T
        s_hub = raw - 0.5 * hh.hub(g)[None, :]
        none = np.zeros_like(raw, bool)
        flag_bg = (qbg @ gbg.T) > hh.TAU_BG
        flag_dup = (raw >= hh.DUP_COS) | ((raw >= hh.DUP_COS_BOX) & (hh.iou(qbox, gbox) >= hh.DUP_IOU))
        junk = (qv[:, None] == gv[None]) & (qc[:, None] == gc[None])
        variants = {"raw (no flag)": (raw, none), "DBA1+bg (Ilya)": (s_dba, flag_bg),
                    "DBA1+dup": (s_dba, flag_dup), "DBA1+bg|dup": (s_dba, flag_bg | flag_dup),
                    "hub0.5+bg": (s_hub, flag_bg)}
        print(f"\n== gallery {gname}")
        print(f"   {'variant':16s} {'score':6s} | {'as test: obj':>12s} {'F1':>5s} {'TNR':>5s} {'thr (min..max)':>16s}"
              f" | {'clean: obj':>10s} {'F1':>5s} {'TNR':>5s} {'thr (min..max)':>16s}")
        for vname, (s, flag) in variants.items():
            for formula in ("raw", "z100"):
                cells = []
                for proto in ("as test", "clean"):
                    objs, f1s, tnrs, thrs = [], [], [], []
                    for rot in range(ROT):
                        open_v = {v for v in vehicles if rot_of[v] == rot}
                        keep = np.array([not (v in open_v and c != keep_cam[v]) for v, c in zip(gv, gc)])
                        kidx = np.flatnonzero(keep)
                        ss, fl = s[:, kidx].copy(), flag[:, kidx]
                        if proto == "clean":
                            ss[junk[:, kidx]] = -np.inf
                        order = np.argsort(-(ss - hh.PUSH * fl), axis=1, kind="stable")[:, :hh.Z_TOP]
                        sc = np.array([hh.refusal_score(np.where(np.isfinite(ss[i, order[i]]), ss[i, order[i]], -9.0),
                                                        fl[i, order[i]] | ~np.isfinite(ss[i, order[i]]), formula)
                                       for i in range(len(qids))])
                        gvk, gck = gv[kidx], gc[kidx]
                        has = np.array([((gvk == qv[i]) & (gck != qc[i])).any() for i in range(len(qids))])
                        correct = gvk[order[:, 0]] == qv
                        for f in (0, 1):
                            tune, held = qfold != f, qfold == f
                            fin = sc[tune & np.isfinite(sc)]
                            cand = np.unique(np.quantile(fin, np.linspace(0, 1, 201)))
                            best = max(cand, key=lambda t: objective(sc[tune] >= t, has[tune], correct[tune])[0])
                            o, f1, tnr = objective(sc[held] >= best, has[held], correct[held])
                            objs.append(o); f1s.append(f1); tnrs.append(tnr); thrs.append(best)
                    cells.append(f"{np.mean(objs):12.3f} {np.mean(f1s):5.3f} {np.mean(tnrs):5.3f} "
                                 f"{min(thrs):7.3f}..{max(thrs):7.3f}")
                print(f"   {vname:16s} {formula:6s} | {cells[0]} | {cells[1]}", flush=True)


if __name__ == "__main__":
    main()
