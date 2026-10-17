"""Head-to-head: raw / hubness / Ilya's DBA k=1 / camera penalty (Ilya's cheap background) /
near-duplicate rule, ranking + refusal, A10, organizer evaluate.py.

Protocol
- Holdout split as the organizers built test: first frame of each (vehicle, camera) -> gallery,
  later frames -> queries (546 q / 361 g).
- Galleries of UNSEEN vehicles: 361, 1111 (+ test gallery), 2.2k (+ all 1860 test frames);
  9005 (+ train-vehicle frames) only as a reference, it inflates gallery-side postprocessing.
- Ranking: every query has a cross-camera positive; top-10 of each variant, junk stripped by evaluator.
- Refusal ("as service"): 4 rotations; in each, 25 % of holdout vehicles keep only the gallery frame
  of one camera (their most frequent query camera), so their queries from that camera have only a
  same-camera frame left (open-set), the rest still have a pair. Threshold out-of-fold (2 folds by
  vehicle): max F1 subject to TNR >= 0.70 on one fold, applied to the other. candidates = top-1 of
  the variant's order when accepted; score = Ilya's z100 over unflagged top-100.
Ilya's code (dba, background, rerank) is imported unchanged from ../ilya_zip.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).parent
SCR = HERE.parent
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
EVALUATOR = Path(r"C:\Users\user\Downloads\evaluate.py")
sys.path.insert(0, str(SCR / "ilya_zip" / "api"))
from app.pp.background import service_descriptor  # noqa: E402
from app.pp.dba import dba  # noqa: E402
from app.pp.rerank import refusal_score  # noqa: E402

REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}
TAU_BG = 0.68127                     # Ilya's PP_CAMERA_TAU
DUP_COS = 0.94                       # near-duplicate rule: very high similarity ...
DUP_COS_BOX, DUP_IOU = 0.85, 0.70    # ... or high similarity + almost the same box (not tuned)
MIN_TNR = 0.70
PUSH = 1e3
Z_TOP = 100


def rows(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def hub(g: np.ndarray, k: int = 5) -> np.ndarray:
    out = np.empty(len(g), "float32")
    for s in range(0, len(g), 2048):
        S = g[s:s + 2048] @ g.T
        S[np.arange(len(S)), s + np.arange(len(S))] = -np.inf
        out[s:s + 2048] = np.partition(S, -k, axis=1)[:, -k:].mean(1)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (Q,4), b (G,4) as x,y,w,h -> (Q,G)."""
    ax1, ay1 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2, None] * a[:, 3, None] + b[None, :, 2] * b[None, :, 3] - inter)


def bg_cache(items: list[tuple[str, dict]]) -> dict[str, np.ndarray]:
    path = SCR / "h2h" / "bg_desc.npz"
    have: dict[str, np.ndarray] = {}
    if path.exists():
        z = np.load(path)
        have = dict(zip(z["ids"].astype(str), z["desc"]))
    todo = [(i, b) for i, b in items if i not in have]
    t0 = time.time()
    for n, (iid, b) in enumerate(todo):
        with Image.open(D / "dataset" / "images" / f"{iid}.jpg") as im:
            have[iid] = service_descriptor(im.convert("RGB"), (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])))
        if n % 1000 == 0:
            print(f"  bg {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    if todo:
        np.savez(path, ids=np.array(list(have)), desc=np.stack(list(have.values())))
    return have


def decision_metrics(acc: np.ndarray, has: np.ndarray, correct: np.ndarray) -> dict:
    """Vectorised evaluate.candidate_metrics (same TP/FP/FN/TN definitions, query level)."""
    tp = int((acc & has & correct).sum()); fp = int((acc & ~(has & correct)).sum())
    fn = int((~acc & has).sum()); tn = int((~acc & ~has).sum()); fp_open = int((acc & ~has).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"F1": 2 * p * r / (p + r) if p + r else 0.0, "TNR": tn / (tn + fp_open) if tn + fp_open else float("nan"),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


CHECKED = {"done": False}


def refusal_eval(ev, qids, qdf, gdf, gids, s, flag, keep, qfold) -> tuple[float, float, float]:
    """Out-of-fold F1/TNR/PR-AUC on one rotation's gallery (columns `keep`)."""
    kidx = np.flatnonzero(keep)
    sub_s, sub_fl = s[:, kidx], flag[:, kidx]
    order = np.argsort(-(sub_s - PUSH * sub_fl), axis=1, kind="stable")[:, :Z_TOP]
    score = np.array([refusal_score(sub_s[i, order[i]], sub_fl[i, order[i]], "z100") for i in range(len(qids))])
    top = [gids[kidx[order[i, 0]]] for i in range(len(qids))]
    gsub = gdf.iloc[kidx]
    gv = gsub.vehicle_id.to_numpy(); gc = gsub.camera_id.to_numpy()
    qv = qdf.vehicle_id.to_numpy(); qc = qdf.camera_id.to_numpy()
    has = np.array([((gv == qv[i]) & (gc != qc[i])).any() for i in range(len(qids))])
    gvid = dict(zip(gsub.index, gv))
    correct = np.array([gvid[top[i]] == qv[i] for i in range(len(qids))])

    f1s, tnrs, aucs = [], [], []
    for f in (0, 1):
        tune, held = qfold != f, qfold == f
        fin = score[tune & np.isfinite(score)]
        best_t, best_f1 = np.inf, -1.0
        for t in np.unique(np.quantile(fin, np.linspace(0, 1, 201))) if len(fin) else []:
            m = decision_metrics(score[tune] >= t, has[tune], correct[tune])
            if m["TNR"] >= MIN_TNR and m["F1"] > best_f1:
                best_t, best_f1 = t, m["F1"]
        m = decision_metrics(score[held] >= best_t, has[held], correct[held])
        if not CHECKED["done"]:  # one-time parity check against the organizer function
            c = {qid: [(top[i], float(score[i]))] for i, qid in enumerate(qids) if held[i] and score[i] >= best_t}
            o = ev.candidate_metrics(qdf[held], gsub, c)
            assert all(o[k] == m[k] for k in ("TP", "FP", "FN", "TN")), (o, m)
            print("   parity with evaluate.candidate_metrics: OK", flush=True)
            CHECKED["done"] = True
        f1s.append(m["F1"]); tnrs.append(m["TNR"])
        sh = np.where(np.isfinite(score[held]), score[held], -np.inf)
        aucs.append(ev.pr_auc(sh, has[held].astype(int)))
    return float(np.mean(f1s)), float(np.mean(tnrs)), float(np.nanmean(aucs))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ev", EVALUATOR)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    train = {r["image_id"]: r for r in rows(D / "train.csv")}
    split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
    seen, hold_g, hold_q = set(), [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        (hold_q if key in seen else hold_g).append(r)
        seen.add(key)
    tq, tgal = rows(D / "test_query.csv"), rows(D / "test_gallery.csv")
    meta = {**{r["image_id"]: r for r in tq + tgal}, **train}
    EMB_NPZ, OUT_JSON = Path(sys.argv[1]), Path(sys.argv[2])  # per-checkpoint embeddings (holdout + test gallery)
    z = np.load(EMB_NPZ)
    emb = {str(i): e for i, e in zip(z["ids"].astype(str), z["emb"])}

    qids = [r["image_id"] for r in hold_q]; hg = [r["image_id"] for r in hold_g]
    tg_ids = [r["image_id"] for r in tgal]; tq_ids = [r["image_id"] for r in tq]
    used = set(qids) | set(hg)
    tr_ids = [i for i in train if i not in REMOVED and i not in used]
    galleries = {"361 unseen": hg, "1111 unseen": hg + tg_ids}
    print("background descriptors ...", flush=True)
    bg = bg_cache([(i, meta[i]) for i in dict.fromkeys(qids + hg + tg_ids)])

    qdf = pd.DataFrame(hold_q).set_index("image_id")
    qv = qdf.vehicle_id.to_numpy(); qc = qdf.camera_id.to_numpy()
    q = l2(np.stack([emb[i] for i in qids])).astype("float32")
    qbg = np.stack([bg[i] for i in qids])
    qbox = np.array([[float(meta[i][k]) for k in "xywh"] for i in qids])
    vehicles = sorted(set(qv), key=int)
    rng = np.random.default_rng(42)
    rot_of = {v: n % 4 for n, v in enumerate(rng.permutation(vehicles))}
    fold_of = {v: n % 2 for n, v in enumerate(rng.permutation(vehicles))}
    keep_cam = {v: Counter(qc[qv == v]).most_common(1)[0][0] for v in vehicles}
    qfold = np.array([fold_of[v] for v in qv])

    report: dict = {}
    for gname, gids in galleries.items():
        t0 = time.time()
        g = l2(np.stack([emb[i] for i in gids])).astype("float32")
        gbg = np.stack([bg[i] for i in gids])
        gbox = np.array([[float(meta[i][k]) for k in "xywh"] for i in gids])
        gv = np.array([train[i]["vehicle_id"] if i in train else "x_" + i for i in gids])
        gc = np.array([train[i]["camera_id"] if i in train else "x_" + i for i in gids])
        gdf = pd.DataFrame({"image_id": gids, "vehicle_id": gv, "camera_id": gc}).set_index("image_id")
        raw = q @ g.T
        s_dba = q @ dba(g, k=1, weights="cos").T
        s_hub = raw - 0.5 * hub(g)[None, :]
        none = np.zeros_like(raw, bool)
        flag_bg = (qbg @ gbg.T) > TAU_BG
        flag_dup = (raw >= DUP_COS) | ((raw >= DUP_COS_BOX) & (iou(qbox, gbox) >= DUP_IOU))
        variants = {
            "raw": (raw, none), "hub0.5": (s_hub, none),
            "DBA1+bg (Ilya preset)": (s_dba, flag_bg), "hub0.5+bg": (s_hub, flag_bg),
        }
        rotations = []
        for rot in range(4):
            open_v = {v for v in vehicles if rot_of[v] == rot}
            rotations.append(np.array([not (v in open_v and c != keep_cam[v]) for v, c in zip(gv, gc)]))
        res = {}
        for vname, (s, flag) in variants.items():
            order = np.argsort(-(s - PUSH * flag), axis=1, kind="stable")[:, :10]
            rk = ev.ranking_metrics(qdf, gdf, {qid: [gids[j] for j in order[i]] for i, qid in enumerate(qids)}, top_k=10)
            ref = np.array([refusal_eval(ev, qids, qdf, gdf, gids, s, flag, keep, qfold) for keep in rotations])
            res[vname] = {"mAP@10": rk["mAP@10"], "R1": rk["Rank-1"], "R5": rk["Rank-5"],
                          "F1": float(ref[:, 0].mean()), "TNR": float(ref[:, 1].mean()), "PR-AUC": float(ref[:, 2].mean())}
        base = res["raw"]["mAP@10"]
        print(f"\n== gallery {gname} ({len(gids)} frames, {time.time() - t0:.0f}s)")
        print(f"   {'variant':24s} {'mAP@10':>8s} {'Δ':>7s} {'R1':>6s} {'R5':>6s} | {'F1':>6s} {'TNR':>6s} {'PR-AUC':>6s}")
        for k, r in res.items():
            print(f"   {k:24s} {r['mAP@10']:8.4f} {r['mAP@10'] - base:+7.4f} {r['R1']:6.3f} {r['R5']:6.3f}"
                  f" | {r['F1']:6.3f} {r['TNR']:6.3f} {r['PR-AUC']:6.3f}", flush=True)
        report[gname] = res

    # flag quality against the true camera, holdout pairs
    hgv = np.array([train[i]["vehicle_id"] for i in hg]); hgc = np.array([train[i]["camera_id"] for i in hg])
    hbg = np.stack([bg[i] for i in hg]); hbox = np.array([[float(meta[i][k]) for k in "xywh"] for i in hg])
    rawh = q @ l2(np.stack([emb[i] for i in hg])).T
    same_v = qv[:, None] == hgv[None]; same_c = qc[:, None] == hgc[None]
    for name, fl in (("bg tau", (qbg @ hbg.T) > TAU_BG),
                     ("dup rule", (rawh >= DUP_COS) | ((rawh >= DUP_COS_BOX) & (iou(qbox, hbox) >= DUP_IOU)))):
        print(f"{name:9s}: flags junk {fl[same_v & same_c].mean():.3f} | same-cam other car {fl[same_c & ~same_v].mean():.3f}"
              f" | true positives {fl[same_v & ~same_c].mean():.4f} | other-cam negatives {fl[~same_c & ~same_v].mean():.4f}")
    OUT_JSON.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
