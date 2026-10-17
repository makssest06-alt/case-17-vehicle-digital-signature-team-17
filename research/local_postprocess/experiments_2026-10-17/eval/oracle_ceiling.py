"""Oracle ceilings for camera-aware, gallery-only postprocessing on the A10 holdout.

Gallery-only: no method looks at other queries. Scored with the organizer
evaluate.ranking_metrics (top-10 first, junk stripped afterwards).
Camera labels: 'oracle' = true train camera_id (pseudo for test-gallery rows),
'pseudo' = Ilya's leave-one-vehicle-out background camera vote for every row.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
EVALUATOR = Path(r"C:\Users\user\Downloads\evaluate.py")
PSEUDO = Path(__file__).with_name("pseudo_camera.csv")
REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}
PUSH = 10.0  # larger than any cosine gap: moves an item below every unpenalised one


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def rows(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def nn_excluding_self(g: np.ndarray, k: int, forbid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    s = g @ g.T
    np.fill_diagonal(s, -np.inf)
    if forbid is not None:
        s[forbid] = -np.inf
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    return idx, np.take_along_axis(s, idx, axis=1)


def dba(g: np.ndarray, k: int = 1, forbid: np.ndarray | None = None) -> np.ndarray:
    idx, _ = nn_excluding_self(g, k, forbid)
    return norm(g + g[idx].sum(axis=1))


def hub(g: np.ndarray, k: int = 5) -> np.ndarray:
    _, v = nn_excluding_self(g, k)
    return v.mean(axis=1)


def diffuse(s: np.ndarray, g: np.ndarray, k: int, alpha: float, forbid: np.ndarray | None) -> np.ndarray:
    """One step of gallery-graph propagation: s + alpha * mean_{g' in kNN(g)} s(q, g')."""
    idx, w = nn_excluding_self(g, k, forbid)
    w = np.clip(w, 0, None)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    prop = np.einsum("qgk,gk->qg", s[:, idx], w)
    return s + alpha * prop


def per_camera_best(s: np.ndarray, gcam: np.ndarray) -> np.ndarray:
    """Push down every item that is not the best-scoring one of its camera for this query."""
    out = s.copy()
    for c in np.unique(gcam):
        cols = np.flatnonzero(gcam == c)
        if len(cols) < 2:
            continue
        sub = s[:, cols]
        best = sub.argmax(axis=1)
        mask = np.ones_like(sub, dtype=bool)
        mask[np.arange(len(s)), best] = False
        out[:, cols] = np.where(mask, sub - PUSH / 2, sub)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ev", EVALUATOR)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

    train = {r["image_id"]: r for r in rows(DATA / "train.csv")}
    pseudo = {r["image_id"]: r["pseudo_cam"] for r in rows(PSEUDO)}
    split = [r for r in rows(DATA / "val_split.csv") if r["image_id"] not in REMOVED]
    seen, hold_g, hold_q = set(), [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        (hold_q if key in seen else hold_g).append(r)
        seen.add(key)
    clean = np.load(DATA / "a10_clean_expanded.npz"); test = np.load(DATA / "a10_test.npz")
    emb = {str(i): e for i, e in zip(clean["ids"].astype(str), clean["emb"])}
    test_ids = test["ids"].astype(str); test_emb = norm(test["emb"])
    qids = [r["image_id"] for r in hold_q]; hg = [r["image_id"] for r in hold_g]
    used = set(qids) | set(hg)
    tr_ids = [i for i in train if i not in REMOVED and i not in used]
    tg = test_ids[1110:].tolist()
    assert len(qids) == 546 and len(hg) == 361 and len(tg) == 750

    def vec(ids: list[str]) -> np.ndarray:
        return norm(np.stack([emb[i] for i in ids]))

    q = vec(qids)
    test_g = test_emb[1110:]
    qdf = pd.DataFrame(hold_q).set_index("image_id")
    qcam_true = qdf.camera_id.astype(str).to_numpy()
    qcam_ps = np.array([pseudo[i] for i in qids])

    def gal_meta(ids: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        recs, tcam, pcam = [], [], []
        for i in ids:
            if i in train:
                recs.append({"image_id": i, "vehicle_id": train[i]["vehicle_id"], "camera_id": train[i]["camera_id"]})
                tcam.append(f"cam{train[i]['camera_id']}")
            else:  # test gallery: unknown vehicle, never a positive
                recs.append({"image_id": i, "vehicle_id": "x_" + i, "camera_id": "x_" + i})
                tcam.append(pseudo[i])
            pcam.append(pseudo[i])
        return pd.DataFrame(recs).set_index("image_id"), np.array(tcam), np.array(pcam)

    cases = {
        "holdout_361": (vec(hg), hg),
        "holdout+test_1111": (np.vstack([vec(hg), test_g]), hg + tg),
        "holdout+train_9005": (np.vstack([vec(hg), vec(tr_ids)]), hg + tr_ids),
    }
    report = {}
    for name, (g, gids) in cases.items():
        gdf, gcam_true, gcam_ps = gal_meta(gids)
        variants: dict[str, np.ndarray] = {}
        raw = q @ g.T
        variants["raw"] = raw
        variants["hub_k5"] = raw - hub(g)[None, :]
        g_dba = dba(g, 1)
        variants["dba_k1"] = q @ g_dba.T
        for tag, qc, gc in (("oracle", np.array([f"cam{c}" for c in qcam_true]), gcam_true), ("pseudo", qcam_ps, gcam_ps)):
            same = qc[:, None] == gc[None, :]
            gsame = gc[:, None] == gc[None, :]
            np.fill_diagonal(gsame, False)
            variants[f"cam_push[{tag}]"] = raw - PUSH * same
            variants[f"cam_pen0.1[{tag}]"] = raw - 0.1 * same
            variants[f"dba_k1+cam_push[{tag}]"] = q @ g_dba.T - PUSH * same
            variants[f"hub+cam_push[{tag}]"] = variants["hub_k5"] - PUSH * same
            # gallery frames of one camera are different vehicles -> never average across them
            g_dba_x = dba(g, 1, forbid=gsame)
            variants[f"dba_k1_xcam+cam_push[{tag}]"] = q @ g_dba_x.T - PUSH * same
            variants[f"cam_push+percam_best[{tag}]"] = per_camera_best(raw - PUSH * same, gc)
            for a in (0.3, 0.6):
                variants[f"diff_k3_a{a}_xcam+cam_push[{tag}]"] = diffuse(raw, g, 3, a, gsame) - PUSH * same
            variants[f"dba_xcam+diff0.3+cam_push+percam[{tag}]"] = per_camera_best(
                diffuse(q @ g_dba_x.T, g_dba_x, 3, 0.3, gsame) - PUSH * same, gc)
        res = {}
        for k, s in variants.items():
            order = np.argsort(-s, axis=1, kind="stable")[:, :10]
            ranked = {qid: [gids[j] for j in order[i]] for i, qid in enumerate(qids)}
            m = ev.ranking_metrics(qdf, gdf, ranked, top_k=10)
            res[k] = {kk: round(v, 4) for kk, v in m.items() if kk.startswith(("mAP", "Rank"))}
        report[name] = res
        base = res["raw"]["mAP@10"]
        print(f"\n== {name} (gallery {len(gids)})")
        for k, v in res.items():
            print(f"  {k:45s} mAP@10 {v['mAP@10']:.4f} ({v['mAP@10'] - base:+.4f})  R1 {v['Rank-1']:.4f}  R5 {v['Rank-5']:.4f}")
    # pseudo-camera accuracy on holdout rows (true cam known)
    acc_q = float(np.mean(qcam_ps == np.array([f"cam{c}" for c in qcam_true])))
    print(f"\npseudo-cam accuracy on holdout queries: {acc_q:.3f}")
    Path(__file__).with_name("oracle_ceiling.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
