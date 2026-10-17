#!/usr/bin/env python3
"""Recalculate the A10 local holdout with the organizer evaluator semantics.

The official evaluator receives the first ten generated candidates and only
then strips junk.  This script deliberately does not strip before selecting
the ten.  Hubness uses a chunked gallery-to-gallery calculation and excludes
the gallery item itself by global gallery index.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # .../BM
DATA = ROOT / "case17_data"
REPO = ROOT / "test-clone" / "repo"
EVALUATOR = Path(r"C:\Users\user\Downloads\evaluate.py")
OUT = ROOT / "data" / "a10_official_recalculation.json"
REMOVED = {
    "1b4ab30599ea41178b775b81f5f6265c",
    "25fe4ed7ba5b433b9ae84795ea909729",
    "4b90daade9334addb76bf838887e1430",
    "64a9bec33f0e4b21b1f9a0c8f8504741",
    "ba950758834d4f26b7df5dd4832b6797",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def hubness(g: np.ndarray, batch: int = 256, k: int = 5) -> np.ndarray:
    """Mean top-k gallery similarity for each gallery item, excluding self."""
    n = len(g)
    out = np.empty(n, dtype=np.float32)
    global_idx = np.arange(n)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        s = g[start:end] @ g.T
        local = np.arange(end - start)
        s[local, start + local] = -np.inf
        top = np.partition(s, -k, axis=1)[:, -k:]
        out[start:end] = top.mean(axis=1)
    return out


def scores(q: np.ndarray, g: np.ndarray, beta: float, batch: int) -> np.ndarray:
    h = hubness(g, batch=batch)
    out = np.empty((len(q), len(g)), dtype=np.float32)
    for start in range(0, len(q), batch):
        end = min(start + batch, len(q))
        out[start:end] = q[start:end] @ g.T - beta * h[None, :]
    return out


def official_metrics(evaluator, query, gallery, s: np.ndarray) -> dict:
    ranked = {}
    gids = gallery.index.to_numpy()
    for i, qid in enumerate(query.index):
        # Full ordering is generated first; evaluator ranking_metrics takes
        # the supplied list and performs junk filtering only inside its top-k.
        ranked[qid] = gids[np.argsort(-s[i], kind="stable")[:10]].tolist()
    return evaluator.ranking_metrics(query, gallery, ranked, top_k=10)


def old_strip_before_top10(query, gallery, s: np.ndarray) -> dict:
    gvid = gallery.vehicle_id.to_dict()
    gcam = gallery.camera_id.to_dict()
    aps, r1, r5 = [], [], []
    for i, (_, q) in enumerate(query.iterrows()):
        same = (gallery.vehicle_id.to_numpy() == q.vehicle_id) & (gallery.camera_id.to_numpy() == q.camera_id)
        npos = int(((gallery.vehicle_id.to_numpy() == q.vehicle_id) & ~same).sum())
        if not npos:
            continue
        order = np.argsort(-s[i], kind="stable")
        clean = [j for j in order if not same[j]][:10]
        rel = np.array([gallery.vehicle_id.iloc[j] == q.vehicle_id for j in clean], dtype=bool)
        cum = np.cumsum(rel)
        aps.append(float((cum / (np.arange(len(rel)) + 1) * rel).sum() / min(npos, 10)))
        r1.append(bool(rel[:1].any())); r5.append(bool(rel[:5].any()))
    return {"n_scored": len(aps), "mAP@10": float(np.mean(aps)),
            "Rank-1": float(np.mean(r1)), "Rank-5": float(np.mean(r5))}


def main() -> None:
    spec = importlib.util.spec_from_file_location("organizer_evaluate", EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    train = rows(DATA / "train.csv")
    # Reconstruct the clean protocol: the first image in each
    # (vehicle_id,camera_id) group is gallery; all later images are query.
    # This is intentionally not the stale role column in val_split.csv.
    split = [r for r in rows(DATA / "val_split.csv") if r["image_id"] not in REMOVED]
    seen_groups: set[tuple[str, str]] = set()
    holdout_g, holdout_q = [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        if key not in seen_groups:
            seen_groups.add(key); holdout_g.append(r)
        else:
            holdout_q.append(r)
    clean = np.load(DATA / "a10_clean_expanded.npz")
    test = np.load(DATA / "a10_test.npz")
    emb_by_id = {str(i): e for i, e in zip(clean["ids"].astype(str), clean["emb"])}
    test_ids = test["ids"].astype(str).tolist(); test_emb = test["emb"]
    qids = [r["image_id"] for r in holdout_q]
    holdout_gids = [r["image_id"] for r in holdout_g]
    train_ids = [r["image_id"] for r in train if r["image_id"] not in REMOVED and r["image_id"] not in set(qids + holdout_gids)]
    test_gids = [str(x) for x in test_ids[1110:]]
    test_qids = [str(x) for x in test_ids[:1110]]
    assert len(qids) == 546 and len(holdout_gids) == 361 and len(train_ids) == 8644
    assert test["emb"].shape[0] == 1860 and len(test_gids) == 750
    q = norm(np.stack([emb_by_id[x] for x in qids]))
    test_q = norm(test_emb[:1110]); test_g = norm(test_emb[1110:])
    train_g = norm(np.stack([emb_by_id[x] for x in train_ids]))
    hold_g = norm(np.stack([emb_by_id[x] for x in holdout_gids]))
    metadata = {r["image_id"]: r for r in train}
    eval_spec = {"evaluator": evaluator, "query": None, "gallery": None}
    def make_eval(gids, ge, synthetic=False):
        qdf = __import__("pandas").DataFrame(holdout_q).set_index("image_id")
        grows = [metadata[x].copy() for x in gids if x in metadata]
        if synthetic:
            grows += [{"image_id": x, "vehicle_id": "synthetic_" + x, "camera_id": "synthetic_" + x} for x in gids if x not in metadata]
        gdf = __import__("pandas").DataFrame(grows).set_index("image_id")
        return qdf, gdf
    cases = {
        "small_holdout": (hold_g, holdout_gids),
        "holdout_plus_test_gallery": (np.vstack([hold_g, test_g]), holdout_gids + test_gids),
        "holdout_plus_train_distractors": (np.vstack([hold_g, train_g]), holdout_gids + train_ids),
        "holdout_plus_train_plus_test_gallery": (np.vstack([hold_g, train_g, test_g]), holdout_gids + train_ids + test_gids),
    }
    result = {"protocol": {"top10_before_junk_filter": True, "hubness_k": 5, "hubness_beta": 1.0, "chunk": 256, "do_not_use_test_query_as_gallery": True}, "counts": {"holdout_query": len(qids), "holdout_gallery": len(holdout_gids), "train_distractors": len(train_ids), "test_gallery": len(test_gids)}, "cases": {}}
    for name, (ge, gids) in cases.items():
        qdf, gdf = make_eval(gids, ge, synthetic=True)
        raw = q @ ge.T
        fixed = scores(q, ge, 1.0, 256)
        rawm = official_metrics(evaluator, qdf, gdf, raw)
        fixedm = official_metrics(evaluator, qdf, gdf, fixed)
        result["cases"][name] = {"q_count": len(q), "g_count": len(ge), "raw_official": rawm, "hubness_fixed_official": fixedm, "raw_old_strip_before_top10": old_strip_before_top10(qdf, gdf, raw), "fixed_old_strip_before_top10": old_strip_before_top10(qdf, gdf, fixed)}
    s256 = scores(q, np.vstack([hold_g, train_g]), 1.0, 256)
    s73 = scores(q, np.vstack([hold_g, train_g]), 1.0, 73)
    result["batch_invariance"] = {
        "gallery": 9005,
        "max_abs_score_delta_batch_256_vs_73": float(np.max(np.abs(s256-s73))),
        "scores_bitwise_identical": bool(np.array_equal(s256, s73)),
        "top10_rankings_identical": bool(np.array_equal(
            np.argsort(-s256, axis=1, kind="stable")[:, :10],
            np.argsort(-s73, axis=1, kind="stable")[:, :10],
        )),
    }
    result["provenance"] = {"evaluator_sha256": sha256(EVALUATOR), "script_sha256": sha256(Path(__file__)), "cache_sha256": {"a10_clean_expanded.npz": sha256(DATA/"a10_clean_expanded.npz"), "a10_test.npz": sha256(DATA/"a10_test.npz")}, "model_sha256_source_session_evidence": "93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce", "model_sha256_note": "source session evidence; cache provenance not independently proven"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
