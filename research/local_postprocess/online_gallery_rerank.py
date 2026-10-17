#!/usr/bin/env python3
"""Gallery-only reciprocal reranking experiment for the Case 17 online API.

Unlike legacy k-reciprocal code, this script never links one query to another.
It precomputes each gallery vector's K nearest gallery neighbours and then
scores a request from its own vector plus that fixed gallery graph.  Therefore
adding another request cannot change a previous answer.

This is RESEARCH ONLY.  Parameters are selected on one vehicle-ID fold and
reported on a held-out vehicle-ID fold.  It must pass service-path latency,
RAM and refusal calibration before production integration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DUPLICATE_HOLDOUT_IDS = {
    "1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
    "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
    "ba950758834d4f26b7df5dd4832b6797",
}


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean_rows(train: list[dict[str, str]], split: list[dict[str, str]]):
    hold = {r["vehicle_id"] for r in split}
    by_camera: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in train:
        if row["vehicle_id"] in hold and row["image_id"] not in DUPLICATE_HOLDOUT_IDS:
            by_camera.setdefault((row["vehicle_id"], row["camera_id"]), []).append(row)
    queries, gallery = [], []
    for _, rows in sorted(by_camera.items(), key=lambda x: (int(x[0][0]), int(x[0][1]))):
        rows.sort(key=lambda r: r["image_id"])
        gallery.append(rows[0])
        queries.extend(rows[1:])
    distractors = [r for r in train if r["vehicle_id"] not in hold]
    return queries, gallery + distractors


def topk_rows(scores: np.ndarray, k: int, self_index: bool = False) -> np.ndarray:
    """Top k indices per row.  Scores are never materialised for all chunks."""
    k = min(k + int(self_index), scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    val = np.take_along_axis(scores, idx, axis=1)
    idx = np.take_along_axis(idx, np.argsort(-val, axis=1), axis=1)
    if self_index:
        rows = np.arange(scores.shape[0])[:, None]
        # self appears first for L2-normalised vectors. Remove it robustly.
        keep = idx != rows
        return np.array([r[m][: k - 1] for r, m in zip(idx, keep)], dtype=np.int32)
    return idx.astype(np.int32)


def gallery_graph(gallery: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    """KNN graph using gallery only; O(G²) time, O(chunk*G) working memory."""
    out = []
    gt = gallery.T
    for start in range(0, len(gallery), chunk):
        score = gallery[start:start + chunk] @ gt
        out.append(topk_rows(score, k, self_index=True))
    return np.vstack(out)


def reciprocal_scores(q: np.ndarray, gallery: np.ndarray, graph: np.ndarray, k: int, mix: float) -> np.ndarray:
    """Online score: raw cosine mixed with weighted reciprocal-neighbour Jaccard.

    Only q, gallery and a graph built from gallery are read. No batch/query
    state is used.  This deliberately returns a full score matrix for fair
    mAP evaluation; an API implementation would score only a retrieve set.
    """
    raw = q @ gallery.T
    q_neighbours = topk_rows(raw, k)
    result = np.empty_like(raw, dtype=np.float32)
    denom_base = float(2 * k)
    for i, neighbours in enumerate(q_neighbours):
        overlap = np.isin(graph, neighbours, assume_unique=False).sum(axis=1)
        # Mutual membership gets a small bonus but never uses another query.
        mutual = np.isin(neighbours, np.flatnonzero(raw[i] >= -np.inf), assume_unique=False)
        del mutual  # documented invariance; do not accidentally add query state.
        jaccard = overlap / np.maximum(denom_base - overlap, 1.0)
        result[i] = mix * raw[i] + (1.0 - mix) * jaccard
    return result


def map10(scores: np.ndarray, query: list[dict[str, str]], gallery: list[dict[str, str]], take: np.ndarray) -> dict:
    vids = np.array([r["vehicle_id"] for r in gallery])
    cams = np.array([r["camera_id"] for r in gallery])
    aps, r1, r5 = [], [], []
    for i in take:
        row = query[int(i)]
        same = vids == row["vehicle_id"]
        junk = same & (cams == row["camera_id"])
        positive = same & ~junk
        n_pos = int(positive.sum())
        if not n_pos:
            continue
        order = np.argsort(-scores[int(i)], kind="stable")
        order = order[~junk[order]][:10]
        rel = same[order]
        if rel.any():
            c = np.cumsum(rel)
            aps.append(float((c * rel / (np.arange(len(rel)) + 1)).sum() / min(n_pos, 10)))
        else:
            aps.append(0.0)
        r1.append(float(rel[:1].any()))
        r5.append(float(rel[:5].any()))
    return {"mAP@10": float(np.mean(aps)), "Rank-1": float(np.mean(r1)), "Rank-5": float(np.mean(r5)), "n": len(aps)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True, help="npz ids,emb from A10 extraction")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--mixes", default="0.5,0.7,0.85,0.95")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    train = read_rows(args.data_dir / "train.csv")
    split = read_rows(args.data_dir / "val_split.csv")
    query, gallery_rows = clean_rows(train, split)
    z = np.load(args.embeddings)
    lookup = {str(v): i for i, v in enumerate(z["ids"].tolist())}
    emb = l2(z["emb"].astype("float32"))
    q = emb[[lookup[r["image_id"]] for r in query]]
    g = emb[[lookup[r["image_id"]] for r in gallery_rows]]

    vids = np.array([r["vehicle_id"] for r in query])
    vehicle_ids = np.array(sorted(set(vids), key=int))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(vehicle_ids)
    tune_ids = set(vehicle_ids[: len(vehicle_ids) // 2])
    tune = np.flatnonzero(np.isin(vids, list(tune_ids)))
    test = np.flatnonzero(~np.isin(vids, list(tune_ids)))
    raw = q @ g.T
    rows = [{"variant": "raw", "tune": map10(raw, query, gallery_rows, tune), "test": map10(raw, query, gallery_rows, test)}]

    for k in [int(x) for x in args.ks.split(",")]:
        graph = gallery_graph(g, k, args.chunk)
        for mix in [float(x) for x in args.mixes.split(",")]:
            score = reciprocal_scores(q, g, graph, k, mix)
            rows.append({"variant": f"gallery_reciprocal k={k} mix={mix:g}",
                         "tune": map10(score, query, gallery_rows, tune),
                         "test": map10(score, query, gallery_rows, test)})

    raw_tune = rows[0]["tune"]["mAP@10"]
    best = max(rows[1:], key=lambda r: r["tune"]["mAP@10"], default=None)
    payload = {
        "protocol": "gallery-only, query-independent reciprocal Jaccard; train-only distractors; clean holdout; split by vehicle_id",
        "warnings": ["research only", "selection uses tune half; report test half", "not a hidden-test estimate", "not production latency/memory evidence"],
        "inputs": {"model_sha256": "93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce",
                   "train_sha256": sha256(args.data_dir / "train.csv"), "split_sha256": sha256(args.data_dir / "val_split.csv"),
                   "preprocessing": "squash 224, bbox_pad=0.10, TTA=false", "query": len(query), "gallery": len(gallery_rows)},
        "raw_tune": raw_tune, "selected_by_tune": best, "rows": rows,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['variant']:<35} tune={row['tune']['mAP@10']:.4f} test={row['test']['mAP@10']:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
