#!/usr/bin/env python3
"""Калибровка порога отказа на воспроизводимом holdout.

Сервис принимает решение по raw top-1 cosine, поэтому порог для него нельзя
переносить из ``artifacts/postprocess.json``: там записан score другой
трандуктивной матрицы. Скрипт строит controlled open-set: для части
vehicle_id убирает все их gallery-кадры и подбирает порог по известным/чужим
запросам.

Пример:
  python research/calibrate_refusal.py \
    --embeddings artifacts/embeddings.npy \
    --index artifacts/embeddings_index.csv \
    --ground-truth /path/to/ground_truth.csv \
    --target-recall 0.70 \
    --json research/results/refusal_calibration.json

Это диагностическая калибровка, не доказательство поведения на скрытом тесте.
Перед публикацией значения нужно подтвердить на чистой выборке организаторов.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_inputs(embeddings_path: Path, index_path: Path, gt_path: Path):
    emb = np.load(embeddings_path).astype("float32", copy=False)
    index = read_rows(index_path)
    gt = {row["image_id"]: row for row in read_rows(gt_path)}
    if emb.ndim != 2 or len(index) != len(emb):
        raise ValueError(f"embeddings/index mismatch: {emb.shape} vs {len(index)}")

    q_pos, g_pos, query, gallery = [], [], [], []
    for pos, row in enumerate(index):
        if row["image_id"] not in gt:
            raise ValueError(f"{row['image_id']} отсутствует в ground truth")
        if row["split"] == "query":
            q_pos.append(pos)
            query.append(gt[row["image_id"]])
        elif row["split"] == "gallery":
            g_pos.append(pos)
            gallery.append(gt[row["image_id"]])
        else:
            raise ValueError(f"неизвестный split: {row['split']}")
    return emb[q_pos], emb[g_pos], query, gallery


def top1_scores(
    q_emb: np.ndarray,
    g_emb: np.ndarray,
    query: list[dict[str, str]],
    gallery: list[dict[str, str]],
    unknown_ids: set[str],
    cross_camera: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sim = q_emb @ g_emb.T
    g_vid = np.array([row["vehicle_id"] for row in gallery])
    g_cam = np.array([row["camera_id"] for row in gallery])
    known, unknown = [], []
    for qi, row in enumerate(query):
        keep = np.ones(len(gallery), dtype=bool)
        if cross_camera:
            keep &= g_cam != row["camera_id"]
        if row["vehicle_id"] in unknown_ids:
            keep &= g_vid != row["vehicle_id"]
            unknown.append(float(sim[qi, keep].max()) if keep.any() else float("-inf"))
        else:
            if keep.any() and np.any(g_vid[keep] == row["vehicle_id"]):
                known.append(float(sim[qi, keep].max()))
    return np.asarray(known, dtype="float64"), np.asarray(unknown, dtype="float64")


def metrics(known: np.ndarray, unknown: np.ndarray, threshold: float) -> dict:
    tp = int((known >= threshold).sum())
    fn = int((known < threshold).sum())
    fp = int((unknown >= threshold).sum())
    tn = int((unknown < threshold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "TNR": tn / (tn + fp) if tn + fp else 0.0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def choose_threshold(known: np.ndarray, unknown: np.ndarray, target_recall: float) -> dict:
    values = np.unique(np.concatenate([known, unknown]))
    if not len(values):
        raise ValueError("нет score для калибровки")
    candidates = np.concatenate(([np.inf], values[::-1], [-np.inf]))
    points = [metrics(known, unknown, float(t)) for t in candidates]
    feasible = [point for point in points if point["recall"] + 1e-12 >= target_recall]
    return max(feasible, key=lambda point: (point["TNR"], point["precision"], -point["threshold"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.70)
    parser.add_argument("--unknown-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross-camera", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not 0 < args.target_recall <= 1:
        parser.error("--target-recall должен быть в (0, 1]")
    if not 0 < args.unknown_fraction < 1:
        parser.error("--unknown-fraction должен быть в (0, 1)")

    q_emb, g_emb, query, gallery = load_inputs(args.embeddings, args.index, args.ground_truth)
    query_vids = np.array(sorted({row["vehicle_id"] for row in query}))
    rng = np.random.default_rng(args.seed)
    unknown_count = max(1, int(len(query_vids) * args.unknown_fraction))
    unknown_ids = set(rng.choice(query_vids, size=unknown_count, replace=False).tolist())
    known, unknown = top1_scores(q_emb, g_emb, query, gallery, unknown_ids, args.cross_camera)
    selected = choose_threshold(known, unknown, args.target_recall)
    report = {
        "diagnostic": True,
        "protocol": "cross-camera" if args.cross_camera else "deployment-no-camera-filter",
        "seed": args.seed,
        "unknown_fraction": args.unknown_fraction,
        "target_recall": args.target_recall,
        "queries": {"known": len(known), "unknown": len(unknown)},
        "selected": selected,
        "score_ranges": {
            "known": [float(known.min()), float(known.max())] if len(known) else [],
            "unknown": [float(unknown.min()), float(unknown.max())] if len(unknown) else [],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"отчёт сохранён: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
