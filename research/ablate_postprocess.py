#!/usr/bin/env python3
"""Сравнить варианты ранжирования на одном и том же ground truth.

Нужен для проверки гипотезы, что локальный прирост даёт хрупкая
трандуктивная постобработка, а не сам эмбеддинг:

  raw cosine -> query expansion -> CSLS -> k-reciprocal -> полный pipeline.

Ground truth должен содержать ``image_id, vehicle_id, camera_id, split``,
где split равен ``query`` или ``gallery``. Порядок строк embeddings задаётся
``embeddings_index.csv`` (split + image_id), поэтому скрипт не полагается на
случайный порядок файлов.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import reid_postprocess as pp  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_inputs(embeddings_path: Path, index_path: Path, gt_path: Path):
    emb = np.load(embeddings_path).astype("float32", copy=False)
    index = read_rows(index_path)
    gt = {row["image_id"]: row for row in read_rows(gt_path)}
    if emb.ndim != 2 or len(index) != len(emb):
        raise ValueError(
            f"embeddings/index mismatch: embeddings={emb.shape}, index={len(index)}"
        )

    q_rows, g_rows, q_pos, g_pos = [], [], [], []
    for pos, row in enumerate(index):
        image_id = row["image_id"]
        if image_id not in gt:
            raise ValueError(f"{image_id} отсутствует в ground truth")
        if row["split"] == "query":
            q_rows.append(gt[image_id])
            q_pos.append(pos)
        elif row["split"] == "gallery":
            g_rows.append(gt[image_id])
            g_pos.append(pos)
        else:
            raise ValueError(f"неизвестный split в index: {row['split']}")

    if not q_rows or not g_rows:
        raise ValueError("query и gallery не должны быть пустыми")
    return emb[q_pos], emb[g_pos], q_rows, g_rows


def ranking_metrics(scores: np.ndarray, query: list[dict], gallery: list[dict], top_k: int) -> dict:
    g_vid = np.array([row["vehicle_id"] for row in gallery])
    g_cam = np.array([row["camera_id"] for row in gallery])
    aps, r1, r5, scored = [], [], [], 0
    open_set = 0

    for qi, row in enumerate(query):
        junk = (g_vid == row["vehicle_id"]) & (g_cam == row["camera_id"])
        valid = ~junk
        n_pos = int(((g_vid == row["vehicle_id"]) & valid).sum())
        if n_pos == 0:
            open_set += 1
            continue
        scored += 1
        order = np.argsort(-scores[qi], kind="stable")
        order = order[valid[order]][:top_k]
        rel = g_vid[order] == row["vehicle_id"]
        if rel.any():
            hits = np.flatnonzero(rel)
            precision = (np.arange(len(hits)) + 1) / (hits + 1)
            # Official evaluator normalizes AP@K by the number of valid
            # positives, capped at K, not by the number of hits found.
            aps.append(float(precision.sum() / min(n_pos, top_k)))
        else:
            aps.append(0.0)
        r1.append(bool(rel[:1].any()))
        r5.append(bool(rel[:5].any()))

    return {
        f"mAP@{top_k}": float(np.mean(aps)) if aps else 0.0,
        "Rank-1": float(np.mean(r1)) if r1 else 0.0,
        "Rank-5": float(np.mean(r5)) if r5 else 0.0,
        "n_scored": scored,
        "n_openset_excluded": open_set,
    }


def score_variants(q: np.ndarray, g: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    q = pp.l2(q)
    g = pp.l2(g)
    expanded = pp.qq_expand(q, params["qq_k"], params["qq_alpha"])
    raw = q @ g.T
    qq = expanded @ g.T
    csls = pp.csls(qq, params["csls_k"])
    rerank_distance = pp.k_reciprocal_rerank(
        expanded,
        g,
        params["rerank_k1"],
        params["rerank_k2"],
        params["rerank_lambda"],
    )
    rerank = -rerank_distance
    full = pp.csls(rerank, params["csls_k"])
    return {"raw": raw, "qq": qq, "csls": csls, "rerank": rerank, "full": full}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, help="postprocess.json с params")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    params = {
        "qq_k": 3,
        "qq_alpha": 3.0,
        "csls_k": 5,
        "rerank_k1": 10,
        "rerank_k2": 6,
        "rerank_lambda": 0.7,
    }
    if args.params_json:
        meta = json.loads(args.params_json.read_text(encoding="utf-8"))
        params.update(meta.get("params", {}))

    q, g, query, gallery = load_inputs(args.embeddings, args.index, args.ground_truth)
    scores = score_variants(q, g, params)
    report = {
        "params": params,
        "inputs": {
            "query": len(query),
            "gallery": len(gallery),
            "embedding_dim": int(q.shape[1]),
        },
        "variants": {
            name: ranking_metrics(matrix, query, gallery, args.top_k)
            for name, matrix in scores.items()
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
