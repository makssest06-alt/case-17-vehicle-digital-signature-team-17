#!/usr/bin/env python3
"""Compare baseline/candidate embeddings with the official evaluator (raw cosine)."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def evaluate(name: str, q: np.ndarray, g: np.ndarray, query_rows: list[dict[str, str]],
             gallery_rows: list[dict[str, str]], output: Path, evaluator: Path) -> dict:
    similarities = q @ g.T
    ordering = np.argsort(-similarities, axis=1, kind="stable")[:, :10]
    with (output / f"{name}_submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for row, order in zip(query_rows, ordering):
            writer.writerow([row["image_id"]] + [gallery_rows[j]["image_id"] for j in order])
    gt_rows = (
        [[r["image_id"], r["vehicle_id"], r["camera_id"], "query"] for r in query_rows]
        + [[r["image_id"], r["vehicle_id"], r["camera_id"], "gallery"] for r in gallery_rows]
    )
    gt = output / f"{name}_gt.csv"
    write_csv(gt, ["image_id", "vehicle_id", "camera_id", "split"], gt_rows)
    report_path = output / f"{name}_official.json"
    result = subprocess.run(
        [sys.executable, str(evaluator), "--gt", str(gt), "--submission",
         str(output / f"{name}_submission.csv"), "--json", str(report_path)],
        check=True, text=True, capture_output=True,
    )
    print(result.stdout)
    return json.loads(report_path.read_text(encoding="utf-8"))["ranking"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-q", type=Path, required=True)
    parser.add_argument("--baseline-g", type=Path, required=True)
    parser.add_argument("--candidate-q", type=Path, required=True)
    parser.add_argument("--candidate-g", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.val_split.open(encoding="utf-8")))
    query = [row for row in rows if row["role"] == "query"]
    gallery = [row for row in rows if row["role"] == "gallery"]
    reports = {}
    for name, q_path, g_path in (
        ("baseline", args.baseline_q, args.baseline_g),
        ("candidate", args.candidate_q, args.candidate_g),
    ):
        q, g = np.load(q_path), np.load(g_path)
        if q.shape != (len(query), 512) or g.shape != (len(gallery), 512):
            raise ValueError(f"{name}: unexpected embedding shapes {q.shape}, {g.shape}")
        q_norm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        g_norm = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        reports[name] = evaluate(name, q_norm, g_norm, query, gallery, args.output, args.evaluator)
    report = {
        "protocol": "official evaluator; raw cosine; junk same vehicle AND same camera; no rerank",
        "baseline": reports["baseline"],
        "candidate": reports["candidate"],
        "delta": {key: reports["candidate"][key] - reports["baseline"][key]
                  for key in ("mAP@10", "Rank-1", "Rank-5")},
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
