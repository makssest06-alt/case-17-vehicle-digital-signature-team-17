#!/usr/bin/env python3
"""Build research-only A10 submission artifacts from a cached embedding file.

This script deliberately uses query-to-gallery raw cosine ranking only.  It does
not import the online service and does not use query-to-query or gallery graph
state, so the resulting artifacts are reproducible without test labels.

Example (PowerShell):
  python research/local_postprocess/build_a10_artifacts.py `
    --data-dir C:/.../case17_data `
    --embeddings C:/.../case17_data/a10_test.npz `
    --out research/local_postprocess/a10_artifacts
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np


TOP_K = 10
KNOWN_A10_CHECKPOINT_SHA256 = (
    "93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [row["image_id"] for row in csv.DictReader(stream)]


def repo_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="A10 .npz containing ids and emb")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True,
                        help="raw top-1 cosine refusal threshold calibrated outside test")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="A10 checkpoint; its SHA-256 is verified")
    parser.add_argument("--cache-provenance", type=Path, required=True,
                        help="JSON manifest binding embedding cache to checkpoint")
    args = parser.parse_args()

    q_ids = ids(args.data_dir / "test_query.csv")
    g_ids = ids(args.data_dir / "test_gallery.csv")
    if len(set(q_ids)) != len(q_ids) or len(set(g_ids)) != len(g_ids):
        raise SystemExit("duplicate image_id in canonical test CSV")
    if set(q_ids) & set(g_ids):
        raise SystemExit("query/gallery image_id overlap")

    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != KNOWN_A10_CHECKPOINT_SHA256:
        raise SystemExit(
            "checkpoint SHA-256 does not match clean A10: "
            f"expected {KNOWN_A10_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )

    if not args.cache_provenance.is_file():
        raise SystemExit(f"cache provenance not found: {args.cache_provenance}")
    try:
        provenance = json.loads(args.cache_provenance.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid cache provenance: {exc}") from exc
    if provenance.get("checkpoint_sha256") != checkpoint_sha:
        raise SystemExit("cache provenance checkpoint SHA does not match verified checkpoint")
    if provenance.get("preprocess") != "squash 224, bbox_pad=0.10, TTA=false":
        raise SystemExit("cache provenance preprocessing is not the required A10 preprocessing")

    cached = np.load(args.embeddings, allow_pickle=False)
    if set(cached.files) != {"ids", "emb"}:
        raise SystemExit(f"expected npz keys ids, emb; got {cached.files}")
    cached_ids = [str(value) for value in cached["ids"].tolist()]
    emb = cached["emb"]
    want_ids = q_ids + g_ids
    if len(cached_ids) != len(set(cached_ids)) or set(cached_ids) != set(want_ids):
        raise SystemExit("A10 cache IDs do not exactly match test query/gallery")
    cache_sha = provenance.get("cache_sha256")
    if not isinstance(cache_sha, str) or len(cache_sha) != 64:
        raise SystemExit("cache provenance must contain a 64-character cache_sha256")
    if cache_sha.lower() != sha256(args.embeddings):
        raise SystemExit("cache provenance SHA does not match embedding cache")
    positions = {image_id: i for i, image_id in enumerate(cached_ids)}
    emb = np.asarray(emb[[positions[value] for value in want_ids]], dtype=np.float32)
    if emb.ndim != 2 or not np.isfinite(emb).all():
        raise SystemExit("cache embeddings must be finite 2-D vectors")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise SystemExit("cache contains a zero vector")
    emb = emb / norms
    q, g = emb[:len(q_ids)], emb[len(q_ids):]
    scores = q @ g.T
    order = np.argsort(-scores, axis=1, kind="stable")
    threshold = float(args.threshold)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embeddings.npy", emb.astype(np.float32))
    with (args.out / "embeddings_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "split", "image_id"])
        for row, image_id in enumerate(want_ids):
            writer.writerow([row, "query" if row < len(q_ids) else "gallery", image_id])

    with (args.out / "submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, TOP_K + 1)])
        for qi, query_id in enumerate(q_ids):
            writer.writerow([query_id] + [g_ids[j] for j in order[qi, :TOP_K]])

    top1 = scores[np.arange(len(q_ids)), order[:, 0]]
    with (args.out / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        accepted = [(q_ids[qi], g_ids[int(order[qi, 0])], float(top1[qi]))
                    for qi in range(len(q_ids)) if top1[qi] >= threshold]
        for query_id, gallery_id, confidence in sorted(accepted, key=lambda row: -row[2]):
            writer.writerow([query_id, gallery_id, round(confidence, 8)])

    # The cache was produced by the A10 checkpoint recorded in the benchmark
    # provenance.  A separately supplied local .pt is recorded as an additional
    # file hash, never substituted for the cache's source checkpoint hash.
    script_path = Path(__file__).resolve()
    metadata = {
        "scope": "research-only-local-postprocess",
        "checkpoint": str(args.checkpoint) if args.checkpoint else "A10 cache provenance",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_file_sha256": checkpoint_sha,
        "embedding_cache": str(args.embeddings),
        "embedding_cache_sha256": sha256(args.embeddings),
        "builder_script": str(script_path),
        "builder_script_sha256": sha256(script_path),
        "preprocess": "squash 224, bbox_pad=0.10, TTA=false, L2-normalize",
        "ranking": "raw query-gallery cosine, stable descending sort; no transductive postprocess",
        "threshold": {
            "space": "raw cosine top-1",
            "value": threshold,
            "rule": "explicit",
            "calibration_source": "externally supplied; verify against clean holdout CV report",
        },
        "commit": repo_commit(),
        "labels": "test labels unavailable; no test-set metric claimed",
        "counts": {"query": len(q_ids), "gallery": len(g_ids), "accepted": len(accepted), "refused": len(q_ids) - len(accepted)},
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "query": len(q_ids), "gallery": len(g_ids),
                      "threshold": threshold, "accepted": len(accepted)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
