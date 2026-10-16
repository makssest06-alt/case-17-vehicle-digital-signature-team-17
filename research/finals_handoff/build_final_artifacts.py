#!/usr/bin/env python3
"""Experimental single-student artifact builder; limitations in adjacent README."""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

THRESHOLD_DEFAULT = 0.7625
GAQE_B = 0.95
CSLS_K = 5
BBOX_PAD = 0.10


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "x", "y", "w", "h"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns {sorted(required)}")
        rows = list(reader)
    ids = [row["image_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: nonempty rows with unique image_id required")
    return rows


def crop(images_dir: Path, row: dict[str, str]) -> Image.Image:
    image_id = row["image_id"]
    if not image_id or Path(image_id).name != image_id or image_id in {".", ".."}:
        raise ValueError("image_id must be a filename stem")
    path = images_dir / f"{image_id}.jpg"
    if not path.is_file():
        path = images_dir / f"{image_id}.png"
    coords = np.array([float(row[key]) for key in ("x", "y", "w", "h")])
    if not np.isfinite(coords).all() or (coords[2:] <= 0).any():
        raise ValueError(f"Invalid bbox for {image_id}")
    x, y, width, height = coords
    # Same bbox rounding as training, including fractional coordinates.
    with Image.open(path) as source:
        image = source.convert("RGB")
    bounds = (
        max(0, round(x - width * BBOX_PAD)),
        max(0, round(y - height * BBOX_PAD)),
        min(image.width, round(x + width * (1 + BBOX_PAD))),
        min(image.height, round(y + height * (1 + BBOX_PAD))),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError(f"BBox outside image for {image_id}")
    return image.crop(bounds)


def embed_dir(
    model: Any, rows: list[dict[str, str]], images_dir: Path, batch_size: int
) -> np.ndarray:
    batches = []
    for start in range(0, len(rows), batch_size):
        images = [crop(images_dir, row) for row in rows[start:start + batch_size]]
        batches.append(model.embed_batch(images))
    result = np.concatenate(batches).astype(np.float64)
    if not np.isfinite(result).all() or (np.linalg.norm(result, axis=1) == 0).any():
        raise ValueError("Model returned invalid embeddings")
    return l2(result)


def compliant_scores(q: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """No cross-query statistics; gallery-only blend is an experimental option."""
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise ValueError("Expected matching 2D query/gallery embeddings")
    if len(g) < 2:
        raise ValueError("Gallery-side CSLS requires at least two gallery vectors")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("Embeddings must be finite")
    q, g = l2(q), l2(g)
    raw = q @ g.T
    nearest = np.argmax(raw, axis=1)
    expanded = l2((1 - GAQE_B) * q + GAQE_B * g[nearest])
    scores = expanded @ g.T
    gallery_scores = g @ g.T
    np.fill_diagonal(gallery_scores, -np.inf)
    gallery_k = min(CSLS_K, len(g) - 1)
    query_k = min(CSLS_K, len(g))
    rg = np.sort(gallery_scores, axis=1)[:, -gallery_k:].mean(axis=1)
    rq = np.sort(scores, axis=1)[:, -query_k:].mean(axis=1, keepdims=True)
    return raw, 2 * scores - rq - rg[None, :]


def write_artifacts(
    out: Path, qrows: list[dict[str, str]], grows: list[dict[str, str]],
    query: np.ndarray, gallery: np.ndarray, threshold: float,
) -> dict[str, int]:
    if len(grows) < 10:
        raise ValueError("Submission requires at least 10 unique gallery objects")
    if len(query) != len(qrows) or len(gallery) != len(grows):
        raise ValueError("Embedding counts must match CSV rows")
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Raw cosine threshold must be finite and in [-1, 1]")
    raw, scores = compliant_scores(query, gallery)
    # CSLS of unit vectors is in [-4,4]; fixed affine map preserves independence.
    confidence = (scores + 4.0) / 8.0
    ranking = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", np.vstack([query, gallery]).astype("float32"))
    with (out / "embeddings_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "split", "image_id"])
        for index, row in enumerate(qrows + grows):
            writer.writerow([index, "query" if index < len(qrows) else "gallery", row["image_id"]])
    accepted = 0
    with (out / "submission.csv").open("w", newline="", encoding="utf-8") as sf, (
        out / "candidates.csv"
    ).open("w", newline="", encoding="utf-8") as cf:
        submission, candidates = csv.writer(sf), csv.writer(cf)
        submission.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, 11)])
        candidates.writerow(["query_id", "gallery_id", "confidence"])
        for index, row in enumerate(qrows):
            order = ranking[index]
            submission.writerow([row["image_id"]] + [grows[j]["image_id"] for j in order])
            if raw[index].max() < threshold:
                continue
            accepted += 1
            for j in order:
                candidates.writerow([row["image_id"], grows[j]["image_id"], float(confidence[index, j])])
    return {"accepted": accepted, "refused": len(qrows) - accepted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("images", "query", "gallery", "weights", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    qrows, grows = read_rows(args.query), read_rows(args.gallery)
    if len(grows) < 10:
        parser.error("at least 10 gallery objects required")
    # Set before config import; explicit experiment profile, one student/no TTA.
    os.environ["REID_TF_MODE"] = "squash"
    os.environ["EMBED_TTA_FLIP"] = "0"
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "api"))
    from app.models.reid_checkpoint import ReidCheckpointModel

    model = ReidCheckpointModel("vit_base_patch16_siglip_224.webli", str(args.weights), 224)
    model.load()
    query = embed_dir(model, qrows, args.images, args.batch_size)
    gallery = embed_dir(model, grows, args.images, args.batch_size)
    counts = write_artifacts(args.out, qrows, grows, query, gallery, args.threshold)
    with args.weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    meta = {
        "status": "experimental_handoff_not_final_submission",
        "pipeline": "gallery-top1-blend -> gallery-side CSLS",
        "params": {"blend": GAQE_B, "csls_k": CSLS_K},
        "threshold": args.threshold,
        "confidence": "(CSLS+4)/8 fixed; no statistics across query",
        "weights_sha256": digest,
        **counts,
    }
    (args.out / "postprocess.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
