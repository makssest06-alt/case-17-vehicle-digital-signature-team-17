#!/usr/bin/env python3
"""Build an identity- and exact-image-clean train/holdout split.

The source train CSV contains the 150 holdout vehicles as well as five exact
JPEG duplicates across the resulting train/holdout boundary. This script
removes every holdout vehicle from training, then removes every train row whose
JPEG bytes occur in holdout. It writes rows only; it does not copy images or
train a model.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_rows(path: Path, data: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    source = rows(args.source_train)
    holdout = rows(args.holdout_split)
    holdout_ids = {row["vehicle_id"] for row in holdout}
    train = [row for row in source if row["vehicle_id"] not in holdout_ids]
    holdout_hashes = {
        digest(args.images_dir / f"{row['image_id']}.jpg"): row["image_id"]
        for row in holdout
    }
    duplicate_train_ids = []
    clean_train = []
    for row in train:
        image_hash = digest(args.images_dir / f"{row['image_id']}.jpg")
        if image_hash in holdout_hashes:
            duplicate_train_ids.append(row["image_id"])
        else:
            clean_train.append(row)
    duplicate_holdout_ids = {
        holdout_hashes[digest(args.images_dir / f"{row['image_id']}.jpg")]
        for row in holdout
        if digest(args.images_dir / f"{row['image_id']}.jpg")
        in {digest(args.images_dir / f"{item}.jpg") for item in duplicate_train_ids}
    }
    clean_holdout = [row for row in holdout if row["image_id"] not in duplicate_holdout_ids]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_fields = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]
    holdout_fields = ["image_id", "vehicle_id", "camera_id", "role"]
    write_rows(args.out_dir / "clean_train_rows.csv", clean_train, train_fields)
    write_rows(args.out_dir / "clean_holdout_split.csv", clean_holdout, holdout_fields)
    manifest = {
        "source_train_rows": len(source),
        "source_train_vehicle_ids": len({row["vehicle_id"] for row in source}),
        "holdout_rows_before": len(holdout),
        "holdout_vehicle_ids": len(holdout_ids),
        "train_rows_before_hash_filter": len(train),
        "clean_train_rows": len(clean_train),
        "exact_duplicate_train_rows_removed": len(duplicate_train_ids),
        "exact_duplicate_holdout_rows_removed": len(duplicate_holdout_ids),
        "clean_holdout_rows": len(clean_holdout),
        "clean_train_vehicle_overlap": len(
            {row["vehicle_id"] for row in clean_train}
            & {row["vehicle_id"] for row in clean_holdout}
        ),
        "removed_train_image_ids": duplicate_train_ids,
        "removed_holdout_image_ids": sorted(duplicate_holdout_ids),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
