#!/usr/bin/env python3
"""Аудит holdout-сплита перед экспериментами по генерализации.

Падение метрики на новом наборе нельзя лечить подбором модели, пока не
подтверждено, что локальная валидация не протекает данными. Скрипт проверяет
identity-disjoint split, пригодность кросс-камерных query, а при --images-dir
ещё и точные дубли изображений между train и holdout.

Пример:
  python research/audit_generalization.py \
    --source-csv /opt/case17/data/crops/crops_train.csv \
    --split-csv research/val_split.csv \
    --images-dir /opt/case17/data/crops/train \
    --json research/results/generalization_audit.json \
    --strict
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_SOURCE_COLUMNS = {"image_id", "vehicle_id", "camera_id"}
REQUIRED_SPLIT_COLUMNS = REQUIRED_SOURCE_COLUMNS | {"role"}
VALID_ROLES = {"gallery", "query"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def require_columns(rows: list[dict[str, str]], required: set[str], path: Path) -> None:
    columns = set(rows[0]) if rows else set()
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path}: нет обязательных колонок: {', '.join(missing)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    source: list[dict[str, str]],
    split: list[dict[str, str]],
    images_dir: Path | None,
    training_rows: list[dict[str, str]] | None,
) -> dict:
    source_by_image = {row["image_id"]: row for row in source}
    source_duplicates = len(source_by_image) != len(source)
    split_image_ids = [row["image_id"] for row in split]
    split_ids = set(split_image_ids)
    heldout_ids = {row["vehicle_id"] for row in split}
    source_ids = {row["vehicle_id"] for row in source}
    training = (
        training_rows
        if training_rows is not None
        else [row for row in source if row["vehicle_id"] not in heldout_ids]
    )
    training_ids = {row["vehicle_id"] for row in training}

    failures: list[str] = []
    warnings: list[str] = []
    roles = Counter(row["role"] for row in split)
    unknown_roles = sorted(set(roles) - VALID_ROLES)
    if unknown_roles:
        failures.append(f"неизвестные роли в split: {', '.join(unknown_roles)}")
    if roles["gallery"] == 0 or roles["query"] == 0:
        failures.append("в split должны быть и gallery, и query")
    if len(split_ids) != len(split_image_ids):
        failures.append(f"дубли image_id внутри split: {len(split_image_ids) - len(split_ids)}")
    if source_duplicates:
        failures.append(f"дубли image_id в source CSV: {len(source) - len(source_by_image)}")

    missing_from_source = sorted(split_ids - set(source_by_image))
    if missing_from_source:
        failures.append(
            f"в split есть image_id, которых нет в source CSV: {len(missing_from_source)} "
            f"(например {missing_from_source[0]})"
        )

    metadata_mismatch = []
    for row in split:
        source_row = source_by_image.get(row["image_id"])
        if source_row and any(row[key] != source_row[key] for key in REQUIRED_SOURCE_COLUMNS):
            metadata_mismatch.append(row["image_id"])
    if metadata_mismatch:
        failures.append(
            f"vehicle_id/camera_id split не совпадает с source CSV: {len(metadata_mismatch)} "
            f"(например {metadata_mismatch[0]})"
        )

    # Именно vehicle_id, а не image_id: модель не должна видеть эту машину при обучении.
    leaked_identities = training_ids & heldout_ids
    if leaked_identities:
        failures.append(f"vehicle_id пересекаются между train и holdout: {len(leaked_identities)}")

    gallery_cameras: dict[str, set[str]] = defaultdict(set)
    for row in split:
        if row["role"] == "gallery":
            gallery_cameras[row["vehicle_id"]].add(row["camera_id"])
    usable_queries = 0
    no_cross_camera_match = []
    for row in split:
        if row["role"] != "query":
            continue
        if gallery_cameras[row["vehicle_id"]] - {row["camera_id"]}:
            usable_queries += 1
        else:
            no_cross_camera_match.append(row["image_id"])
    if no_cross_camera_match:
        warnings.append(
            f"query без релевантного кандидата с другой камеры: {len(no_cross_camera_match)}; "
            "их нужно исключать из ranking-метрик и учитывать отдельно как open-set"
        )

    report = {
        "source": {
            "images": len(source),
            "vehicle_ids": len(source_ids),
            "duplicate_image_ids": len(source) - len(source_by_image),
        },
        "holdout": {
            "images": len(split),
            "vehicle_ids": len(heldout_ids),
            "gallery": roles["gallery"],
            "query": roles["query"],
            "usable_cross_camera_query": usable_queries,
            "query_without_cross_camera_match": len(no_cross_camera_match),
        },
        "train": {
            "images": len(training),
            "vehicle_ids": len(training_ids),
            "vehicle_id_overlap_with_holdout": len(leaked_identities),
            "source": "явно переданный training CSV" if training_rows is not None else
                      "выведен из source CSV исключением holdout vehicle_id",
        },
        "failures": failures,
        "warnings": warnings,
    }

    if images_dir is not None:
        if not images_dir.is_dir():
            failures.append(f"папка изображений не найдена: {images_dir}")
        else:
            train_hashes: dict[str, str] = {}
            missing_files = []
            for row in training:
                path = images_dir / f"{row['image_id']}.jpg"
                if path.is_file():
                    train_hashes[sha256(path)] = row["image_id"]
                else:
                    missing_files.append(row["image_id"])

            duplicate_images = []
            for row in split:
                path = images_dir / f"{row['image_id']}.jpg"
                if not path.is_file():
                    missing_files.append(row["image_id"])
                    continue
                digest = sha256(path)
                if digest in train_hashes:
                    duplicate_images.append(
                        {"holdout_image_id": row["image_id"], "train_image_id": train_hashes[digest]}
                    )
            if duplicate_images:
                failures.append(
                    f"точные дубли изображений между train и holdout: {len(duplicate_images)}"
                )
            if missing_files:
                warnings.append(f"нет файлов для проверки хэшей: {len(missing_files)}")
            report["image_hashes"] = {
                "checked": len(training) + len(split) - len(missing_files),
                "missing_files": len(missing_files),
                "train_holdout_exact_duplicates": duplicate_images[:20],
                "train_holdout_exact_duplicate_count": len(duplicate_images),
            }

    report["ok"] = not failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True, help="crops_train.csv с метками")
    parser.add_argument("--split-csv", type=Path, required=True, help="val_split.csv с role=gallery/query")
    parser.add_argument(
        "--training-csv",
        type=Path,
        help="фактический CSV, использованный при обучении; без него train выводится из source CSV",
    )
    parser.add_argument("--images-dir", type=Path, help="необязательно: папка <image_id>.jpg для аудита дублей")
    parser.add_argument("--json", type=Path, help="куда сохранить полный JSON-отчёт")
    parser.add_argument("--strict", action="store_true", help="вернуть ненулевой код при warning")
    args = parser.parse_args()

    try:
        source = read_csv(args.source_csv)
        split = read_csv(args.split_csv)
        require_columns(source, REQUIRED_SOURCE_COLUMNS, args.source_csv)
        require_columns(split, REQUIRED_SPLIT_COLUMNS, args.split_csv)
        training_rows = read_csv(args.training_csv) if args.training_csv else None
        if training_rows is not None:
            require_columns(training_rows, REQUIRED_SOURCE_COLUMNS, args.training_csv)
        report = audit(source, split, args.images_dir, training_rows)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"отчёт сохранён: {args.json}", file=sys.stderr)

    if report["failures"] or (args.strict and report["warnings"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
