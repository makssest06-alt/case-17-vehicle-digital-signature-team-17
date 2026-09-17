#!/usr/bin/env python3
"""Сборка трёх артефактов сдачи по требованиям README датасета.

Запускать внутри контейнера api:
    docker compose exec api python scripts/make_submission.py

На выходе в ARTIFACTS_DIR:
  submission.csv   query_id,gallery_id_1..gallery_id_10  — топ-10 по убыванию близости
  embeddings.npy   float32 (N_query + N_gallery, dim); сначала все query
                   в порядке test_query.csv, затем все gallery в порядке test_gallery.csv
  candidates.csv   query_id,gallery_id,confidence — только принятые кандидаты;
                   если для query ничего не прошло порог, строк для него нет (отказ)
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/srv")

from app.config import settings          # noqa: E402
from app.models.registry import build_model  # noqa: E402

TOP_K = 10


def read_ids(csv_path: Path) -> list:
    with csv_path.open(newline="") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


def embed_folder(model, folder: Path, ids: list, batch: int, label: str) -> np.ndarray:
    out = np.zeros((len(ids), model.dim), dtype="float32")
    t0 = time.perf_counter()
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        imgs = [Image.open(folder / f"{i}.jpg").convert("RGB") for i in chunk]
        out[start : start + len(chunk)] = model.embed_batch(imgs)
        done = start + len(chunk)
        if done % (batch * 10) == 0 or done == len(ids):
            speed = done / (time.perf_counter() - t0)
            print(f"  {label}: {done}/{len(ids)}  {speed:.1f} кадров/с", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(settings.data_dir, "crops"))
    ap.add_argument("--out", default=settings.artifacts_dir)
    ap.add_argument("--threshold", type=float, default=settings.search_threshold)
    ap.add_argument("--batch", type=int, default=settings.batch_size)
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q_ids = read_ids(data / "crops_query.csv")
    g_ids = read_ids(data / "crops_gallery.csv")
    print(f"query: {len(q_ids)}, gallery: {len(g_ids)}")
    if len(g_ids) < TOP_K:
        print(f"ВНИМАНИЕ: в галерее всего {len(g_ids)} объектов, топ-10 не набрать", file=sys.stderr)

    model = build_model(settings.embed_model, settings.input_size)
    t0 = time.perf_counter()
    model.load()
    print(f"модель {model.name} загружена за {time.perf_counter() - t0:.1f} с, dim={model.dim}")

    t_embed = time.perf_counter()
    q_emb = embed_folder(model, data / "query", q_ids, args.batch, "query")
    g_emb = embed_folder(model, data / "gallery", g_ids, args.batch, "gallery")
    embed_s = time.perf_counter() - t_embed
    total = len(q_ids) + len(g_ids)
    print(f"эмбеддинги посчитаны за {embed_s:.1f} с — {total / embed_s:.1f} кадров/с")

    # векторы уже L2-нормированы, значит скалярное произведение = косинус
    sim = q_emb @ g_emb.T
    order = np.argsort(-sim, axis=1)

    sub_path = out / "submission.csv"
    with sub_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, TOP_K + 1)])
        for qi, q in enumerate(q_ids):
            top = order[qi, :TOP_K]
            w.writerow([q] + [g_ids[j] for j in top])

    emb_path = out / "embeddings.npy"
    np.save(emb_path, np.vstack([q_emb, g_emb]).astype("float32"))

    cand_path = out / "candidates.csv"
    accepted, refused = 0, 0
    with cand_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for qi, q in enumerate(q_ids):
            rows = [
                (q, g_ids[j], round(float(sim[qi, j]), 4))
                for j in order[qi, :TOP_K]
                if float(sim[qi, j]) >= args.threshold
            ]
            if rows:
                w.writerows(rows)
                accepted += 1
            else:
                refused += 1

    print()
    print(f"submission.csv   {sub_path}  строк {len(q_ids)}")
    print(f"embeddings.npy   {emb_path}  форма {(total, model.dim)}")
    print(f"candidates.csv   {cand_path}  порог {args.threshold}")
    print(f"  принято query: {accepted}, отказов: {refused} ({refused / len(q_ids) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
