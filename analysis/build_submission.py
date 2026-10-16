#!/usr/bin/env python3
"""Сборка итоговых артефактов для сдачи (реальный тест: 1110 query / 750 gallery).

Конфигурация — лучшая по результатам экспериментов:
    siglip2-squash 0.45 + sq30-squash 0.25 + ep29-base 0.30,
    затем постобработка: qq-expand -> k-reciprocal re-ranking -> CSLS.

Две squash-модели видят всю машину целиком, ep29 — base-кроп
(Resize 248 + CenterCrop 224). Ошибки частично независимы, поэтому смесь
сильнее каждой модели по отдельности.

Три модели, а не четыре: geom-base вытесняется в нулевой вес — полный
перебор по симплексу даёт с ним то же, что и без него (проверено в
`mix4_grid.py`).

Прежняя лучшая (geom-squash 0.8 + ep29-base 0.2) давала mAP@10 0.8319.
Предыдущая тройка (sq30 0.6 + ep29-base 0.2 + geom-base 0.2) — 0.8671.

Пишет в `submission_out/`:
    submission.csv   — query_id,gallery_id_1..10 (без заголовка, как в ТЗ)
    embeddings.npy   — порядок: query, затем gallery
    candidates.csv   — query_id,gallery_id,confidence

    python build_submission.py
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from extract_emb import embed_rows, load_model
from postproc_v2 import pipe
from reid_postprocess import l2
from train_reid import build_transforms

HERE = Path(__file__).parent
IMAGES = HERE / "dataset" / "images"
MODELS = HERE.parent / "test-clone" / "repo" / "models"
EP29_CKPT = MODELS / "siglip_vitb16_reid_ep29.pt"
SQ30_CKPT = HERE / "gpu_models" / "sq_scratch30_ep26.pt"
SIGLIP2_CKPT = HERE / "gpu_models" / "siglip2_scratch30_ep28.pt"

# (чекпоинт, режим кропа, вес). Сумма весов = 1.
CONFIG = [
    (SIGLIP2_CKPT, "squash", 0.45),
    (SQ30_CKPT, "squash", 0.25),
    (EP29_CKPT, "timm", 0.30),
]


def read_split(csv_path):
    return list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))


def embed_ckpt(ckpt, rows, tf_mode, device, bs, tta, context=0.10):
    model, spec, _ = load_model(ckpt, device)
    tf = build_transforms(spec, tf_mode, train=False)
    e = embed_rows(model, tf, rows, IMAGES, context, device, tta=tta, bs=bs)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return l2(e.astype("float64"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "submission_out"))
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--no-tta", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = not a.no_tta
    print(f"устройство {device}, tta={tta}", flush=True)

    qry = read_split(HERE / "dataset" / "test_query.csv")
    gal = read_split(HERE / "dataset" / "test_gallery.csv")
    print(f"query {len(qry)}, gallery {len(gal)}", flush=True)

    q = np.zeros((len(qry), 512))
    g = np.zeros((len(gal), 512))
    for ckpt, tf_mode, weight in CONFIG:
        print(f"  {Path(ckpt).name} [{tf_mode}] вес {weight}", flush=True)
        q += weight * embed_ckpt(ckpt, qry, tf_mode, device, a.bs, tta)
        g += weight * embed_ckpt(ckpt, gal, tf_mode, device, a.bs, tta)
    q, g = l2(q), l2(g)

    # embeddings.npy: порядок query, затем gallery (как в ТЗ)
    np.save(out / "embeddings.npy", np.vstack([q, g]).astype("float32"))

    S = pipe(q, g)
    order = np.argsort(-S, axis=1)

    with (out / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        for qi in range(len(qry)):
            wr.writerow([qry[qi]["image_id"]] +
                        [gal[j]["image_id"] for j in order[qi, :10]])
    print(f"submission.csv -> {len(qry)} строк (без заголовка)", flush=True)

    conf = S[np.arange(len(qry))[:, None], order[:, :1]].ravel()
    thr = float(np.quantile(conf, 0.10))
    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["query_id", "gallery_id", "confidence"])
        kept = 0
        for qi in range(len(qry)):
            c = float(conf[qi])
            if c >= thr:
                wr.writerow([qry[qi]["image_id"], gal[order[qi, 0]]["image_id"], round(c, 4)])
                kept += 1
    print(f"candidates.csv -> принято {kept}, отказов {len(qry) - kept} (порог {thr:.4f})",
          flush=True)
    print(f"embeddings.npy -> {q.shape[0] + g.shape[0]} x {q.shape[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
