#!/usr/bin/env python3
"""Прогон серии чекпоинтов: извлечение эмбеддингов + официальная метрика.

Удобно смотреть траекторию обучения по эпохам и выбирать лучшую.

    python eval_sweep.py --glob "gpu_models/squash_ft_ep*.pt" --tf-mode squash --tta
    python eval_sweep.py --glob "gpu_models/*.pt" --tf-mode squash --ref geom
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch

from extract_emb import embed_rows, load_model, DEFAULT_IMAGES
from official_protocol import official_metrics
from postproc_v2 import pipe
from rerank_test import l2, load_rows
from train_reid import build_transforms

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--tf-mode", default="timm", choices=["timm", "squash"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--ref", default="", help="тег эталона для колонки сравнения")
    ap.add_argument("--val-csv", default=str(HERE / "val_split.csv"))
    ap.add_argument("--boxes-csv", default=str(HERE / "dataset" / "train.csv"))
    ap.add_argument("--context", type=float, default=0.10)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(a.glob))
    if not ckpts:
        raise SystemExit(f"ничего не найдено по {a.glob}")
    print(f"устройство {device}, чекпоинтов {len(ckpts)}\n", flush=True)

    rows = list(csv.DictReader(Path(a.val_csv).open(encoding="utf-8")))
    qry = [r for r in rows if r["role"] == "query"]
    gal = [r for r in rows if r["role"] == "gallery"]
    boxes = {r["image_id"]: r for r in
             csv.DictReader(Path(a.boxes_csv).open(encoding="utf-8"))}
    for r in qry + gal:
        b = boxes[r["image_id"]]
        for k in ("x", "y", "w", "h"):
            r[k] = b[k]
    images_dir = Path(DEFAULT_IMAGES)

    ref = None
    if a.ref:
        rq = l2(np.load(HERE / f"emb_{a.ref}_qry.npy").astype("float64"))
        rg = l2(np.load(HERE / f"emb_{a.ref}_gal.npy").astype("float64"))
        ref = official_metrics(pipe(rq, rg), qry, gal)
        print(f"эталон {a.ref}: mAP {ref['mAP@10']:.2f}  Rank-1 {ref['Rank-1']:.2f}\n")

    print(f"{'чекпоинт':44s} {'base mAP':>9s} {'base R1':>8s} "
          f"{'post mAP':>9s} {'post R1':>8s} {'d mAP':>7s}")
    print("-" * 90)
    for path in ckpts:
        model, spec, _ = load_model(path, device)
        tf = build_transforms(spec, a.tf_mode, train=False)
        embs = {}
        for name, subset in (("qry", qry), ("gal", gal)):
            e = embed_rows(model, tf, subset, images_dir, a.context, device,
                           tta=a.tta, bs=a.bs)
            embs[name] = l2(e.astype("float64"))
        base = official_metrics(embs["qry"] @ embs["gal"].T, qry, gal)
        post = official_metrics(pipe(embs["qry"], embs["gal"]), qry, gal)
        delta = "" if ref is None else f"{post['mAP@10'] - ref['mAP@10']:+7.2f}"
        print(f"{Path(path).name:44s} {base['mAP@10']:9.2f} {base['Rank-1']:8.2f} "
              f"{post['mAP@10']:9.2f} {post['Rank-1']:8.2f} {delta:>7s}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
