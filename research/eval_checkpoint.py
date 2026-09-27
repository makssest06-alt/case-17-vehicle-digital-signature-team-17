#!/usr/bin/env python3
"""Замер дообученного чекпоинта finetune_clip.py на cross-camera валидации.

Чекпоинт содержит полный FullModel (encoder + ReidHead), поэтому
восстанавливаем ту же архитектуру, грузим веса и считаем эмбеддинги
по тем же кадрам, что и обучение: crop bbox + 10% запаса из full-frame.

Метрики как у жюри: кандидаты с той же камерой, что у query, исключаются.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import backbones
from finetune_clip import FullModel

HERE = Path(__file__).parent
MODELS = Path("D:/bm/gpu_models_siglip")
IMAGES = Path("D:/bm/dataset_extracted/images")
TRAIN_CSV = Path("D:/bm/dataset_extracted/train.csv")


def load_split():
    rows = list(csv.DictReader((HERE / "val_split.csv").open(encoding="utf-8")))
    box = {r["image_id"]: r for r in csv.DictReader(TRAIN_CSV.open(encoding="utf-8"))}
    for r in rows:
        b = box[r["image_id"]]
        r["x"], r["y"], r["w"], r["h"] = b["x"], b["y"], b["w"], b["h"]
    gal = [r for r in rows if r["role"] == "gallery"]
    qry = [r for r in rows if r["role"] == "query"]
    return qry, gal


def crop(img: Image.Image, r, context: float) -> Image.Image:
    x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
    px, py = w * context, h * context
    left = max(0, int(round(x - px)))
    top = max(0, int(round(y - py)))
    right = min(img.width, int(round(x + w + px)))
    bottom = min(img.height, int(round(y + h + py)))
    return img.crop((left, top, right, bottom))


def build(spec):
    import timm

    encoder = timm.create_model(spec.model_name, pretrained=False, num_classes=0)
    cfg = timm.data.resolve_data_config({}, model=encoder)
    cfg["input_size"] = (3, spec.input_size, spec.input_size)
    tf = timm.data.create_transform(**cfg, is_training=False)
    return encoder, tf


@torch.inference_mode()
def embed(model, tf, rows, batch: int, context: float, device, flip: bool):
    feats, t0 = [], time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        imgs = [tf(crop(Image.open(IMAGES / f"{r['image_id']}.jpg").convert("RGB"), r, context)) for r in chunk]
        x = torch.stack(imgs).to(device)
        f = model(x)[1]
        if flip:
            f = (f + model(torch.flip(x, dims=[3]))[1]) / 2
        feats.append(torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy())
    return np.vstack(feats).astype("float32"), time.perf_counter() - t0


def evaluate(q_emb, g_emb, qry, gal):
    g_vid = np.array([r["vehicle_id"] for r in gal])
    g_cam = np.array([r["camera_id"] for r in gal])
    sim = q_emb @ g_emb.T
    aps, r1, r5, used = [], 0, 0, 0
    for i, q in enumerate(qry):
        keep = g_cam != q["camera_id"]
        if not keep.any():
            continue
        s = sim[i][keep]
        rel = g_vid[keep] == q["vehicle_id"]
        if not rel.any():
            continue
        used += 1
        order = np.argsort(-s)
        rel_sorted = rel[order]
        hits = np.flatnonzero(rel_sorted)
        prec = (np.arange(len(hits)) + 1) / (hits + 1)
        aps.append(float(prec.mean()))
        r1 += int(rel_sorted[0])
        r5 += int(rel_sorted[:5].any())
    return {
        "queries_used": used,
        "mAP": round(float(np.mean(aps)) * 100, 2),
        "rank1": round(r1 / used * 100, 2),
        "rank5": round(r5 / used * 100, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="siglip_base")
    ap.add_argument("--context", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--flip", action="store_true", help="усреднить с горизонтальным отражением")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = backbones.get(a.backbone)
    ckpt = torch.load(a.ckpt, map_location="cpu")
    args = ckpt["args"]
    encoder, tf = build(spec)
    model = FullModel(encoder, spec.dim, args["hidden"], args["out_dim"], args["dropout"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    qry, gal = load_split()
    print(f"[{Path(a.ckpt).name}] query {len(qry)}, gallery {len(gal)}, device {device}", flush=True)

    g_emb, t_g = embed(model, tf, gal, a.batch, a.context, device, a.flip)
    q_emb, t_q = embed(model, tf, qry, a.batch, a.context, device, a.flip)
    n, t = len(gal) + len(qry), t_g + t_q
    res = evaluate(q_emb, g_emb, qry, gal)
    res.update(ckpt=Path(a.ckpt).name, context=a.context, flip=a.flip, dim=int(q_emb.shape[1]),
               images_per_sec=round(n / t, 2))
    print(f"  mAP {res['mAP']}  Rank-1 {res['rank1']}  Rank-5 {res['rank5']}  "
          f"({n} кадров, {n/t:.1f} кадр/с)", flush=True)

    outdir = HERE / "results"; outdir.mkdir(exist_ok=True)
    (outdir / f"ckpt_{a.tag or Path(a.ckpt).stem}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
