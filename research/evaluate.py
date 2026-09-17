#!/usr/bin/env python3
"""Замер готовой модели на cross-camera валидации. Без обучения.

Протокол как у жюри: для каждого query кандидаты из gallery ранжируются
по косинусу, кадры С ТОЙ ЖЕ камеры, что у query, исключаются из выдачи.
Релевантным считается кандидат с тем же vehicle_id.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CROPS = Path("/opt/case17/data/crops/train")
HERE = Path(__file__).parent


def load_split():
    rows = list(csv.DictReader((HERE / "val_split.csv").open()))
    gal = [r for r in rows if r["role"] == "gallery"]
    qry = [r for r in rows if r["role"] == "query"]
    return qry, gal


def build(model_name: str, input_size: int):
    import timm  # локальный импорт: модуль переиспользуется из окружения fast-reid, где timm не нужен

    kw = dict(pretrained=True, num_classes=0)
    try:
        model = timm.create_model(model_name, img_size=input_size, **kw)
    except TypeError:
        model = timm.create_model(model_name, **kw)
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    cfg["input_size"] = (3, input_size, input_size)
    tf = timm.data.create_transform(**cfg, is_training=False)
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    return model, tf, mb


@torch.inference_mode()
def embed(model, tf, rows, batch: int):
    feats, t0 = [], time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        x = torch.stack([tf(Image.open(CROPS / f"{r['image_id']}.jpg").convert("RGB")) for r in chunk])
        f = model(x)
        feats.append(torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy())
    return np.vstack(feats).astype("float32"), time.perf_counter() - t0


def evaluate(q_emb, g_emb, qry, gal, dump_scores=False):
    g_vid = np.array([r["vehicle_id"] for r in gal])
    g_cam = np.array([r["camera_id"] for r in gal])
    sim = q_emb @ g_emb.T

    aps, r1, r5, used = [], 0, 0, 0
    pos_top1, neg_top1 = [], []

    for i, q in enumerate(qry):
        keep = g_cam != q["camera_id"]          # исключаем ту же камеру
        if not keep.any():
            continue
        s = sim[i][keep]
        rel = (g_vid[keep] == q["vehicle_id"])
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

        if dump_scores:
            (pos_top1 if rel_sorted[0] else neg_top1).append(float(s[order[0]]))

    out = {
        "queries_used": used,
        "mAP": round(float(np.mean(aps)) * 100, 2),
        "rank1": round(r1 / used * 100, 2),
        "rank5": round(r5 / used * 100, 2),
    }
    if dump_scores:
        out["top1_correct"] = pos_top1
        out["top1_wrong"] = neg_top1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dump-scores", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(2)
    qry, gal = load_split()
    print(f"[{a.model}] query {len(qry)}, gallery {len(gal)}")

    t0 = time.perf_counter()
    model, tf, mb = build(a.model, a.input_size)
    load_s = time.perf_counter() - t0
    print(f"  модель загружена за {load_s:.1f} с, весов {mb:.0f} МБ")

    g_emb, t_g = embed(model, tf, gal, a.batch)
    q_emb, t_q = embed(model, tf, qry, a.batch)
    n, t = len(gal) + len(qry), t_g + t_q
    ips = n / t
    print(f"  {n} кадров за {t:.1f} с -> {ips:.2f} кадров/с на CPU")

    res = evaluate(q_emb, g_emb, qry, gal, dump_scores=a.dump_scores)
    res.update(
        model=a.model, dim=int(q_emb.shape[1]), input_size=a.input_size,
        weights_mb=round(mb, 1), images_per_sec=round(ips, 2), load_s=round(load_s, 1),
    )
    print(f"  mAP {res['mAP']}  Rank-1 {res['rank1']}  Rank-5 {res['rank5']}  (query {res['queries_used']})")

    outdir = HERE / "results"; outdir.mkdir(exist_ok=True)
    (outdir / f"{a.tag or a.model.split('.')[0]}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
