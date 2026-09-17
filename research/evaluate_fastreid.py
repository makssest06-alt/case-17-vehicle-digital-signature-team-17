#!/usr/bin/env python3
"""Замер готовых ReID-весов fast-reid (обучены именно на ReID автомобилей).

Тот же сплит и та же метрика, что в evaluate.py — цифры сравнимы напрямую.

Веса: https://github.com/JDAI-CV/fast-reid/releases/tag/v0.1.1
Код и веса: Apache License 2.0, Copyright 2019 JD.com Inc. JD AI.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent
FASTREID_ROOT = Path("/opt/case17/third_party/fast-reid")
CROPS = Path("/opt/case17/data/crops/train")

sys.path.insert(0, str(FASTREID_ROOT))
sys.path.insert(0, str(HERE))

from evaluate import evaluate, load_split  # noqa: E402  тот же сплит и метрика


def build(config: str, weights: str):
    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model
    from fastreid.utils.checkpoint import Checkpointer

    cfg = get_cfg()
    cfg.merge_from_file(str(FASTREID_ROOT / config))
    cfg.MODEL.BACKBONE.PRETRAIN = False   # полный чекпоинт грузим сами
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(weights)
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    h, w = cfg.INPUT.SIZE_TEST
    return model, (h, w), mb


@torch.inference_mode()
def embed(model, size, rows, batch: int):
    h, w = size
    feats, t0 = [], time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        arrs = []
        for r in chunk:
            im = Image.open(CROPS / f"{r['image_id']}.jpg").convert("RGB").resize((w, h), Image.BILINEAR)
            arrs.append(torch.from_numpy(np.asarray(im, dtype="float32")).permute(2, 0, 1))
        # fast-reid ждёт RGB 0..255 и сам вычитает mean/std внутри preprocess_image
        out = model({"images": torch.stack(arrs)})
        if isinstance(out, dict):
            out = out.get("features", next(iter(out.values())))
        feats.append(torch.nn.functional.normalize(out, p=2, dim=1).cpu().numpy())
    return np.vstack(feats).astype("float32"), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="путь конфига внутри fast-reid")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dump-scores", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(2)
    qry, gal = load_split()
    print(f"[{a.tag}] query {len(qry)}, gallery {len(gal)}", flush=True)

    t0 = time.perf_counter()
    model, size, mb = build(a.config, a.weights)
    print(f"  модель загружена за {time.perf_counter() - t0:.1f} с, весов {mb:.0f} МБ, вход {size[0]}x{size[1]}", flush=True)

    g_emb, t_g = embed(model, size, gal, a.batch)
    q_emb, t_q = embed(model, size, qry, a.batch)
    n, t = len(gal) + len(qry), t_g + t_q
    print(f"  {n} кадров за {t:.1f} с -> {n / t:.2f} кадров/с на CPU", flush=True)

    res = evaluate(q_emb, g_emb, qry, gal, dump_scores=a.dump_scores)
    res.update(model=a.tag, dim=int(q_emb.shape[1]), input_size=f"{size[0]}x{size[1]}",
               weights_mb=round(mb, 1), images_per_sec=round(n / t, 2))
    print(f"  mAP {res['mAP']}  Rank-1 {res['rank1']}  Rank-5 {res['rank5']}  (query {res['queries_used']})", flush=True)

    out = HERE / "results"; out.mkdir(exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
