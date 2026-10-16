#!/usr/bin/env python3
"""Build test-set artifacts with the 4-model ensemble + time it.

Outputs (in --out-dir): submission.csv, embeddings.npy, candidates.csv.
Order of embeddings.npy: all test_query in file order, then all test_gallery.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from extract_emb import embed_rows, load_model
from train_reid import build_transforms
from reid_postprocess import l2
import reid_postprocess as rp

HERE = Path(__file__).parent
IMAGES = Path(r"D:\bm\dataset_extracted\images")
TESTQ = Path(r"D:\bm\dataset_extracted\test_query.csv")
TESTG = Path(r"D:\bm\dataset_extracted\test_gallery.csv")
TRAIN = Path(r"D:\bm\dataset_extracted\train.csv")
MODELS = Path(r"D:\bm_improve\models")
EXTRA = Path(r"D:\bm\improve_models")

# (name, path, tf_mode, weight)
CONFIG = [
    ("siglip2", MODELS / "siglip2_reid_scratch30_ep28.pt", "squash", 0.364),
    ("sq30", MODELS / "siglip_reid_scratch30_ep26.pt", "squash", 0.232),
    ("ep29", MODELS / "siglip_vitb16_reid_ep29.pt", "timm", 0.182),
    ("clip", EXTRA / "clip_reid_scratch30_ep28.pt", "squash", 0.222),
]
RR = dict(qq_k=1, qq_alpha=3.0, csls_k=5, rerank_lambda=0.7, rerank_k1=30, rerank_k2=6)


def read_rows(p):
    return list(csv.DictReader(p.open(encoding="utf-8")))


def attach_boxes(rows, boxes):
    for r in rows:
        b = boxes[r["image_id"]]
        for k in ("x", "y", "w", "h"):
            r[k] = b[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=r"D:\bm\test_artifacts")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--tta", action="store_true", default=True)
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    qry = read_rows(TESTQ)
    gal = read_rows(TESTG)
    boxes = {r["image_id"]: r for r in read_rows(TRAIN)}
    # test bbox comes from test_query/test_gallery themselves
    for r in qry + gal:
        pass  # x,y,w,h already present
    print(f"query {len(qry)}, gallery {len(gal)}", flush=True)

    t0 = time.perf_counter()
    q = np.zeros((len(qry), 512), dtype="float64")
    g = np.zeros((len(gal), 512), dtype="float64")
    per_model = {}
    for name, path, tfm, w in CONFIG:
        tm = time.perf_counter()
        model, spec, _ = load_model(path, device)
        tf = build_transforms(spec, tfm, train=False)
        eq = embed_rows(model, tf, qry, IMAGES, 0.10, device, tta=a.tta, bs=a.bs)
        eg = embed_rows(model, tf, gal, IMAGES, 0.10, device, tta=a.tta, bs=a.bs)
        q += w * l2(eq.astype("float64"))
        g += w * l2(eg.astype("float64"))
        dt = time.perf_counter() - tm
        per_model[name] = round(dt, 1)
        print(f"  {name:8s} {len(qry)+len(gal)} imgs in {dt:.1f}s "
              f"({(len(qry)+len(gal))/dt:.1f} img/s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    q, g = l2(q), l2(g)
    total = time.perf_counter() - t0
    print(f"embeddings total {total:.1f}s", flush=True)

    # embeddings.npy: query then gallery (file order)
    np.save(out / "embeddings.npy", np.vstack([q, g]).astype("float32"))

    # postprocess
    t1 = time.perf_counter()
    sim, _, _ = rp.rank([q], [g], [1.0], **RR)
    post_s = time.perf_counter() - t1
    order = np.argsort(-sim, axis=1)
    print(f"postprocess {post_s:.2f}s", flush=True)

    with (out / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, 11)])
        for qi, qr in enumerate(qry):
            wr.writerow([qr["image_id"]] + [gal[j]["image_id"] for j in order[qi, :10]])

    conf = sim[np.arange(len(qry))[:, None], order[:, :1]].ravel()
    thr = float(np.quantile(conf, 0.10))
    kept = 0
    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["query_id", "gallery_id", "confidence"])
        for qi, qr in enumerate(qry):
            c = float(conf[qi])
            if c >= thr:
                wr.writerow([qr["image_id"], gal[order[qi, 0]]["image_id"], round(c, 4)])
                kept += 1
    print(f"candidates kept {kept}/{len(qry)} (thr {thr:.4f})", flush=True)

    meta = dict(device=str(device), query=len(qry), gallery=len(gal),
                embed_total_s=round(total, 1), per_model_s=per_model,
                postprocess_s=round(post_s, 2),
                embed_per_image_ms=round(total / (len(qry) + len(gal)) * 1000, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
