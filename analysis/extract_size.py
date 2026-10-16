#!/usr/bin/env python3
"""Extract embeddings at an arbitrary squash size (for multi-scale TTA)."""
import argparse
import csv
import dataclasses
from pathlib import Path

import numpy as np
import torch

import backbones
from head_model import ReidHead
from train_reid import build_transforms
from extract_emb import FullModel, crop_image, embed_rows

HERE = Path(__file__).parent


def load_model_at(ckpt_path, size, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    spec_base = backbones.get(a.get("backbone", "siglip_base"))
    spec = dataclasses.replace(spec_base, input_size=size)
    # build encoder at its NATIVE size so pos_embed matches the checkpoint;
    # dynamic_img_size interpolates pos_embed on the fly for other input sizes.
    _, _, encoder = backbones.build_encoder(spec_base, pretrained=False,
                                            input_size=spec_base.input_size)
    model = FullModel(encoder, spec.dim, a.get("hidden", 1024),
                      a.get("out_dim", 512), a.get("dropout", 0.3))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"WARN missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    return model, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--tf-mode", default="squash")
    ap.add_argument("--val-csv", default=r"D:\bm\research\ensemble\val_split.csv")
    ap.add_argument("--boxes-csv", default=r"D:\bm\dataset_extracted\train.csv")
    ap.add_argument("--images-dir", default=r"D:\bm\dataset_extracted\images")
    ap.add_argument("--out-dir", default=r"D:\bm\research\ensemble")
    ap.add_argument("--context", type=float, default=0.10)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, spec = load_model_at(a.ckpt, a.size, device)
    tf = build_transforms(spec, a.tf_mode, train=False)
    print(f"size={a.size} tf={a.tf_mode} tta={a.tta}", flush=True)

    rows = list(csv.DictReader(Path(a.val_csv).open(encoding="utf-8")))
    qry = [r for r in rows if r["role"] == "query"]
    gal = [r for r in rows if r["role"] == "gallery"]
    boxes = {r["image_id"]: r for r in csv.DictReader(Path(a.boxes_csv).open(encoding="utf-8"))}
    for r in qry + gal:
        b = boxes[r["image_id"]]
        for k in ("x", "y", "w", "h"):
            r[k] = b[k]

    for name, subset in (("qry", qry), ("gal", gal)):
        emb = embed_rows(model, tf, subset, Path(a.images_dir), a.context, device,
                         tta=a.tta, bs=a.bs)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        out = Path(a.out_dir) / f"emb_{a.tag}_{name}.npy"
        np.save(out, emb.astype("float32"))
        print(f"  {out.name}: {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
