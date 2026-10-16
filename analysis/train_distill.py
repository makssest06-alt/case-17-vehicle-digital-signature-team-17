#!/usr/bin/env python3
"""Distillation: train ONE student to mimic the 4-model ensemble.

Loss = ArcFace + batch-hard Triplet  +  alpha * (1 - cos(student_bn, teacher))
Teacher = weighted ensemble embeddings on the train split (precomputed).

    python train_distill.py --backbone siglip2_base --init-from siglip2_init.pt \
        --teacher D:/bm/improve_models/teacher_emb.npy \
        --rows-csv D:/bm/improve_models/distill_train_rows.csv \
        --tag siglip2_distill --alpha 2.0 --epochs 30
"""
import argparse
import csv
import dataclasses
import math
import os
import random
import time
from pathlib import Path

# Обход системного SOCKS-прокси при скачивании pretrained-весов timm.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
try:
    import httpx
    _orig_client_init = httpx.Client.__init__

    def _client_init(self, *args, **kwargs):
        kwargs["trust_env"] = False
        _orig_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _client_init
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import backbones
from head_model import MarginHead, ReidHead, batch_hard_triplet
from train_reid import (GeomAug, NightAndGlare, PKBatchSampler, RandomErasing,
                        build_transforms)

HERE = Path(__file__).parent
DEFAULT_IMAGES = Path(r"D:\bm\dataset_extracted\images")


class DistillDataset(Dataset):
    """Returns (x, label, idx) so we can look up the teacher embedding."""

    def __init__(self, rows, tf, aug, images_dir, context=0.10):
        self.rows, self.tf, self.aug = rows, tf, aug
        self.images_dir = Path(images_dir)
        self.context = context

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.images_dir / f"{r['image_id']}.jpg").convert("RGB")
        x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
        px, py = w * self.context, h * self.context
        left = max(0, int(round(x - px))); top = max(0, int(round(y - py)))
        right = min(img.width, int(round(x + w + px)))
        bottom = min(img.height, int(round(y + h + py)))
        img = img.crop((left, top, right, bottom))
        if self.aug is not None:
            img = self.aug(img)
        x = self.tf(img)
        x = RandomErasing()(x)
        return x, r["_label"], i


class FullModel(torch.nn.Module):
    def __init__(self, encoder, dim, hidden, out_dim, dropout):
        super().__init__()
        self.encoder = encoder
        self.head = ReidHead(dim, hidden, out_dim, dropout)

    def forward(self, x):
        return self.head(self.encoder(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="siglip2_base")
    ap.add_argument("--init-from", default="", help="локальный case17-чекпоинт; пусто = pretrained timm")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--rows-csv", required=True)
    ap.add_argument("--images-dir", default=str(DEFAULT_IMAGES))
    ap.add_argument("--models-dir", default=r"D:\bm\improve_models")
    ap.add_argument("--tag", default="siglip2_distill")
    ap.add_argument("--alpha", type=float, default=2.0, help="distillation weight")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--context", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    spec_base = backbones.get(a.backbone)
    spec = dataclasses.replace(spec_base, input_size=224)
    init = (torch.load(a.init_from, map_location="cpu", weights_only=False)
            if a.init_from else None)
    _, _, encoder = backbones.build_encoder(spec_base, pretrained=init is None,
                                            input_size=224)
    if init is not None:
        enc_sd = {k[len("encoder."):]: v for k, v in init["model"].items()
                  if k.startswith("encoder.")}
        encoder.load_state_dict(enc_sd, strict=True)
        print(f"student init from {Path(a.init_from).name} ({len(enc_sd)} tensors)",
              flush=True)
    else:
        print(f"student init: pretrained timm {spec_base.model_name}", flush=True)
    for p in encoder.parameters():
        p.requires_grad_(True)

    rows = list(csv.DictReader(Path(a.rows_csv).open(encoding="utf-8")))
    teacher = np.load(a.teacher).astype("float32")
    assert len(rows) == len(teacher), (len(rows), len(teacher))
    vids = sorted({r["vehicle_id"] for r in rows})
    lab = {v: i for i, v in enumerate(vids)}
    for r in rows:
        r["_label"] = lab[r["vehicle_id"]]
    teacher_t = torch.from_numpy(teacher).to(device)
    print(f"train {len(rows)} imgs / {len(vids)} vehicles, alpha={a.alpha}", flush=True)

    tf = build_transforms(spec, "squash", train=False)
    aug = T.Compose([GeomAug(size=224), NightAndGlare(size=224)])
    ds = DistillDataset(rows, tf, aug, a.images_dir, a.context)
    sampler = PKBatchSampler(rows, a.P, a.K, a.seed, cross_cam=True)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=a.workers, pin_memory=True)

    model = FullModel(encoder, spec.dim, 1024, 512, 0.3).to(device)
    margin = MarginHead(512, len(vids), "arcface").to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": a.lr_backbone},
        {"params": model.head.parameters(), "lr": a.lr_head},
        {"params": margin.parameters(), "lr": a.lr_head},
    ], weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)

    steps_total = a.epochs * max(1, len(sampler))
    step, t0 = 0, time.perf_counter()
    out = Path(a.models_dir); out.mkdir(parents=True, exist_ok=True)

    for ep in range(a.epochs):
        model.train(); margin.train()
        for x, y, idx in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            idx = idx.to(device)
            frac = step / max(1, steps_total)
            scale = (step + 1) / max(1, a.warmup * len(sampler)) if frac < 0.1 else 0.5 * (1 + math.cos(math.pi * frac))
            for g, base in zip(opt.param_groups, (a.lr_backbone, a.lr_head, a.lr_head)):
                g["lr"] = base * min(1.0, scale)
            with torch.amp.autocast("cuda", enabled=a.amp):
                feat, feat_bn = model(x)
                loss = margin(feat_bn, y) + batch_hard_triplet(feat, y, 0.3)
                tgt = teacher_t[idx]
                cos = F.cosine_similarity(F.normalize(feat_bn, dim=1),
                                          F.normalize(tgt, dim=1), dim=1)
                loss = loss + a.alpha * (1 - cos).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            step += 1
            if step % 100 == 0:
                print(f"  STEP {step}/{steps_total} loss={loss.item():.3f} "
                      f"cos={cos.mean().item():.3f} s/step="
                      f"{(time.perf_counter()-t0)/step:.2f}", flush=True)
        torch.save({"model": model.state_dict(), "margin": margin.state_dict(), "args": vars(a)},
                   out / f"{a.tag}_ep{ep+1}.pt")
        print(f"  saved {a.tag}_ep{ep+1}.pt", flush=True)
    print(f"done {step} steps in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
