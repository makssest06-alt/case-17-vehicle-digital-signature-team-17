#!/usr/bin/env python3
"""Fine-tune the A10 student on look-alike vehicles (local research, not production).

Starts from the full A10 checkpoint (encoder + ReID head + ArcFace margin if stored) and keeps
the A10 recipe (ArcFace + batch-hard triplet + alpha * (1 - cos(student, anchor)), cross-camera
PK sampling, same augmentations), with two changes:
  * anchor = A10's own embedding of each train row (self-distillation) instead of the old
    teacher, so the student stays close to A10 and no non-clean teacher data is used;
  * --mine > 0: in every PK batch, that fraction of the P vehicles are the nearest look-alike
    vehicles (by A10 centroid cosine) of the other vehicles in the batch, so batch-hard
    triplet and ArcFace see the confusions that cause most top-1 errors.
--mine 0 is the control run (identical except sampling).

    python ft_hardmine.py --init clean_student_a10_ep28.pt --rows-csv clean_train_rows.csv \
        --anchor a10_anchor.npy --images-dir <images> --out-dir runs --tag hm --mine 0.5
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

import backbones
from head_model import MarginHead, batch_hard_triplet
from train_distill import DistillDataset, FullModel
from train_reid import GeomAug, NightAndGlare, PKBatchSampler, build_transforms


class LookAlikePKSampler(PKBatchSampler):
    """PK batches where a `mine` fraction of vehicles are look-alikes of the others."""

    def __init__(self, rows, P, K, seed, centroids: np.ndarray, vids: list[str], mine: float, topn: int = 10):
        super().__init__(rows, P, K, seed, cross_cam=True)
        assert list(self.vids) == list(vids)
        self.n_mined = int(round(P * mine))
        sim = centroids @ centroids.T
        np.fill_diagonal(sim, -np.inf)
        self.near = np.argsort(-sim, axis=1)[:, :topn]

    def __iter__(self):
        order = list(range(len(self.vids)))
        self.rng.shuffle(order)
        used: set[int] = set()
        n_seed = self.P - self.n_mined
        for s in order:
            if s in used:
                continue
            seeds = [s] + [v for v in order if v not in used and v != s][: n_seed - 1]
            if len(seeds) < n_seed:
                break
            used.update(seeds)
            batch_v = list(seeds)
            for seed in self.rng.sample(seeds, len(seeds)):
                if len(batch_v) == self.P:
                    break
                cands = [int(v) for v in self.near[seed] if int(v) not in batch_v]
                if cands:
                    pick = self.rng.choice(cands[:5])
                    batch_v.append(pick)
                    used.add(pick)
            while len(batch_v) < self.P:  # not enough look-alikes left: fill randomly
                rest = [v for v in order if v not in used and v not in batch_v]
                if not rest:
                    break
                batch_v.append(rest[0]); used.add(rest[0])
            if len(batch_v) < self.P:
                break
            batch = []
            for v in batch_v:
                batch += self._draw(self.vids[v])
            yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--rows-csv", required=True)
    ap.add_argument("--anchor", required=True, help="(N,512) A10 embeddings aligned to rows-csv")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mine", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr-backbone", type=float, default=3e-6)
    ap.add_argument("--lr-head", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps")
    ap.add_argument("--hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="rectangular squash input, e.g. 224 320 (ViT pos-embed is interpolated)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda")
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("args") or ck.get("cfg") or {})
    spec = dataclasses.replace(backbones.get(str(cfg.get("backbone", "siglip2_base"))), input_size=224)
    _, _, encoder = backbones.build_encoder(spec, pretrained=False, input_size=224)

    rows = list(csv.DictReader(Path(a.rows_csv).open(encoding="utf-8")))
    anchor = np.load(a.anchor).astype("float32")
    assert len(rows) == len(anchor), (len(rows), len(anchor))
    vids = sorted({r["vehicle_id"] for r in rows})
    lab = {v: i for i, v in enumerate(vids)}
    for r in rows:
        r["_label"] = lab[r["vehicle_id"]]

    model = FullModel(encoder, spec.dim, 1024, 512, 0.3)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    margin = MarginHead(512, len(vids), "arcface").to(device)
    if "margin" in ck:
        margin.load_state_dict(ck["margin"], strict=True)
        print("margin: loaded from A10 checkpoint", flush=True)
    else:
        print("margin: NOT in checkpoint, fresh ArcFace", flush=True)

    an = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
    by_vid: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_vid.setdefault(r["vehicle_id"], []).append(i)
    cent = np.stack([an[by_vid[v]].mean(0) for v in vids])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    sampler = (LookAlikePKSampler(rows, a.P, a.K, a.seed, cent, vids, a.mine) if a.mine > 0
               else PKBatchSampler(rows, a.P, a.K, a.seed, cross_cam=True))
    tf = build_transforms(spec, "squash", train=False)
    aug = T.Compose([GeomAug(size=224), NightAndGlare(size=224)])
    if a.hw:  # keep the car's aspect ratio instead of squashing it into a square
        h, w = a.hw
        tf = T.Compose([T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                        *tf.transforms[1:]])
        geom = GeomAug(size=224)
        geom.partcrop = T.RandomResizedCrop(size=(h, w), scale=(0.6, 1.0),
                                            ratio=(0.75 * w / h, 1.3333 * w / h),
                                            interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        aug = T.Compose([geom, NightAndGlare(size=224)])
        cfg["input_hw"] = [h, w]
    dl = DataLoader(DistillDataset(rows, tf, aug, a.images_dir, 0.10), batch_sampler=sampler,
                    num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    anchor_t = torch.from_numpy(an).to(device)

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": a.lr_backbone},
        {"params": model.head.parameters(), "lr": a.lr_head},
        {"params": margin.parameters(), "lr": a.lr_head},
    ], weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda")
    steps_per_epoch = len(vids) // a.P  # same nominal length for both samplers
    steps_total = a.epochs * steps_per_epoch
    warm = max(1, steps_per_epoch // 2)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"train {len(rows)} rows / {len(vids)} vehicles, mine={a.mine}, ~{steps_per_epoch} steps/epoch", flush=True)
    step, t0 = 0, time.perf_counter()
    for ep in range(a.epochs):
        model.train(); margin.train()
        for x, y, idx in dl:
            x, y, idx = x.to(device, non_blocking=True), y.to(device, non_blocking=True), idx.to(device)
            frac = min(step / steps_total, 1.0)
            scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * frac))
            for g, base in zip(opt.param_groups, (a.lr_backbone, a.lr_head, a.lr_head)):
                g["lr"] = base * scale
            with torch.amp.autocast("cuda"):
                feat, feat_bn = model(x)
                l_id = margin(feat_bn, y)
                l_tri = batch_hard_triplet(feat, y, 0.3)
                cos = F.cosine_similarity(F.normalize(feat_bn, dim=1), anchor_t[idx], dim=1)
                loss = l_id + l_tri + a.alpha * (1 - cos).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"  ep{ep + 1} step {step}/{steps_total} loss {loss.item():.3f} id {l_id.item():.3f} "
                      f"tri {l_tri.item():.3f} anchor_cos {cos.mean().item():.4f} "
                      f"s/step {(time.perf_counter() - t0) / step:.2f}", flush=True)
            if a.max_steps and step >= a.max_steps:
                print("smoke test done", flush=True)
                return
        args_out = {**cfg, **{k: v for k, v in vars(a).items() if k not in cfg},
                    "ft_from": a.init, "tf_mode": "squash", "input_size": 224,
                    "backbone": cfg.get("backbone", "siglip2_base")}
        torch.save({"model": model.state_dict(), "margin": margin.state_dict(), "args": args_out},
                   out / f"{a.tag}_ep{ep + 1}.pt")
        print(f"  saved {a.tag}_ep{ep + 1}.pt ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
