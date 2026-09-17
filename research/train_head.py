#!/usr/bin/env python3
"""Обучение ReID-головы поверх замороженных признаков CLIP. Только CPU.

Сплит: 150 машин из val_ids.txt — валидация, все остальные — обучение головы.
В валидацию не подглядываем: ни порог, ни выбор эпохи по ней не настраиваем
внутри обучения, метрики считаются в конце для отчёта.
"""
import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

import backbones
from head_model import MarginHead, ReidHead, batch_hard_triplet

HERE = Path(__file__).parent
FEAT = Path("/opt/case17/data/features")
MODELS = Path("/opt/case17/models")


def load_train_features(backbone: str = "clip_vitb16"):
    idx = list(csv.DictReader((FEAT / "train_index.csv").open()))
    orig = np.load(FEAT / f"train_{backbone}_orig.npy")
    flip = np.load(FEAT / f"train_{backbone}_flip.npy")
    val_ids = {l.strip() for l in (HERE / "val_ids.txt").read_text().split() if l.strip()}
    tr_rows = [(i, r) for i, r in enumerate(idx) if r["vehicle_id"] not in val_ids]
    va_rows = [(i, r) for i, r in enumerate(idx) if r["vehicle_id"] in val_ids]
    return orig, flip, tr_rows, va_rows


class PKSampler:
    """P машин по K кадров в батче — обязательное условие для batch-hard triplet."""

    def __init__(self, rows, P=32, K=4, seed=42):
        self.by_vid = defaultdict(list)
        for pos, (i, r) in enumerate(rows):
            self.by_vid[r["vehicle_id"]].append(pos)
        self.vids = sorted(self.by_vid)
        self.P, self.K = P, K
        self.rng = random.Random(seed)

    def __iter__(self):
        vids = self.vids[:]
        self.rng.shuffle(vids)
        for s in range(0, len(vids) - self.P + 1, self.P):
            batch = []
            for v in vids[s : s + self.P]:
                pool = self.by_vid[v]
                batch += (self.rng.sample(pool, self.K) if len(pool) >= self.K
                          else [self.rng.choice(pool) for _ in range(self.K)])
            yield batch

    def __len__(self):
        return len(self.vids) // self.P


def evaluate_val(head, orig, flip, va_rows, device="cpu"):
    """Кросс-камерная метрика на val: gallery — по одному кадру с каждой камеры."""
    head.eval()
    by_v = defaultdict(lambda: defaultdict(list))
    for pos, (i, r) in enumerate(va_rows):
        by_v[r["vehicle_id"]][r["camera_id"]].append((i, r))
    gal, qry = [], []
    for v in sorted(by_v):
        for cam in sorted(by_v[v], key=int):
            items = sorted(by_v[v][cam], key=lambda t: t[1]["image_id"])
            gal.append(items[0]); qry += items[1:]

    def emb(rows):
        idxs = [i for i, _ in rows]
        with torch.inference_mode():
            a = head(torch.from_numpy(orig[idxs]))[1]
            b = head(torch.from_numpy(flip[idxs]))[1]
        return F.normalize((a + b) / 2, p=2, dim=1).numpy()

    q, g = emb(qry), emb(gal)
    g_vid = np.array([r["vehicle_id"] for _, r in gal])
    g_cam = np.array([r["camera_id"] for _, r in gal])
    sim = q @ g.T

    aps, r1, r5, used, pos_s, neg_s = [], 0, 0, 0, [], []
    for i, (_, qr) in enumerate(qry):
        keep = g_cam != qr["camera_id"]
        rel = g_vid[keep] == qr["vehicle_id"]
        if not keep.any() or not rel.any():
            continue
        used += 1
        s = sim[i][keep]; order = np.argsort(-s); rs = rel[order]
        hits = np.flatnonzero(rs)
        aps.append(float(((np.arange(len(hits)) + 1) / (hits + 1)).mean()))
        r1 += int(rs[0]); r5 += int(rs[:5].any())
        (pos_s if rs[0] else neg_s).append(float(s[order[0]]))
    head.train()
    return {"mAP": round(float(np.mean(aps)) * 100, 2), "rank1": round(r1 / used * 100, 2),
            "rank5": round(r5 / used * 100, 2), "queries_used": used}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip_vitb16")
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--loss", default="arcface", choices=["arcface", "cosface"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--tri-weight", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.set_num_threads(2)

    spec = backbones.get(a.backbone)
    orig, flip, tr_rows, va_rows = load_train_features(a.backbone)
    vids = sorted({r["vehicle_id"] for _, r in tr_rows})
    vid2lab = {v: i for i, v in enumerate(vids)}
    print(f"[{a.tag}] train: {len(tr_rows)} кадров / {len(vids)} машин | val: {len(va_rows)} кадров", flush=True)

    head = ReidHead(spec.dim, a.hidden, a.out_dim, a.dropout)
    margin = MarginHead(a.out_dim, len(vids), a.loss, margin=a.margin)
    opt = AdamW(list(head.parameters()) + list(margin.parameters()), lr=a.lr, weight_decay=a.wd)

    sampler = PKSampler(tr_rows, a.P, a.K, a.seed)
    steps = max(1, len(sampler))
    rng = random.Random(a.seed)
    t0 = time.perf_counter()

    for ep in range(a.epochs):
        # warmup затем cosine
        if ep < a.warmup:
            lr = a.lr * (ep + 1) / a.warmup
        else:
            p = (ep - a.warmup) / max(1, a.epochs - a.warmup)
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * p))
        for gparam in opt.param_groups:
            gparam["lr"] = lr

        tot_c, tot_t, nb = 0.0, 0.0, 0
        for batch in sampler:
            # аугментация на уровне признаков: оригинал или отражение
            idxs = [tr_rows[p][0] for p in batch]
            src = np.where(np.array([rng.random() < 0.5 for _ in idxs]), 1, 0)
            x = np.where(src[:, None] == 1, flip[idxs], orig[idxs])
            x = torch.from_numpy(np.ascontiguousarray(x))
            y = torch.tensor([vid2lab[tr_rows[p][1]["vehicle_id"]] for p in batch])

            feat, feat_bn = head(x)
            l_cls = margin(feat_bn, y)
            l_tri = batch_hard_triplet(feat, y, 0.3)
            loss = l_cls + a.tri_weight * l_tri
            opt.zero_grad(); loss.backward(); opt.step()
            tot_c += l_cls.item(); tot_t += l_tri.item(); nb += 1

        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            print(f"  эпоха {ep+1}/{a.epochs}  lr {lr:.2e}  cls {tot_c/nb:.3f}  tri {tot_t/nb:.3f}", flush=True)

    train_s = time.perf_counter() - t0
    res = evaluate_val(head, orig, flip, va_rows)
    res.update(tag=a.tag, backbone=a.backbone, loss=a.loss, out_dim=a.out_dim, hidden=a.hidden, epochs=a.epochs,
               P=a.P, K=a.K, margin=a.margin, train_s=round(train_s, 1),
               head_params_mb=round(sum(p.numel() * 4 for p in head.parameters()) / 1024**2, 2))
    print(f"  val: mAP {res['mAP']}  Rank-1 {res['rank1']}  Rank-5 {res['rank5']}  "
          f"(обучение {train_s:.0f} с)", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    ck_args = dict(vars(a)); ck_args["in_dim"] = spec.dim
    torch.save({"head": head.state_dict(), "args": ck_args}, MODELS / f"head_{a.tag}.pt")
    out = HERE / "results"; out.mkdir(exist_ok=True)
    (out / f"head_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
