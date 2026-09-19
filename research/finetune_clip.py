#!/usr/bin/env python3
"""Полное дообучение бэкбона + головы на наших данных. Писалось под GPU.

Отличие от train_head.py: там бэкбон заморожен и признаки посчитаны заранее,
здесь градиент идёт через всю сеть, поэтому нужны картинки, аугментации и GPU.

На сервере без видеокарты запускать только smoke-тест:
    python finetune_clip.py --smoke --steps 2

Полный запуск на GPU — команда в README.
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

import backbones
from head_model import MarginHead, ReidHead, batch_hard_triplet

HERE = Path(__file__).parent
CROPS = Path("/opt/case17/data/crops/train")
MODELS = Path("/opt/case17/models")


# ---------------------------------------------------------------- аугментации

class NightAndGlare:
    """Аугментации под наши условия: ночь, блики фар, блюр движения, JPEG.

    Каждый седьмой кадр в датасете ночной, поэтому обычного ColorJitter мало —
    нужно явно моделировать провал яркости, засветы и потерю деталей.
    """

    def __init__(self, p: float = 0.5, size: int = 224):
        self.p, self.size = p, size

    def __call__(self, img: Image.Image) -> Image.Image:
        import PIL.ImageEnhance as E

        r = random.random
        if r() < self.p:  # ночь: темнее и менее контрастно
            img = E.Brightness(img).enhance(random.uniform(0.25, 0.7))
            img = E.Contrast(img).enhance(random.uniform(0.6, 1.0))
        if r() < 0.3:  # блик: локальное пересветление
            img = E.Brightness(img).enhance(random.uniform(1.2, 1.8))
        if r() < 0.3:  # смаз
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.6)))
        if r() < 0.3:  # компрессия камеры
            import io

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(25, 70))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if r() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.2)):
        self.p, self.scale = p, scale

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return t
        c, h, w = t.shape
        for _ in range(10):
            area = h * w * random.uniform(*self.scale)
            ar = random.uniform(0.3, 3.3)
            eh, ew = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
            if eh < h and ew < w:
                y, x = random.randint(0, h - eh), random.randint(0, w - ew)
                t[:, y : y + eh, x : x + ew] = torch.randn(c, eh, ew) * 0.2
                return t
        return t


# ------------------------------------------------------------------- датасет

class CropDataset(Dataset):
    def __init__(self, rows, tf, aug=None, train=True, images_dir=None, context=0.10):
        self.rows, self.tf, self.aug, self.train = rows, tf, aug, train
        self.images_dir = Path(images_dir) if images_dir else None
        self.context = context

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.images_dir is None:
            img = Image.open(CROPS / f"{r['image_id']}.jpg").convert("RGB")
        else:
            img = Image.open(self.images_dir / f"{r['image_id']}.jpg").convert("RGB")
            # The public dataset stores full frames. Crop the annotated vehicle
            # here so the training script works without a second 1GB crop cache.
            x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
            px, py = w * self.context, h * self.context
            left = max(0, int(round(x - px)))
            top = max(0, int(round(y - py)))
            right = min(img.width, int(round(x + w + px)))
            bottom = min(img.height, int(round(y + h + py)))
            img = img.crop((left, top, right, bottom))
        if self.train and self.aug is not None:
            img = self.aug(img)
        x = self.tf(img)
        if self.train:
            x = RandomErasing()(x)
        return x, r["_label"]


class PKBatchSampler(torch.utils.data.Sampler):
    def __init__(self, rows, P=16, K=4, seed=42):
        self.by = defaultdict(list)
        for i, r in enumerate(rows):
            self.by[r["vehicle_id"]].append(i)
        self.vids = sorted(self.by)
        self.P, self.K, self.rng = P, K, random.Random(seed)

    def __iter__(self):
        vids = self.vids[:]
        self.rng.shuffle(vids)
        for s in range(0, len(vids) - self.P + 1, self.P):
            batch = []
            for v in vids[s : s + self.P]:
                pool = self.by[v]
                batch += (self.rng.sample(pool, self.K) if len(pool) >= self.K
                          else [self.rng.choice(pool) for _ in range(self.K)])
            yield batch

    def __len__(self):
        return len(self.vids) // self.P


# --------------------------------------------------------------------- модель

class FullModel(nn.Module):
    def __init__(self, encoder, dim, hidden, out_dim, dropout):
        super().__init__()
        self.encoder = encoder
        self.head = ReidHead(dim, hidden, out_dim, dropout)

    def forward(self, x):
        return self.head(self.encoder(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip_vitb16")
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--loss", default="arcface", choices=["arcface", "cosface"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="mixed precision, только на GPU")
    ap.add_argument("--smoke", action="store_true", help="2 батча на CPU для проверки, что код живой")
    ap.add_argument("--steps", type=int, default=0, help="ограничить число шагов (для smoke)")
    ap.add_argument("--tag", default="finetune")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images-dir", default="", help="full-frame JPEG directory; crop is made from bbox on the fly")
    ap.add_argument("--train-csv", default="", help="CSV with image_id,x,y,w,h,vehicle_id,camera_id")
    ap.add_argument("--val-ids", default=str(HERE / "val_ids.txt"))
    ap.add_argument("--models-dir", default=str(MODELS))
    # Must match api/app/imaging.py and BBOX_PAD in the service configuration.
    # The supplied/precomputed crops use 10% of bbox width/height on every side.
    ap.add_argument("--context", type=float, default=0.10)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)
        if not a.smoke:
            print("ВНИМАНИЕ: GPU не найден. Полное дообучение на CPU займёт недели.")
            print("Запусти с --smoke для проверки кода или перенеси на GPU-машину.")
            return 1

    spec = backbones.get(a.backbone)
    if device.type == "cuda":
        print(f"CUDA_ACTIVE device={torch.cuda.get_device_name(0)} "
              f"vram_total={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GiB",
              flush=True)
    else:
        print("CUDA_INACTIVE device=cpu", flush=True)
    _, tf, encoder = backbones.build_encoder(spec)
    for p in encoder.parameters():
        p.requires_grad_(True)  # в отличие от train_head.py бэкбон размораживаем

    train_csv = Path(a.train_csv) if a.train_csv else Path("/opt/case17/data/crops/crops_train.csv")
    images_dir = Path(a.images_dir) if a.images_dir else None
    idx = list(csv.DictReader(train_csv.open(encoding="utf-8")))
    val_ids = {l.strip() for l in Path(a.val_ids).read_text().split() if l.strip()}
    rows = [r for r in idx if r["vehicle_id"] not in val_ids]
    vids = sorted({r["vehicle_id"] for r in rows})
    lab = {v: i for i, v in enumerate(vids)}
    for r in rows:
        r["_label"] = lab[r["vehicle_id"]]
    print(f"[{a.tag}] train {len(rows)} кадров / {len(vids)} машин, устройство {device}", flush=True)

    ds = CropDataset(rows, tf, NightAndGlare(size=spec.input_size), train=True,
                     images_dir=images_dir, context=a.context)
    sampler = PKBatchSampler(rows, a.P, a.K, a.seed)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=0 if a.smoke else a.workers,
                    pin_memory=device.type == "cuda")

    model = FullModel(encoder, spec.dim, a.hidden, a.out_dim, a.dropout).to(device)
    margin = MarginHead(a.out_dim, len(vids), a.loss).to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": a.lr_backbone},
        {"params": model.head.parameters(), "lr": a.lr_head},
        {"params": margin.parameters(), "lr": a.lr_head},
    ], weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device.type == "cuda")

    steps_total = a.steps if a.steps else a.epochs * max(1, len(sampler))
    step, t0 = 0, time.perf_counter()
    models_dir = Path(a.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(a.epochs):
        model.train()
        margin.train()
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            frac = step / max(1, steps_total)
            scale = (step + 1) / max(1, a.warmup * len(sampler)) if frac < 0.1 else 0.5 * (1 + math.cos(math.pi * frac))
            for g, base in zip(opt.param_groups, (a.lr_backbone, a.lr_head, a.lr_head)):
                g["lr"] = base * min(1.0, scale)

            with torch.amp.autocast("cuda", enabled=a.amp and device.type == "cuda"):
                feat, feat_bn = model(x)
                loss = margin(feat_bn, y) + batch_hard_triplet(feat, y, 0.3)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            step += 1
            if step % 20 == 0 or a.smoke:
                gpu_info = ""
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    gpu_info = (f"  gpu_mem={torch.cuda.memory_allocated() / 1024**3:.2f}/"
                                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GiB")
                print(f"  STEP {step}/{steps_total}  loss={loss.item():.3f}  "
                      f"sec_per_step={(time.perf_counter()-t0)/step:.2f}{gpu_info}", flush=True)
            if a.steps and step >= a.steps:
                break
        if a.steps and step >= a.steps:
            break
        torch.save({"model": model.state_dict(), "margin": margin.state_dict(), "args": vars(a)},
                   models_dir / f"{a.tag}_ep{ep+1}.pt")

    dt = time.perf_counter() - t0
    print(f"готово: {step} шагов за {dt:.1f} с ({dt/max(1,step):.2f} с/шаг)", flush=True)
    if a.smoke:
        print("SMOKE OK — код рабочий, переносим на GPU", flush=True)
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / "finetune_smoke.json").write_text(json.dumps(
            {"steps": step, "sec_per_step_cpu": round(dt / max(1, step), 2),
             "batch": a.P * a.K, "backbone": a.backbone, "device": str(device)},
            ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
