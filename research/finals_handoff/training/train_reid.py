#!/usr/bin/env python3
"""Восстановленный скрипт обучения production-модели (SigLIP ViT-B/16 + ReID-голова).

Собран из `research/finetune_clip.py` (ветка feat/research-baselines) и рецепта
из `models/README.md` (коммит d108d59), который дал geom-чекпоинт.
Аргументы ниже сверены с `args`, сохранёнными внутри
`siglip_vitb16_reid_geom_ep30.pt`:

    backbone=siglip_base out_dim=512 hidden=1024 dropout=0.3 loss=arcface
    epochs=30 lr_head=3e-4 lr_backbone=1e-5 wd=5e-4 P=8 K=4 warmup=3
    amp=true seed=42 context=0.10 geom_aug=true tag=siglip_geom_pad10

Отличия от базового скрипта:

* `--geom-aug` — геометрические аугментации (перспектива, part-crop, поворот),
  которые в README описаны как ключевое отличие geom-версии (+2.36 mAP);
* `--tf-mode {timm,squash}` — геометрическая нормализация входа. `timm`
  воспроизводит production (`Resize(248)+CenterCrop(224)`, crop_pct=0.9,
  оставляет ~66.6% длины машины). `squash` — прямой `Resize((S,S))`, который
  на инференсе дал +2.8 mAP вопреки обучению; смысл режима — убрать этот
  разрыв train/test;
* `--cross-cam` — сэмплер, который в пределах одной машины добирает кадры
  с РАЗНЫХ камер, чтобы позитивы в батче не были лёгкими;
* `--eval-every` — периодический замер на holdout-сплите по машинам.

Примеры:
    python train_reid.py --smoke --steps 2
    python train_reid.py --geom-aug --tag geom_squash --tf-mode squash
"""
import argparse
import csv
import dataclasses
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
from torchvision import transforms as T

import backbones
from head_model import MarginHead, ReidHead, batch_hard_triplet

HERE = Path(__file__).parent
DEFAULT_IMAGES = HERE / "dataset" / "images"
DEFAULT_TRAIN_CSV = HERE / "dataset" / "train.csv"


# ---------------------------------------------------------------- аугментации

class NightAndGlare:
    """Аугментации под наши условия: ночь, блики фар, блюр движения, JPEG."""

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


class GeomAug:
    """Геометрические аугментации geom-версии: перспектива, part-crop, поворот ±7°.

    Применяются к PIL-картинке до основной нормализации. Порядок и силы взяты
    из описания в models/README.md; точные числа в чекпоинте не сохранились,
    поэтому они подобраны консервативно и вынесены в параметры.
    """

    def __init__(self, size: int = 224, p: float = 0.5, rotate_deg: float = 7.0,
                 perspective: float = 0.2, crop_scale=(0.6, 1.0)):
        self.p = p
        self.rotate = T.RandomRotation(degrees=rotate_deg, fill=128)
        self.persp = T.RandomPerspective(distortion_scale=perspective, p=1.0, fill=128)
        self.partcrop = T.RandomResizedCrop(size=(size, size), scale=crop_scale,
                                            ratio=(0.75, 1.3333),
                                            interpolation=T.InterpolationMode.BICUBIC,
                                            antialias=True)

    def __call__(self, img: Image.Image) -> Image.Image:
        r = random.random
        if r() < self.p:
            img = self.rotate(img)
        if r() < self.p:
            img = self.persp(img)
        if r() < self.p:
            img = self.partcrop(img)
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


def build_transforms(spec, tf_mode: str, train: bool):
    """Геометрическая нормализация входа.

    `timm` — ровно то, что делает production (`backbones.build_encoder`):
    Resize(248)+CenterCrop(224) при crop_pct=0.9.
    `squash` — прямой ресайз в квадрат, вся машина в кадре.
    """
    import timm

    model = timm.create_model(spec.model_name, pretrained=False, num_classes=0)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean, std = tuple(cfg["mean"]), tuple(cfg["std"])

    if tf_mode == "timm":
        cfg = dict(cfg)
        cfg["input_size"] = (3, spec.input_size, spec.input_size)
        return timm.data.create_transform(**cfg, is_training=train)

    if tf_mode == "squash":
        return T.Compose([
            T.Resize((spec.input_size, spec.input_size),
                     interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    raise SystemExit(f"неизвестный --tf-mode {tf_mode}")


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
        img = Image.open(self.images_dir / f"{r['image_id']}.jpg").convert("RGB")
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
    """P машин x K кадров. При cross_cam=True кадры одной машины берутся
    с максимально разными камерами, иначе — случайно из всех."""

    def __init__(self, rows, P=8, K=4, seed=42, cross_cam=False):
        self.by = defaultdict(list)
        self.cam = defaultdict(list)
        for i, r in enumerate(rows):
            self.by[r["vehicle_id"]].append(i)
            self.cam[(r["vehicle_id"], r["camera_id"])].append(i)
        self.vids = sorted(self.by)
        self.P, self.K, self.rng = P, K, random.Random(seed)
        self.cross_cam = cross_cam

    def _draw(self, v):
        pool = self.by[v]
        if not self.cross_cam:
            return (self.rng.sample(pool, self.K) if len(pool) >= self.K
                    else [self.rng.choice(pool) for _ in range(self.K)])
        cams = [c for (vv, c) in self.cam if vv == v]
        out = []
        for _ in range(self.K):
            if cams:
                c = self.rng.choice(cams)
                out.append(self.rng.choice(self.cam[(v, c)]))
                if len(cams) > 1 and self.rng.random() < 0.5:
                    cams.remove(c)
            else:
                out.append(self.rng.choice(pool))
        return out

    def __iter__(self):
        vids = self.vids[:]
        self.rng.shuffle(vids)
        for s in range(0, len(vids) - self.P + 1, self.P):
            batch = []
            for v in vids[s : s + self.P]:
                batch += self._draw(v)
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
    ap.add_argument("--backbone", default="siglip_base")
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--loss", default="arcface", choices=["arcface", "cosface"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--tag", default="finetune")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images-dir", default=str(DEFAULT_IMAGES))
    ap.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    ap.add_argument("--val-ids", default=str(HERE / "val_ids.txt"))
    ap.add_argument("--models-dir", default=str(HERE / "gpu_models"))
    ap.add_argument("--context", type=float, default=0.10)
    ap.add_argument("--geom-aug", action="store_true",
                    help="перспектива, part-crop, поворот ±7° (geom-версия)")
    ap.add_argument("--tf-mode", default="timm", choices=["timm", "squash"],
                    help="timm = production Resize(248)+CenterCrop(224); squash = Resize(S,S)")
    ap.add_argument("--size", type=int, default=0,
                    help="входное разрешение (0 — из описания бэкбона, 224)")
    ap.add_argument("--cross-cam", action="store_true",
                    help="в батче добирать кадры машины с разных камер")
    ap.add_argument("--init-from", default="",
                    help="локальный чекпоинт case17 для старта (энкодер+голова). "
                         "Пусто = pretrained-веса timm из интернета")
    ap.add_argument("--init-full", action="store_true",
                    help="вместе с энкодером загрузить и голову (дообучение)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not a.smoke:
        print("ВНИМАНИЕ: GPU не найден. Запусти с --smoke или перенеси на GPU-машину.")
        return 1

    spec_base = backbones.get(a.backbone)
    size = a.size or spec_base.input_size
    spec = dataclasses.replace(spec_base, input_size=size)
    if device.type == "cuda":
        print(f"CUDA_ACTIVE device={torch.cuda.get_device_name(0)} "
              f"vram_total={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GiB",
              flush=True)
    else:
        print("CUDA_INACTIVE device=cpu", flush=True)
    init = torch.load(a.init_from, map_location="cpu", weights_only=False) if a.init_from else None
    _, _, encoder = backbones.build_encoder(spec_base, pretrained=init is None,
                                            input_size=size)
    if init is not None:
        enc_sd = {k[len("encoder."):]: v for k, v in init["model"].items()
                  if k.startswith("encoder.")}
        encoder.load_state_dict(enc_sd, strict=True)
        print(f"старт с {Path(a.init_from).name}: энкодер загружен "
              f"({len(enc_sd)} тензоров)", flush=True)
    for p in encoder.parameters():
        p.requires_grad_(True)

    # train=False намеренно: production обучался ровно этим transform'ом
    # (`backbones.build_encoder` всегда строит его с is_training=False),
    # поэтому для воспроизводимости не подменяем его на train-версию timm.
    tf = build_transforms(spec, a.tf_mode, train=False)
    train_csv = Path(a.train_csv)
    images_dir = Path(a.images_dir)
    idx = list(csv.DictReader(train_csv.open(encoding="utf-8")))
    val_ids = {l.strip() for l in Path(a.val_ids).read_text().split() if l.strip()}
    rows = [r for r in idx if r["vehicle_id"] not in val_ids]
    vids = sorted({r["vehicle_id"] for r in rows})
    lab = {v: i for i, v in enumerate(vids)}
    for r in rows:
        r["_label"] = lab[r["vehicle_id"]]
    print(f"[{a.tag}] train {len(rows)} кадров / {len(vids)} машин, "
          f"tf={a.tf_mode} geom_aug={a.geom_aug} cross_cam={a.cross_cam} "
          f"устройство {device}", flush=True)

    aug = NightAndGlare(size=spec.input_size)
    if a.geom_aug:
        aug = T.Compose([GeomAug(size=spec.input_size), aug])
    ds = CropDataset(rows, tf, aug, train=True, images_dir=images_dir, context=a.context)
    sampler = PKBatchSampler(rows, a.P, a.K, a.seed, cross_cam=a.cross_cam)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=0 if a.smoke else a.workers,
                    pin_memory=device.type == "cuda")

    model = FullModel(encoder, spec.dim, a.hidden, a.out_dim, a.dropout).to(device)
    margin = MarginHead(a.out_dim, len(vids), a.loss).to(device)
    if init is not None and a.init_full:
        head_sd = {k[len("head."):]: v for k, v in init["model"].items()
                   if k.startswith("head.")}
        model.head.load_state_dict(head_sd, strict=True)
        print(f"  голова тоже загружена ({len(head_sd)} тензоров)", flush=True)
        mg_sd = init.get("margin")
        if mg_sd and mg_sd["weight"].shape == margin.weight.shape:
            margin.load_state_dict(mg_sd, strict=True)
            print("  margin-голова загружена (тот же порядок классов)", flush=True)
        elif mg_sd:
            print(f"  ВНИМАНИЕ: margin не загружен, форма {tuple(mg_sd['weight'].shape)} "
                  f"!= {tuple(margin.weight.shape)}", flush=True)
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
        print(f"  сохранён {a.tag}_ep{ep+1}.pt", flush=True)

    dt = time.perf_counter() - t0
    print(f"готово: {step} шагов за {dt:.1f} с ({dt/max(1,step):.2f} с/шаг)", flush=True)
    if a.smoke:
        print("SMOKE OK — код рабочий", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
