#!/usr/bin/env python3
"""Извлечение эмбеддингов обученным чекпоинтом на holdout-сплите.

Читает `val_split.csv` (551 query / 361 gallery), режет bbox с запасом 10%
(как production `BBOX_PAD=0.10`) и сохраняет две матрицы:
`emb_<tag>_qry.npy` и `emb_<tag>_gal.npy`.

Дальше их можно сравнивать любым скриптом замеров
(`official_protocol.py`, `final_stack.py`), не пересчитывая сеть.

    python extract_emb.py --ckpt models/reid_current.pt --tag geom_check --tf-mode timm
    python extract_emb.py --ckpt gpu_models/geom_squash_ep30.pt --tag squash_new --tf-mode squash --tta
"""
import argparse
import csv
import dataclasses
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import backbones
from head_model import ReidHead
from train_reid import build_transforms

HERE = Path(__file__).parent
DEFAULT_IMAGES = HERE / "dataset" / "images"


class FullModel(torch.nn.Module):
    def __init__(self, encoder, dim, hidden, out_dim, dropout):
        super().__init__()
        self.encoder = encoder
        self.head = ReidHead(dim, hidden, out_dim, dropout)

    def forward(self, x):
        return self.head(self.encoder(x))


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    spec_base = backbones.get(a.get("backbone", "siglip_base"))
    # чекпоинт мог быть обучен на нестандартном разрешении — берём его же
    size = a.get("size") or spec_base.input_size
    spec = dataclasses.replace(spec_base, input_size=size)
    # Веса целиком придут из чекпоинта, качать pretrained из интернета не нужно.
    _, _, encoder = backbones.build_encoder(spec_base, pretrained=False,
                                            input_size=size)
    model = FullModel(encoder, spec.dim, a.get("hidden", 1024),
                      a.get("out_dim", 512), a.get("dropout", 0.3))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"ВНИМАНИЕ: missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    return model, spec, a


def crop_image(path, row, context):
    img = Image.open(path).convert("RGB")
    x, y, w, h = (float(row[k]) for k in ("x", "y", "w", "h"))
    px, py = w * context, h * context
    left = max(0, int(round(x - px)))
    top = max(0, int(round(y - py)))
    right = min(img.width, int(round(x + w + px)))
    bottom = min(img.height, int(round(y + h + py)))
    return img.crop((left, top, right, bottom))


@torch.no_grad()
def embed_rows(model, tf, rows, images_dir, context, device, tta=False, bs=16):
    out = []
    for s in range(0, len(rows), bs):
        chunk = rows[s : s + bs]
        imgs = [crop_image(images_dir / f"{r['image_id']}.jpg", r, context) for r in chunk]
        x = torch.stack([tf(im) for im in imgs]).to(device)
        _, fb = model(x)
        f = fb.float()
        if tta:
            xf = torch.stack([tf(im.transpose(Image.FLIP_LEFT_RIGHT)) for im in imgs]).to(device)
            _, fb2 = model(xf)
            f = f + fb2.float()
        out.append(f.cpu().numpy())
    return np.concatenate(out, axis=0)


def resolve_tf_mode(override, checkpoint_args):
    """Use checkpoint preprocessing unless the caller explicitly overrides it."""
    checkpoint_mode = checkpoint_args.get("tf_mode")
    if override and checkpoint_mode and override != checkpoint_mode:
        raise ValueError(
            f"Requested tf-mode={override!r} conflicts with checkpoint tf_mode="
            f"{checkpoint_mode!r}"
        )
    return override or checkpoint_mode or "timm"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tf-mode", default=None, choices=["timm", "squash"],
                    help="override checkpoint preprocessing (default: checkpoint args, else timm)")
    ap.add_argument("--images-dir", default=str(DEFAULT_IMAGES))
    ap.add_argument("--val-csv", default=str(HERE / "val_split.csv"))
    ap.add_argument("--boxes-csv", default=str(HERE / "dataset" / "train.csv"),
                    help="CSV с bbox (image_id,x,y,w,h,...): в val_split.csv их нет")
    ap.add_argument("--context", type=float, default=0.10)
    ap.add_argument("--tta", action="store_true", help="усреднение с отражением, как в production")
    ap.add_argument("--bs", type=int, default=16, help="размер батча при извлечении")
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"устройство: {device}", flush=True)

    model, spec, ck_args = load_model(a.ckpt, device)
    tf_mode = resolve_tf_mode(a.tf_mode, ck_args)
    tf = build_transforms(spec, tf_mode, train=False)
    print(f"чекпоинт: {Path(a.ckpt).name}  backbone={spec.key}  "
          f"tf={tf_mode}  tta={a.tta}  context={a.context}", flush=True)

    rows = list(csv.DictReader(Path(a.val_csv).open(encoding="utf-8")))
    qry = [r for r in rows if r["role"] == "query"]
    gal = [r for r in rows if r["role"] == "gallery"]
    images_dir = Path(a.images_dir)

    boxes = {r["image_id"]: r for r in
             csv.DictReader(Path(a.boxes_csv).open(encoding="utf-8"))}
    for r in qry + gal:
        b = boxes.get(r["image_id"])
        if b is None:
            raise SystemExit(f"нет bbox для {r['image_id']} в {a.boxes_csv}")
        for k in ("x", "y", "w", "h"):
            r[k] = b[k]

    for name, subset in (("qry", qry), ("gal", gal)):
        emb = embed_rows(model, tf, subset, images_dir, a.context, device,
                         tta=a.tta, bs=a.bs)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        out = Path(a.out_dir) / f"emb_{a.tag}_{name}.npy"
        np.save(out, emb.astype("float32"))
        print(f"  {out.name}: {emb.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
