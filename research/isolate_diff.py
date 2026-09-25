#!/usr/bin/env python3
"""Где именно расходятся пути: в извлечении признаков или в сервисе."""
import csv, json, subprocess, sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path("/opt/case17/app/research")
FEAT = Path("/opt/case17/data/features")
CROPS = Path("/opt/case17/data/crops")
sys.path.insert(0, str(HERE))
from head_model import ReidHead
import backbones

iid = next(r["image_id"] for r in csv.DictReader((HERE / "val_split.csv").open()))
sizes = {r["image_id"]: (int(r["out_w"]), int(r["out_h"]))
         for r in csv.DictReader((CROPS / "crops_train.csv").open())}
w, h = sizes[iid]
img = Image.open(CROPS / "train" / f"{iid}.jpg").convert("RGB")
print(f"кадр {iid}, размер {img.size}, из csv {w}x{h}")

ck = torch.load("/opt/case17/models/reid_current.pt", map_location="cpu", weights_only=False)
ar = ck["args"]
head = ReidHead(ar.get("in_dim", 768), ar["hidden"], ar["out_dim"], ar["dropout"])
head.load_state_dict(ck["head"]); head.eval()
print(f"бэкбон из args чекпоинта: {ar.get('backbone')}")

# путь А: как считал extract_features (через реестр backbones)
spec = backbones.get(ar.get("backbone", "siglip_base"))
fwd, tf_a, model_a = backbones.build_encoder(spec)
# путь Б: как делает сервис
model_b = timm.create_model(spec.model_name, pretrained=True, num_classes=0).eval()
cfg = timm.data.resolve_data_config({}, model=model_b)
cfg["input_size"] = (3, 224, 224)
tf_b = timm.data.create_transform(**cfg, is_training=False)

print(f"\nтрансформ A: {str(tf_a)[:130]}")
print(f"трансформ B: {str(tf_b)[:130]}")

def emb(model, tf, im):
    x = tf(im).unsqueeze(0)
    with torch.inference_mode():
        a = head(model(x))[1]
        b = head(model(torch.flip(x, dims=[3])))[1]
    return F.normalize((a + b) / 2, p=2, dim=1)[0].numpy()

va = emb(model_a, tf_a, img)
vb = emb(model_b, tf_b, img)

# предрассчитанный признак
idx = list(csv.DictReader((FEAT / "train_index.csv").open()))
pos = {r["image_id"]: i for i, r in enumerate(idx)}
i = pos[iid]
o = np.load(FEAT / "train_siglip_base_orig.npy")[i:i+1]
fl = np.load(FEAT / "train_siglip_base_flip.npy")[i:i+1]
with torch.inference_mode():
    a = head(torch.from_numpy(o))[1].numpy(); b = head(torch.from_numpy(fl))[1].numpy()
vc = ((a + b) / 2)[0]; vc /= np.linalg.norm(vc)

out = subprocess.run(["curl", "-s", "-X", "POST", "http://localhost:8000/v1/embed",
                      "-F", f"image=@{CROPS/'train'/(iid+'.jpg')}", "-F", "x=0", "-F", "y=0",
                      "-F", f"w={w}", "-F", f"h={h}"], capture_output=True, text=True).stdout
vs_api = np.array(json.loads(out)["embedding"], dtype="float32")

print()
print(f"A (реестр backbones)   vs сервис : {va @ vs_api:.4f}")
print(f"B (как в сервисе)      vs сервис : {vb @ vs_api:.4f}")
print(f"C (предрассчитанные)   vs сервис : {vc @ vs_api:.4f}")
print(f"A vs C (реестр vs предрассчитанные): {va @ vc:.4f}")
print(f"A vs B (реестр vs сервисный путь)  : {va @ vb:.4f}")
