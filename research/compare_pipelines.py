#!/usr/bin/env python3
"""Сверка двух путей получения вектора: офлайн-исследование против сервиса.

Все наши метрики посчитаны офлайн по предрассчитанным признакам, а порог
отказа работает в сервисе. Если пути расходятся, метрики описывают не то,
что реально отвечает пользователю.
"""
import csv, json, subprocess, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path("/opt/case17/app/research")
FEAT = Path("/opt/case17/data/features")
CROPS = Path("/opt/case17/data/crops")
sys.path.insert(0, str(HERE))
from head_model import ReidHead

idx = list(csv.DictReader((FEAT / "train_index.csv").open()))
pos = {r["image_id"]: i for i, r in enumerate(idx)}
orig = np.load(FEAT / "train_siglip_base_orig.npy")
flip = np.load(FEAT / "train_siglip_base_flip.npy")
ck = torch.load("/opt/case17/models/reid_current.pt", map_location="cpu", weights_only=False)
ar = ck["args"]
head = ReidHead(ar.get("in_dim", orig.shape[1]), ar["hidden"], ar["out_dim"], ar["dropout"])
head.load_state_dict(ck["head"]); head.eval()

sizes = {r["image_id"]: (r["out_w"], r["out_h"])
         for r in csv.DictReader((CROPS / "crops_train.csv").open())}
vs = list(csv.DictReader((HERE / "val_split.csv").open()))
sample = [r["image_id"] for r in vs[:6]]

print(f"{'image_id':<20}{'косинус офлайн/сервис':>24}{'норма офлайн':>14}")
for iid in sample:
    i = pos[iid]
    with torch.inference_mode():
        a = head(torch.from_numpy(orig[i:i+1]))[1].numpy()
        b = head(torch.from_numpy(flip[i:i+1]))[1].numpy()
    v = (a + b) / 2
    v = (v / np.linalg.norm(v, axis=1, keepdims=True))[0]

    w, h = sizes[iid]
    f = CROPS / "train" / f"{iid}.jpg"
    out = subprocess.run(["curl", "-s", "-X", "POST", "http://localhost:8000/v1/embed",
                          "-F", f"image=@{f}", "-F", "x=0", "-F", "y=0",
                          "-F", f"w={w}", "-F", f"h={h}"],
                         capture_output=True, text=True).stdout
    d = json.loads(out)
    s = np.array(d["embedding"], dtype="float32")
    print(f"{iid[:18]:<20}{float(v @ s):>24.4f}{float(np.linalg.norm(v)):>14.4f}")
