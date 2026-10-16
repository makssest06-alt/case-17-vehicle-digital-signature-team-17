#!/usr/bin/env python3
"""Extract NO-TTA embeddings for all 4 models (to quantify the TTA tradeoff)."""
import subprocess
import sys
from pathlib import Path

PY = r"D:\venvs\case17-gpu\Scripts\python.exe"
HERE = Path(__file__).parent
OUT = r"D:\bm\research\ensemble"
COMMON = ["--val-csv", r"D:\bm\research\ensemble\val_split.csv",
          "--boxes-csv", r"D:\bm\dataset_extracted\train.csv",
          "--images-dir", r"D:\bm\dataset_extracted\images", "--out-dir", OUT]

JOBS = [
    (r"D:\bm_improve\models\siglip2_reid_scratch30_ep28.pt", "siglip2_ep28_noTTA", "squash"),
    (r"D:\bm_improve\models\siglip_reid_scratch30_ep26.pt", "sq30_ep26_noTTA", "squash"),
    (r"D:\bm_improve\models\siglip_vitb16_reid_ep29.pt", "ep29_noTTA", "timm"),
    (r"D:\bm\improve_models\clip_reid_scratch30_ep28.pt", "clip_ep28_noTTA", "squash"),
]

for ckpt, tag, tfm in JOBS:
    cmd = [PY, "extract_emb.py", "--ckpt", ckpt, "--tag", tag, "--tf-mode", tfm] + COMMON
    print(">>>", tag, flush=True)
    subprocess.run(cmd, check=True)
print("done")
