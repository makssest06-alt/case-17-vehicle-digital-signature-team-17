#!/usr/bin/env python3
"""Train small-backbone students sequentially (distillation from the ensemble)."""
import subprocess
import sys
from pathlib import Path

PY = r"D:\venvs\case17-gpu\Scripts\python.exe"
HERE = Path(__file__).parent
MODELS = r"D:\bm\improve_models"
TEACHER = r"D:\bm\improve_models\teacher_emb.npy"
ROWS = r"D:\bm\improve_models\distill_train_rows.csv"
IMAGES = r"D:\bm\dataset_extracted\images"

# (backbone, init_from, tag)
JOBS = [
    ("mobilenetv3", "", "distill_mnv3"),
    ("dinov2_vits14", r"D:\bm\improve_models\dinov2_vits14_init.pt", "distill_dv2s"),
    ("efficientnet_b0", "", "distill_effb0"),
    ("resnet18", "", "distill_r18"),
]

for backbone, init, tag in JOBS:
    cmd = [PY, "train_distill.py", "--backbone", backbone,
           "--teacher", TEACHER, "--rows-csv", ROWS, "--images-dir", IMAGES,
           "--models-dir", MODELS, "--tag", tag, "--alpha", "20.0",
           "--epochs", "30", "--workers", "6", "--seed", "7", "--amp"]
    if init:
        cmd += ["--init-from", init]
    print(">>>", tag, flush=True)
    r = subprocess.run(cmd, cwd=str(HERE))
    print(f"<<< {tag} rc={r.returncode}", flush=True)
print("ALL DONE")
