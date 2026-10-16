#!/usr/bin/env python3
"""Train remaining small-backbone students with local init checkpoints."""
import subprocess
from pathlib import Path

PY = r"D:\venvs\case17-gpu\Scripts\python.exe"
HERE = Path(__file__).parent
MODELS = r"D:\bm\improve_models"

JOBS = [
    ("mobilenetv3", r"D:\bm\improve_models\mobilenetv3_init.pt", "distill_mnv3"),
    ("efficientnet_b0", r"D:\bm\improve_models\efficientnet_b0_init.pt", "distill_effb0"),
    ("resnet18", r"D:\bm\improve_models\resnet18_init.pt", "distill_r18"),
]

for backbone, init, tag in JOBS:
    cmd = [PY, "train_distill.py", "--backbone", backbone,
           "--init-from", init,
           "--teacher", r"D:\bm\improve_models\teacher_emb.npy",
           "--rows-csv", r"D:\bm\improve_models\distill_train_rows.csv",
           "--images-dir", r"D:\bm\dataset_extracted\images",
           "--models-dir", MODELS, "--tag", tag, "--alpha", "20.0",
           "--epochs", "30", "--workers", "6", "--seed", "7", "--amp"]
    print(">>>", tag, flush=True)
    r = subprocess.run(cmd, cwd=str(HERE))
    print(f"<<< {tag} rc={r.returncode}", flush=True)
print("ALL DONE")
