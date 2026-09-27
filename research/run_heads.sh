#!/bin/bash
cd /opt/case17/app/research
. .venv/bin/activate
set -x
python -u train_head.py --backbone clip_vitb16 --tag v1_arc512 --out-dim 512 --loss arcface --epochs 40
python -u train_head.py --backbone clip_vitb16 --tag v2_arc256 --out-dim 256 --loss arcface --epochs 40
python -u train_head.py --backbone clip_vitb16 --tag v3_cos512 --out-dim 512 --loss cosface --epochs 40 --margin 0.35
set +x
echo HEADS_DONE
