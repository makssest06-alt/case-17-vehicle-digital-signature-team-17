#!/bin/bash
cd /opt/case17/app/research
. .venv/bin/activate
python -u train_head.py --backbone siglip_base --tag s1_arc512 --out-dim 512 --loss arcface --epochs 40
python -u train_head.py --backbone siglip_base --tag s2_arc256 --out-dim 256 --loss arcface --epochs 40
python -u train_head.py --backbone siglip_base --tag s3_cos512 --out-dim 512 --loss cosface --epochs 40 --margin 0.35
echo SIGHEADS_DONE
