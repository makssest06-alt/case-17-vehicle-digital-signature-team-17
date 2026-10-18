#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/l1zard/work/venv/bin/python"
TRAIN="/home/l1zard/diskD/bm_finals_handoff/research/finals_handoff/training/train_distill.py"
COMMON=(
  --backbone siglip2_base
  --init-from /home/l1zard/diskD/bm/improve_models/siglip2_init.pt
  --teacher /tmp/opencode/clean_training/clean_teacher_emb.npy
  --rows-csv /tmp/opencode/clean_training/clean_train_rows.csv
  --images-dir /home/l1zard/diskD/bm/dataset_extracted/images
  --models-dir /home/l1zard/diskD/bm_improve/clean_student
  --epochs 30 --lr-head 3e-4 --lr-backbone 1e-5 --wd 5e-4
  --P 8 --K 4 --warmup 3 --workers 4 --amp --seed 7
  --context 0.10 --subcenters 1 --save-epochs 28
)

export PYTHONPATH=/home/l1zard/diskD/bm_finals_handoff/research/finals_handoff/training
export HF_HUB_OFFLINE=1

for alpha in 2 6 50; do
  "$PYTHON" "$TRAIN" "${COMMON[@]}" \
    --tag "clean_student_a${alpha}" --alpha "$alpha" \
    > "/home/l1zard/diskD/bm_improve/clean_logs/clean_student_a${alpha}.log" 2>&1
done
