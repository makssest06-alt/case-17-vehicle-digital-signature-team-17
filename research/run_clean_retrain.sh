#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/l1zard/work/venv/bin/python"
TRAIN="/home/l1zard/diskD/bm_finals_handoff/research/finals_handoff/training/train_reid.py"
EXTRACT="/home/l1zard/diskD/bm_finals_handoff/research/finals_handoff/training/extract_emb.py"
EVAL="/home/l1zard/diskD/bm_finals_handoff/research/finals_handoff/training/evaluate_pair.py"
OFFICIAL="/home/l1zard/diskD/bm_finals_handoff/research/finals_handoff/training/official_evaluate.py"
IMAGES="/home/l1zard/diskD/bm/dataset_extracted/images"
ROWS="/tmp/opencode/clean_training/clean_train_rows.csv"
VAL="/tmp/opencode/clean_training/clean_holdout_split.csv"
BOXES="/home/l1zard/diskD/bm/dataset_extracted/train.csv"
MODELS="/home/l1zard/diskD/bm_improve/clean_models"
OUT="/tmp/opencode/clean_training"

export PYTHONPATH="/home/l1zard/diskD/bm_finals_handoff/research/finals_handoff/training"
export HF_HUB_OFFLINE=1
mkdir -p "$MODELS" "$OUT/eval"

train_teacher() {
  local tag="$1" backbone="$2" init="$3" seed="$4" tf_mode="$5" extra="$6"
  "$PYTHON" "$TRAIN" --backbone "$backbone" --init-from "$init" \
    --images-dir "$IMAGES" --train-csv "$ROWS" \
    --val-ids /home/l1zard/diskD/bm/research/val_ids.txt \
    --models-dir "$MODELS" --tag "$tag" --epochs 30 \
    --lr-head 3e-4 --lr-backbone 1e-5 --wd 5e-4 --P 8 --K 4 --warmup 3 \
    --workers 4 --amp --seed "$seed" --context 0.10 --save-epochs 28 $extra
  "$PYTHON" "$EXTRACT" --ckpt "$MODELS/${tag}_ep28.pt" --tag "$tag" \
    --images-dir "$IMAGES" --val-csv "$VAL" --boxes-csv "$BOXES" \
    --context 0.10 --bs 32 --out-dir "$OUT"
  "$PYTHON" "$EVAL" \
    --baseline-q "$OUT/emb_baseline_cleanholdout_qry.npy" \
    --baseline-g "$OUT/emb_baseline_cleanholdout_gal.npy" \
    --candidate-q "$OUT/emb_${tag}_qry.npy" \
    --candidate-g "$OUT/emb_${tag}_gal.npy" \
    --val-split "$VAL" --evaluator "$OFFICIAL" --output "$OUT/eval/$tag"
}

# Run one teacher at a time so the 4060 memory stays bounded.
train_teacher siglip1_clean siglip_base /tmp/opencode/siglip1_init.pt 42 squash "--geom-aug --cross-cam"
train_teacher siglip1_base_clean siglip_base /tmp/opencode/siglip1_init.pt 42 timm ""
train_teacher clip_clean clip_vitb16 /tmp/opencode/clip_vitb16_init.pt 7 squash "--geom-aug --cross-cam"
