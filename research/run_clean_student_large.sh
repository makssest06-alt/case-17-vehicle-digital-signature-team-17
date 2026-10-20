#!/usr/bin/env bash
# Студент ViT-L/16@256 этой ветки (l150, кандидат по качеству): дистилляция из чистого
# ансамбля учителей тем же рецептом, что a10 (alpha=10, squash, контекст рамки 0.10),
# но backbone siglip_large и вход 256. Эпох 150: лестница 75 -> 150 ещё растёт
# (+2.2 пп expanded mAP), 300 — перелёт вниз (bm_improve/large_scale/notes.md).
# K=1 вместо 4 — из-за VRAM 4060; длина эпохи от K не зависит (173 шага), поэтому
# покрытие за эпоху меньше и эпох нужно больше, чем у a10.
#
# Порядок как в README §2.7: шаги 1–2 (учителя и clean_teacher_emb.npy) общие с a10;
# init-чекпоинт — prepare_public_init.py из timm vit_large_patch16_siglip_256
# (pretrained, squash, --input-size 256); дальше этот скрипт.
#
# Пути — с машины, где обучали; заменить на свои (переменные ниже).
set -euo pipefail
cd "$(dirname "$0")/.."

INPUTS=${INPUTS:-/home/l1zard/diskD/bm_improve/ema_experiment/inputs}
IMAGES=${IMAGES:-/home/l1zard/diskD/bm/dataset_extracted/images}
INIT=${INIT:-/home/l1zard/diskD/bm/improve_models/siglip_large_init.pt}
MODELS_DIR=${MODELS_DIR:-$PWD/weights_out}
TRAINING=research/finals_handoff/training

python "$TRAINING/train_distill.py" \
  --backbone siglip_large --init-from "$INIT" --input-size 256 \
  --teacher "$INPUTS/clean_teacher_emb.npy" --rows-csv "$INPUTS/clean_train_rows.csv" \
  --images-dir "$IMAGES" --models-dir "$MODELS_DIR" \
  --tag clean_student_large_150 --epochs 150 --save-epochs 150 \
  --alpha 10 --lr-head 3e-4 --lr-backbone 1e-5 --wd 5e-4 \
  --P 8 --K 1 --warmup 3 --workers 4 --amp --seed 7 --context 0.10 --subcenters 1

# Ожидаемый результат: $MODELS_DIR/clean_student_large_150_ep150.pt, 1 273 134 679 байт,
# sha256 6937d1163e239bcfd8ee76fe00da12a9d60becc89563b8fc5a475dc72df13d52
# (clean mAP@10 0.9005 / expanded 0.6467 на официальном evaluator).
