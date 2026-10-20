#!/usr/bin/env bash
# Пороги отказа, τ фона и правило почти-дублей под модель из окружения — одной командой
# (research/recalibrate_for_model.py: тот же код, что посчитал пресеты a10; протокол Q17, без данных теста).
#
#   EMBED_MODEL=reid:vit_large_patch16_siglip_256 REID_WEIGHTS=/путь/l75.pt INPUT_SIZE=256 \
#   DATA_DIR=/путь/к/данным DATASET_ZIP=/путь/dataset.zip ./research/recalibrate_for_model.sh l75 --source frames
#
# DATA_DIR — каталог с train.csv, test_*.csv (только для таблицы рамок) и, если есть, crops/train/ (так считали
# a10). Кадры — из DATASET_ZIP (--source frames; без crops/ — сами) и режутся кодом сервиса, как extract()
# (app.imaging.crop_frame, JPEG draft); crops/ нарезаны полным декодом — только с --env EXTRACT_JPEG_DRAFT=0.
# DATASET_ZIP нужен и для признака фона, если его нет в кэше. PYTHON — интерпретатор с зависимостями api/requirements.txt (по умолчанию python3).
# Итог: пресет для api/app/config.py, строки .env, таблица метрик на экране и
# research/honest_eval_results/recalibrate_<имя>.json.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-${TAG:-}}"
: "${TAG:?первый аргумент — имя модели в отчёте, например l75}"
: "${DATA_DIR:?DATA_DIR — каталог с train.csv (и crops/train/, если есть)}"
: "${REID_WEIGHTS:?REID_WEIGHTS — путь к весам модели}"
[ -f "$REID_WEIGHTS" ] || { echo "нет файла весов: $REID_WEIGHTS" >&2; exit 1; }

ARGS=(--data-dir "$DATA_DIR" --tag "$TAG")
[ -n "${DATASET_ZIP:-}" ] && ARGS+=(--zip "$DATASET_ZIP")
exec "${PYTHON:-python3}" research/recalibrate_for_model.py "${ARGS[@]}" "${@:2}"
