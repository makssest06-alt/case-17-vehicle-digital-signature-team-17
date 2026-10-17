#!/usr/bin/env bash
# FPS и латентность p50/p95 конвейера сервиса на GPU: постобработка выключена и
# включена (DBA k=1 + штраф за ту же камеру), батч 1 и 8. Одна команда для стенда
# с RTX 4060, из корня репозитория на ветке с api/app/pp, при поднятом стеке:
#
#     docker compose up -d --build api      # если образ собран до этой ветки
#     scripts/bench_fps.sh
#
# Галерея сервиса не трогается: замер строит свою во временной схеме bench_pp той же
# базы и удаляет её. Конвейер — тот же код, что /v1/search, без HTTP
# (api/scripts/benchmark_pp.py). Отчёт — artifacts/bench_fps_<дата>.json и таблица
# в консоли.
#
# Переменные (по умолчанию — раскладка датасета в DATA_DIR контейнера):
#   BENCH_IMAGES         папка полных кадров <image_id>.jpg   (/data/images)
#   BENCH_QUERY          CSV запросов image_id,x,y,w,h        (/data/test_query.csv)
#   BENCH_GALLERY        CSV кадров галереи                   (/data/train.csv)
#   BENCH_GALLERY_LIMIT  кадров в галерее                     (10847)
#   BENCH_REFUSAL        оценка отказа в режиме on: raw|z100  (raw)
#   BENCH_ITERATIONS     батчей на точку                      (100)
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGES="${BENCH_IMAGES:-/data/images}"
QUERY="${BENCH_QUERY:-/data/test_query.csv}"
GALLERY="${BENCH_GALLERY:-/data/train.csv}"
LIMIT="${BENCH_GALLERY_LIMIT:-10847}"
REFUSAL="${BENCH_REFUSAL:-raw}"
ITER="${BENCH_ITERATIONS:-100}"
NAME="bench_fps_$(date +%Y%m%d_%H%M%S).json"

if ! docker compose exec -T api test -f app/pp/state.py; then
  echo "в контейнере api нет app/pp — образ старый: docker compose up -d --build api" >&2
  exit 1
fi
for p in "$IMAGES" "$QUERY" "$GALLERY"; do
  if ! docker compose exec -T api test -e "$p"; then
    echo "в контейнере api нет $p — задайте BENCH_IMAGES / BENCH_QUERY / BENCH_GALLERY" >&2
    exit 1
  fi
done

docker compose exec -T api python scripts/benchmark_pp.py \
  --images-dir "$IMAGES" --query-csv "$QUERY" --gallery-csv "$GALLERY" --gallery-limit "$LIMIT" \
  --batches 1,8 --modes off,on --refusal-score "$REFUSAL" --iterations "$ITER" \
  --require-cuda --json "/artifacts/$NAME"
echo "отчёт: artifacts/$NAME"
