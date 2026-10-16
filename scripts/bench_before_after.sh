#!/usr/bin/env bash
# Замеры ДО/ПОСЛЕ на одинаковой методике.
#
#   ДО    — ветка feat/research-baselines: torch + Pillow + поиск в pgvector
#   ПОСЛЕ — ветка feat/fast-inference: ONNX Runtime + TurboJPEG + in-memory индекс
#
# Обе ветки поднимаются как один и тот же compose-проект case17, поэтому
# база и импортированная галерея переиспользуются — меняется только код API.
# Харнесс замера (benchmark.py, profile_pipeline.py) берётся ОДИН И ТОТ ЖЕ
# и копируется в контейнер: сравниваем реализации, а не измерители.
#
#   ./scripts/bench_before_after.sh all      # весь цикл
#   ./scripts/bench_before_after.sh before   # только ДО
#   ./scripts/bench_before_after.sh export   # экспорт ONNX из текущего стека
#   ./scripts/bench_before_after.sh after    # только ПОСЛЕ
#   ./scripts/bench_before_after.sh report   # пересобрать таблицу из json
set -euo pipefail

BEFORE_DIR=${BEFORE_DIR:-/opt/case17/app}
AFTER_DIR=${AFTER_DIR:-/opt/case17/app-fast}
MODELS_DIR=${MODELS_DIR:-/opt/case17/models}
HARNESS=${HARNESS:-$AFTER_DIR/api/scripts}
OUT=${OUT:-$AFTER_DIR/research/results/bench}
QUICK=${QUICK:-1}
N_LAT=${N_LAT:-30}
BATCHES=${BATCHES:-1,8,16}
N_PROFILE=${N_PROFILE:-30}

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

wait_ready() {
  local i
  for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/ready >/dev/null 2>&1; then
      echo "api готов (${i}0 с)"; return 0
    fi
    sleep 10
  done
  echo "api не поднялся за 20 минут" >&2
  docker logs --tail 40 case17-api >&2 || true
  return 1
}

push_harness() {
  docker cp "$HARNESS/benchmark.py"        case17-api:/srv/scripts/benchmark.py
  docker cp "$HARNESS/profile_pipeline.py" case17-api:/srv/scripts/profile_pipeline.py
}

measure() { # $1 = метка (before|after)
  local label=$1 dir=$2
  push_harness
  local qflag=""
  [ "$QUICK" = "1" ] && qflag="--quick"
  docker compose exec -T api python scripts/benchmark.py $qflag \
    --n-latency "$N_LAT" --batches "$BATCHES" \
    --out "/artifacts/benchmark_$label.json"
  docker compose exec -T api python scripts/profile_pipeline.py \
    --n "$N_PROFILE" --label "$label" --out "/artifacts/profile_$label.json"
  mkdir -p "$OUT"
  cp "$dir/artifacts/benchmark_$label.json" "$dir/artifacts/profile_$label.json" "$OUT/"
  echo "сохранено в $OUT"
}

stage_before() {
  log "ДО: $BEFORE_DIR"
  cd "$BEFORE_DIR"
  docker compose up -d --build
  wait_ready
  measure before "$BEFORE_DIR"
}

stage_export() {
  log "экспорт ONNX (из текущего поднятого стека)"
  # /models смонтирован ro, поэтому пишем в rw-том /artifacts и копируем сами
  docker cp "$HARNESS/export_onnx.py" case17-api:/srv/scripts/export_onnx.py
  cd "$AFTER_DIR"
  docker compose exec -T api python scripts/export_onnx.py --out /artifacts/reid.onnx
  cp "$AFTER_DIR/artifacts/reid.onnx" "$MODELS_DIR/reid.onnx"
  ls -lh "$MODELS_DIR/reid.onnx"
}

stage_after() {
  log "ПОСЛЕ: $AFTER_DIR"
  cd "$AFTER_DIR"
  # тот же .env, что у ДО, чтобы отличался только код и формат модели
  cp "$BEFORE_DIR/.env" "$AFTER_DIR/.env"
  docker compose up -d --build
  wait_ready
  stage_export
  sed -i 's#^EMBED_MODEL=.*#EMBED_MODEL=onnx:/models/reid.onnx#' "$AFTER_DIR/.env"
  docker compose up -d --force-recreate api
  wait_ready
  measure after "$AFTER_DIR"
}

stage_report() {
  log "сводная таблица"
  python3 "$AFTER_DIR/research/compare_speed.py" --dir "$OUT"
}

case "${1:-all}" in
  before) stage_before ;;
  export) stage_export ;;
  after)  stage_after ;;
  report) stage_report ;;
  all)    stage_before; stage_after; stage_report ;;
  *) echo "использование: $0 [all|before|export|after|report]" >&2; exit 2 ;;
esac
