#!/usr/bin/env bash
# Выбор профиля решения под железо стенда.
#
#   ./scripts/select_profile.sh          # определить автоматически
#   ./scripts/select_profile.sh cpu      # заставить CPU-профиль
#   ./scripts/select_profile.sh gpu      # заставить GPU-профиль
#   ./scripts/select_profile.sh --show   # показать текущий, ничего не менять
#
# Профили отличаются ОДНОЙ переменной EMBED_MODEL; остальное (EMBED_DIM,
# REID_WEIGHTS) подставляется автоматически, потому что размерность вектора
# жёстко связана с бэкбоном, а база хранит векторы фиксированной длины.
#
# ВАЖНО: EMBED_MODEL это ещё и аргумент сборки — веса бэкбона запекаются
# в образ. После смены профиля нужен `docker compose up -d --build`, скрипт
# об этом напомнит и предложит сделать сам.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env"

# --- описание профилей --------------------------------------------------
# cpu: soup150s2 (model soup ViT-L/16@256: l150 + s2, SigLIP Large + ReID-голова), 512D.
#      Это production-профиль, проверенный на clean holdout.
CPU_MODEL="reid:vit_large_patch16_siglip_256"
CPU_DIM=512
CPU_WEIGHTS="/opt/weights/soup150s2.pt"   # в образе (api/Dockerfile)

# gpu: SigLIP so400m. 1631 МБ весов (+6.5 МБ голова = 1638 МБ при лимите 2048),
#      примерно в 6 раз больше вычислений на кадр. На CPU это 0.35 кадра/с и
#      профиль бессмысленен; на видеокарте разница уходит в незаметное.
GPU_MODEL_HEAD="reid:vit_so400m_patch14_siglip_224.webli"
GPU_MODEL_ZS="timm:vit_so400m_patch14_siglip_224.webli"
GPU_DIM=512
GPU_WEIGHTS="/models/reid_so400m.pt"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m%s\033[0m\n' "$*"; }
warn() { printf '  \033[33m%s\033[0m\n' "$*"; }

current() {
  grep -E '^EMBED_MODEL=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- || echo "(не задан)"
}

detect() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo gpu
  else
    echo cpu
  fi
}

set_kv() {  # ключ значение
  if grep -qE "^$1=" "$ENV_FILE"; then
    sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"
  else
    echo "$1=$2" >> "$ENV_FILE"
  fi
}

MODE="${1:-auto}"

if [ "$MODE" = "--show" ]; then
  say "Текущий профиль"
  echo "  EMBED_MODEL = $(current)"
  echo "  железо      = $(detect)"
  exit 0
fi

[ -f "$ENV_FILE" ] || { cp "$ROOT/.env.example" "$ENV_FILE"; warn ".env создан из .env.example"; }

if [ "$MODE" = "auto" ]; then
  MODE="$(detect)"
  say "Железо определено автоматически: $MODE"
else
  say "Профиль задан вручную: $MODE"
fi

case "$MODE" in
  cpu)
    set_kv EMBED_MODEL "$CPU_MODEL"
    set_kv EMBED_DIM   "$CPU_DIM"
    set_kv REID_WEIGHTS "$CPU_WEIGHTS"
    set_kv BAKE_BACKBONE 0             # soup150s2 — полное дообучение, бэкбон с HF не нужен
    ok "CPU/GPU-safe production-профиль: soup150s2 (model soup ViT-L/16@256), 512D"
    ;;
  gpu)
    # Голова под so400m появляется только после обучения на GPU (kaggle/train_reid.ipynb).
    # Без неё so400m работает zero-shot и даёт mAP 60.80 — ХУЖЕ, чем
    # base с головой (65.16). Поэтому молча на zero-shot не переключаемся.
    HOST_MODELS="$(grep -E '^MODELS_HOST_DIR=' "$ENV_FILE" | cut -d= -f2- || echo ./weights)"
    if [ -f "$ROOT/${HOST_MODELS#./}/reid_so400m.pt" ] || [ -f "$HOST_MODELS/reid_so400m.pt" ]; then
      set_kv EMBED_MODEL "$GPU_MODEL_HEAD"
      set_kv EMBED_DIM   "$GPU_DIM"
      set_kv REID_WEIGHTS "$GPU_WEIGHTS"
      set_kv BAKE_BACKBONE 1           # head-only: предобученный so400m нужен в образе
      ok "GPU-профиль: SigLIP so400m + обученная голова, 512D"
    else
      warn "GPU-профиль запрошен, но обученной головы под so400m ещё нет"
      echo "  Без головы so400m даёт mAP 60.80 — это ХУЖЕ, чем soup150s2."
      echo "  Поэтому оставляю production-профиль soup150s2: он лучше по качеству и на видеокарте"
      echo "  тоже работает, просто быстрее."
      echo
      echo "  Чтобы включить so400m: обучить голову ноутбуком kaggle/train_reid.ipynb"
      echo "  (в ячейке CFG: backbone = \"vit_so400m_patch14_siglip_224.webli\"), затем"
      echo "    ./scripts/install_trained_weights.sh <файл>.pt --name reid_so400m.pt"
      set_kv EMBED_MODEL "$CPU_MODEL"
      set_kv EMBED_DIM   "$CPU_DIM"
      set_kv REID_WEIGHTS "$CPU_WEIGHTS"
    fi
    ;;
  *)
    echo "неизвестный профиль: $MODE (нужен cpu, gpu, auto или --show)" >&2
    exit 2
    ;;
esac

say "Применить"
echo "  EMBED_MODEL = $(current)"
echo
echo "  Веса бэкбона запекаются в образ, поэтому нужна пересборка:"
echo "    docker compose up -d --build"
echo
echo "  Если сменилась размерность вектора — база пересоздаётся:"
echo "    docker compose down -v && docker compose up -d --build"
