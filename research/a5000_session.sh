#!/usr/bin/env bash
# Один заход на арендованной машине с A5000: честное сравнение a10 и L75 так, как будут мерить организаторы.
#   1. окружение: nvidia-smi (GPU, драйвер, CUDA), модель и число ядер CPU, RAM, Docker; torch и CUDA в образе;
#   2. образ — как соберёт жюри (docker compose build api), или готовый, если он уже есть;
#   3. видна ли видеокарта из контейнера (torch.cuda.is_available()) — нет: остановка с понятной ошибкой;
#   4. официальный api/scripts/bench_extract.py в контейнере для a10 и L75, EXTRACT_THREADS 4, 8, 16;
#   5. для L75 — research/recalibrate_for_model.sh (порог отказа, τ фона, почти-дубли; протокол Q17);
#   6. одна таблица (модель, latency_b1, лучший FPS, VRAM, баллы скорости Q34, порог и формула отказа) —
#      <OUT_DIR>/session.md и session.json (research/a5000_report.py).
#
#   L75_WEIGHTS=/путь/l75.pt DATA_DIR=/путь/к/данным DATASET_ZIP=/путь/dataset.zip ./research/a5000_session.sh
#
# Пути — из окружения или аргументов (--data-dir, --zip, --l75-weights, --a10-weights, --image, --out-dir,
# --cache-dir, --threads "4 8 16", --build):
#   DATA_DIR       test_query.csv и images/ (замер); train.csv, test_*.csv и crops/train/, если есть (пересчёт)
#   DATASET_ZIP    dataset.zip — кадры и признак фона для пересчёта, если нет crops/train/ и кэша фона
#   L75_WEIGHTS    веса L75; L75_EMBED_MODEL (reid:vit_large_patch16_siglip_256), L75_INPUT_SIZE (256),
#                  L75_EMBED_DIM и L75_TF_MODE — если отличаются (README, раздел 4.3)
#   A10_WEIGHTS    веса a10 — нужны, только если weights/clean_student_a10_ep28.pt — указатель LFS и нет
#                  WEIGHTS_URL: файл кладётся в weights/ после сверки sha256 с WEIGHTS_SHA256 из api/Dockerfile
#   WEIGHTS_URL    как у жюри: ссылка на веса a10 для сборки без git-lfs
#   IMAGE          образ сервиса (case17-api:local — как после docker compose build api); BUILD=1 — собрать заново
#   OUT_DIR        research/a5000_session/<дата_время> (в git не входит); CACHE_DIR — кэш эмбеддингов
#   SKIP_L75=1, SKIP_RECALIBRATE=1 — пропустить шаги
# Сеть нужна только сборке (базовый образ, pip); замеры и пересчёт идут с --network none.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --zip) DATASET_ZIP="$2"; shift 2 ;;
    --l75-weights) L75_WEIGHTS="$2"; shift 2 ;;
    --a10-weights) A10_WEIGHTS="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --build) BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "неизвестный аргумент: $1 (--help)" >&2; exit 2 ;;
  esac
done

IMAGE="${IMAGE:-case17-api:local}"
RIMAGE="${RESEARCH_IMAGE:-case17-api-research:local}"
THREADS="${THREADS:-4 8 16}"
L75_EMBED_MODEL="${L75_EMBED_MODEL:-reid:vit_large_patch16_siglip_256}"
L75_INPUT_SIZE="${L75_INPUT_SIZE:-256}"
OUT_DIR="${OUT_DIR:-$REPO/research/a5000_session/$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR="${CACHE_DIR:-$HOME/lct-case17/cache/honest_eval}"
MODELS="a10 l75"; [ "${SKIP_L75:-0}" = 1 ] && MODELS="a10"
RECAL=1; { [ "${SKIP_L75:-0}" = 1 ] || [ "${SKIP_RECALIBRATE:-0}" = 1 ]; } && RECAL=0
A10_FILE="weights/clean_student_a10_ep28.pt"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
exec > >(tee -a "$OUT_DIR/session.log") 2>&1
rm -f "$OUT_DIR/status.txt" "$OUT_DIR/bench_done"

say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
to() {  # to СЕКУНДЫ команда… — ограничение по времени, если есть timeout(1)
  local s="$1"; shift
  if command -v timeout >/dev/null 2>&1; then timeout "$s" "$@"; else "$@"; fi
}
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
T_STEP=$(date +%s)
mark() { local now; now=$(date +%s); echo "$1=$((now - T_STEP))" >> "$OUT_DIR/timings.txt"; T_STEP=$now; }
finish() {  # finish КОД СТРОКА_СТАТУСА [ПОДРОБНО] — статус и отчёт
  local rc="$1"; shift
  printf '%s\n' "$@" > "$OUT_DIR/status.txt"
  python3 research/a5000_report.py "$OUT_DIR" > /dev/null || say "ОШИБКА: отчёт не собран (research/a5000_report.py)"
  say "отчёт: $OUT_DIR/session.md, $OUT_DIR/session.json"
  exit "$rc"
}
stop() { local step="$1"; shift; say "ОШИБКА (п. $step): $*"; finish 3 "остановлено на п. $step" "$*"; }

# ---------------------------------------------------------------- 0. входные данные
say "заход: образ $IMAGE, модели $MODELS, EXTRACT_THREADS $THREADS, отчёт в $OUT_DIR"
: > "$OUT_DIR/timings.txt"
{ echo "models=$MODELS"; echo "threads=$THREADS"; echo "image=$IMAGE"; echo "l75_embed_model=$L75_EMBED_MODEL"
  echo "l75_input_size=$L75_INPUT_SIZE"; echo "l75_weights=${L75_WEIGHTS:-}"; echo "recalibrate=$RECAL"; } > "$OUT_DIR/session.cfg"
command -v python3 >/dev/null 2>&1 || { echo "нужен python3 на хосте (отчёт)" >&2; exit 2; }
command -v docker >/dev/null 2>&1 || stop 0 "нет docker"
[ -n "${DATA_DIR:-}" ] && [ -d "$DATA_DIR" ] || stop 0 "DATA_DIR не задан или не каталог: ${DATA_DIR:-—}"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
[ -f "$DATA_DIR/test_query.csv" ] && [ -d "$DATA_DIR/images" ] || stop 0 "в $DATA_DIR нет test_query.csv и images/ — замеру нечего гонять"
if [ "$MODELS" != a10 ]; then
  [ -n "${L75_WEIGHTS:-}" ] && [ -f "$L75_WEIGHTS" ] || stop 0 "L75_WEIGHTS не задан или нет файла: ${L75_WEIGHTS:-—} (или SKIP_L75=1)"
  L75_WEIGHTS="$(cd "$(dirname "$L75_WEIGHTS")" && pwd)/$(basename "$L75_WEIGHTS")"
fi
if [ "$RECAL" = 1 ]; then
  [ -f "$DATA_DIR/train.csv" ] || stop 0 "в $DATA_DIR нет train.csv — пересчёту порогов не на чем (или SKIP_RECALIBRATE=1)"
  if [ -n "${DATASET_ZIP:-}" ]; then
    [ -f "$DATASET_ZIP" ] || stop 0 "нет файла DATASET_ZIP=$DATASET_ZIP"
    DATASET_ZIP="$(cd "$(dirname "$DATASET_ZIP")" && pwd)/$(basename "$DATASET_ZIP")"
  elif [ ! -d "$DATA_DIR/crops/train" ] || { [ ! -f "$CACHE_DIR/cheap_bg_desc.npz" ] && [ ! -f "$CACHE_DIR/cheap_bg_desc_train.npz" ]; }; then
    stop 0 "для пересчёта нужен DATASET_ZIP: нет crops/train/ в DATA_DIR или признака фона в кэше $CACHE_DIR"
  fi
  mkdir -p "$CACHE_DIR"; CACHE_DIR="$(cd "$CACHE_DIR" && pwd)"
fi
mark "проверка входа"

# ---------------------------------------------------------------- 1. окружение
say "п. 1: окружение"
{
  dirty=""; [ -n "$(git status --porcelain 2>/dev/null)" ] && dirty=" (есть незакоммиченные правки)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo —)$dirty"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$OUT_DIR/nvidia-smi.txt" 2>&1 || true
    IFS=, read -r gname gdrv gmem < <(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1) || true
    echo "gpu=${gname:-—}"; echo "driver=${gdrv# }"; echo "gpu_memory=${gmem# }"
    echo "cuda_driver=$(sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' "$OUT_DIR/nvidia-smi.txt" | head -1)"
    echo "gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    echo "gpu=nvidia-smi не найден"
  fi
  if command -v lscpu >/dev/null 2>&1; then
    lscpu > "$OUT_DIR/lscpu.txt" 2>&1 || true
    echo "cpu_model=$(sed -n 's/^Model name: *//p' "$OUT_DIR/lscpu.txt" | head -1)"
    echo "cpu_threads=$(nproc 2>/dev/null || sed -n 's/^CPU(s): *//p' "$OUT_DIR/lscpu.txt" | head -1)"
    echo "cores_per_socket=$(sed -n 's/^Core(s) per socket: *//p' "$OUT_DIR/lscpu.txt" | head -1)"
    echo "sockets=$(sed -n 's/^Socket(s): *//p' "$OUT_DIR/lscpu.txt" | head -1)"
  else                                                                   # macOS — только для сухого прогона
    echo "cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo —)"
    echo "cpu_threads=$(sysctl -n hw.logicalcpu 2>/dev/null || echo —)"
    echo "cores_per_socket=$(sysctl -n hw.physicalcpu 2>/dev/null || echo —)"; echo "sockets=1"
  fi
  if [ -r /proc/meminfo ]; then echo "ram=$(awk '/MemTotal/ {printf "%.0f ГБ", $2 / 1048576}' /proc/meminfo)"
  else echo "ram=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f ГБ", $1 / 1073741824}')"; fi
  echo "docker=$(to 60 docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'не отвечает')"
  echo "nvidia_ctk=$(nvidia-ctk --version 2>/dev/null | head -1 || echo —)"
} > "$OUT_DIR/env_host.txt"
sed 's/^/    /' "$OUT_DIR/env_host.txt"
mark "окружение"

# ---------------------------------------------------------------- 2. образ
ensure_a10_weights() {   # сборке нужен настоящий файл a10 в weights/ или WEIGHTS_URL — как у жюри
  local want have
  if [ -f "$A10_FILE" ] && [ "$(wc -c < "$A10_FILE")" -ge 1048576 ] && [ "$(head -c 23 "$A10_FILE")" != "version https://git-lfs" ]; then return 0; fi
  [ -n "${WEIGHTS_URL:-}" ] && return 0
  [ -n "${A10_WEIGHTS:-}" ] || stop 2 "$A10_FILE — указатель LFS, а WEIGHTS_URL и A10_WEIGHTS не заданы: git lfs pull, WEIGHTS_URL=<ссылка> или A10_WEIGHTS=<файл>"
  [ -f "$A10_WEIGHTS" ] || stop 2 "нет файла A10_WEIGHTS=$A10_WEIGHTS"
  want=$(sed -n 's/^ARG WEIGHTS_SHA256=//p' api/Dockerfile | head -1)
  have=$(sha256 "$A10_WEIGHTS")
  [ "$have" = "$want" ] || stop 2 "sha256 A10_WEIGHTS = $have, а сборка ждёт $want (WEIGHTS_SHA256 в api/Dockerfile)"
  cp "$A10_WEIGHTS" "$A10_FILE"
  say "    $A10_FILE был указателем LFS — положил A10_WEIGHTS (sha256 совпал); с git-lfs git status остаётся чистым"
}
say "п. 2: образ $IMAGE"
built="взят готовый"
if [ "${BUILD:-0}" = 1 ] || ! to 120 docker image inspect "$IMAGE" >/dev/null 2>&1; then
  [ "$IMAGE" = "case17-api:local" ] || stop 2 "образа $IMAGE нет, а docker compose build api собирает case17-api:local — уберите IMAGE"
  ensure_a10_weights
  say "    сборка как у жюри: docker compose build api (README, раздел 3)"
  WEIGHTS_URL="${WEIGHTS_URL:-}" docker compose build api || stop 2 "сборка образа не прошла — лог выше"
  built="собран сейчас: docker compose build api"
fi
{ echo "image=$IMAGE"; echo "built=$built"
  echo "id=$(to 120 docker image inspect "$IMAGE" --format '{{.Id}} создан {{.Created}}, {{.Size}} байт' 2>/dev/null || echo —)"; } > "$OUT_DIR/image.txt"
if [ "$RECAL" = 1 ]; then
  if [ "$built" != "взят готовый" ] || ! to 120 docker image inspect "$RIMAGE" >/dev/null 2>&1; then
    say "    образ для пересчёта $RIMAGE: $IMAGE + pandas, matplotlib (research/Dockerfile.research)"
    docker build -t "$RIMAGE" --build-arg BASE="$IMAGE" - < research/Dockerfile.research || stop 2 "образ $RIMAGE не собрался"
  fi
  echo "research_image=$RIMAGE" >> "$OUT_DIR/image.txt"
fi
mark "образ"

# ---------------------------------------------------------------- 3. GPU из контейнера
say "п. 3: видна ли видеокарта из контейнера"
PROBE='import json, torch
c = torch.cuda.is_available()
print(json.dumps({"cuda": c, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version() if c else None, "gpu": torch.cuda.get_device_name(0) if c else None,
                  "capability": list(torch.cuda.get_device_capability(0)) if c else None,
                  "arch_list": torch.cuda.get_arch_list() if c else []}))
raise SystemExit(0 if c else 3)'
set +e
to 600 docker run --rm --gpus all --network none "$IMAGE" python -c "$PROBE" > "$OUT_DIR/gpu_check.json" 2> "$OUT_DIR/gpu_check.err"
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  if grep -q '"cuda"' "$OUT_DIR/gpu_check.json" 2>/dev/null; then
    why="контейнер с --gpus all запустился, но torch.cuda.is_available() = False"
  else
    [ "$rc" = 124 ] && why="docker run --gpus all не ответил за 600 с" || \
      why="docker run --gpus all не запустился: $(tr '\n' ' ' < "$OUT_DIR/gpu_check.err" | tail -c 300 | sed 's/[ .]*$//')"
    to 600 docker run --rm --network none "$IMAGE" python -c "$PROBE" > "$OUT_DIR/gpu_check.json" 2>/dev/null || true   # версии torch — без GPU
    grep -q '"cuda"' "$OUT_DIR/gpu_check.json" 2>/dev/null || rm -f "$OUT_DIR/gpu_check.json"
  fi
  stop 3 "видеокарта не видна из контейнера — $why. Проверьте: nvidia-smi на хосте; nvidia-container-toolkit (sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker); torch в образе — CUDA-сборка."
fi
say "    $(cat "$OUT_DIR/gpu_check.json")"
mark "проверка GPU"

# ---------------------------------------------------------------- 4. замер extract()
L75_ENV=(-v "${L75_WEIGHTS:-/dev/null}":/opt/ext/l75.pt:ro -e "EMBED_MODEL=$L75_EMBED_MODEL" -e REID_WEIGHTS=/opt/ext/l75.pt
         -e "INPUT_SIZE=$L75_INPUT_SIZE")
[ -n "${L75_EMBED_DIM:-}" ] && L75_ENV+=(-e "EMBED_DIM=$L75_EMBED_DIM")
[ -n "${L75_TF_MODE:-}" ] && L75_ENV+=(-e "REID_TF_MODE=$L75_TF_MODE")
for m in $MODELS; do
  extra=(); [ "$m" = l75 ] && extra=("${L75_ENV[@]}")
  for t in $THREADS; do
    say "п. 4: bench_extract.py — $m, EXTRACT_THREADS=$t"
    rm -f "$OUT_DIR/bench_${m}_t${t}.json" "$OUT_DIR/bench_${m}_t${t}.err"
    if to 1800 docker run --rm --gpus all --network none -v "$DATA_DIR":/data:ro -v "$OUT_DIR":/out \
         ${extra[@]+"${extra[@]}"} -e "EXTRACT_THREADS=$t" "$IMAGE" \
         python scripts/bench_extract.py --data /data --out "/out/bench_${m}_t${t}.json" > "$OUT_DIR/bench_${m}_t${t}.log" 2>&1; then
      say "    $(python3 research/a5000_report.py --line "$OUT_DIR/bench_${m}_t${t}.json")"
    else
      tail -5 "$OUT_DIR/bench_${m}_t${t}.log" > "$OUT_DIR/bench_${m}_t${t}.err"
      say "    ОШИБКА замера $m t$t: $(tail -1 "$OUT_DIR/bench_${m}_t${t}.err") — лог bench_${m}_t${t}.log"
    fi
  done
done
touch "$OUT_DIR/bench_done"
mark "замеры"

# ---------------------------------------------------------------- 5. пересчёт порогов для L75
if [ "$RECAL" = 1 ]; then
  say "п. 5: research/recalibrate_for_model.sh для L75 (протокол Q17, без данных теста)"
  zip_args=(); [ -n "${DATASET_ZIP:-}" ] && zip_args=(-v "$DATASET_ZIP":/zip/dataset.zip:ro -e DATASET_ZIP=/zip/dataset.zip)
  to 7200 docker run --rm --gpus all --network none -v "$REPO":/repo:ro -v "$DATA_DIR":/data:ro \
      ${zip_args[@]+"${zip_args[@]}"} -v "$CACHE_DIR":/cache -v "$OUT_DIR":/out "${L75_ENV[@]}" -e DATA_DIR=/data \
      -e PYTHON=python -w /repo "$RIMAGE" bash research/recalibrate_for_model.sh l75 --cache-dir /cache \
      --out /out/recalibrate_l75.json > "$OUT_DIR/recalibrate_l75.log" 2>&1 \
    || stop 5 "пересчёт порогов L75 не прошёл: $(tail -2 "$OUT_DIR/recalibrate_l75.log" | tr '\n' ' ') — лог recalibrate_l75.log"
  grep -E "^\| (361|1111) |^SEARCH_THRESHOLD|^PP_CAMERA_TAU" "$OUT_DIR/recalibrate_l75.log" | sed 's/^/    /'
  mark "пересчёт L75"
fi

# ---------------------------------------------------------------- 6. отчёт
finish 0 "готово"
