#!/bin/bash
# Ставит веса, обученные на Kaggle или в Colab, в прототип — одной командой.
#
#   ./scripts/install_trained_weights.sh ~/best.pt
#   ./scripts/install_trained_weights.sh ~/best_so400m.pt --name reid_so400m.pt
#   ./scripts/install_trained_weights.sh ~/clean_student_a10_ep28.pt \
#       --name clean_student_a10_ep28.pt --tf-mode squash --tta-flip 0
#   ./scripts/install_trained_weights.sh ~/best.pt --no-reimport   # быстрее, без галереи
#   ./scripts/install_trained_weights.sh ~/best.pt --dry-run       # только прочитать чекпоинт
#
# Что делает:
#   1. читает чекпоинт и достаёт конфиг: бэкбон, размерности, метрики;
#   2. кладёт веса и в /opt/case17/models, и в weights/ репозитория —
#      второе обязательно, иначе чистый клон поднимется без нашей модели;
#   3. прописывает EMBED_MODEL, EMBED_DIM, REID_WEIGHTS в .env;
#      --tf-mode/--tta-flip позволяют явно зафиксировать preprocessing модели;
#   4. пересобирает образ и поднимает стек, ждёт healthy;
#   5. переимпортирует галерею и пересчитывает артефакты сдачи.
#
# Если изменилась размерность вектора — том базы пересоздаётся: pgvector
# хранит колонку фиксированной длины, старые векторы несовместимы.
set -euo pipefail

APP="$(cd "$(dirname "$0")/.." && pwd)"
MODELS=/opt/case17/models
NAME=reid_current.pt
REIMPORT=1
DRY=0
CKPT=""
TF_MODE_OVERRIDE=""
TTA_OVERRIDE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --name)        NAME="$2"; shift 2 ;;
    --tf-mode)     TF_MODE_OVERRIDE="$2"; shift 2 ;;
    --tta-flip)    TTA_OVERRIDE="$2"; shift 2 ;;
    --no-reimport) REIMPORT=0; shift ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *)             CKPT="$1"; shift ;;
  esac
done

[ -n "$CKPT" ] || { echo "укажи путь к чекпоинту, см. --help" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "нет файла $CKPT" >&2; exit 1; }
[ -z "$TF_MODE_OVERRIDE" ] || case "$TF_MODE_OVERRIDE" in
  timm|squash) ;;
  *) echo "--tf-mode должен быть timm или squash" >&2; exit 2 ;;
esac
[ -z "$TTA_OVERRIDE" ] || case "$TTA_OVERRIDE" in
  0|1) ;;
  *) echo "--tta-flip должен быть 0 или 1" >&2; exit 2 ;;
esac

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
ok()  { printf '  \033[32m%s\033[0m\n' "$*"; }

cd "$APP"

say "1. читаю чекпоинт"
INFO=$(docker compose run --rm --no-deps -v "$(realpath "$CKPT")":/tmp/ckpt.pt:ro api python - <<'PY'
import torch, json
ck = torch.load("/tmp/ckpt.pt", map_location="cpu", weights_only=False)
cfg = ck.get("cfg") or ck.get("args") or {}
m = ck.get("metrics") or {}
print(json.dumps({
    "backbone": cfg.get("backbone", "?"),
    "out_dim": cfg.get("out_dim", 512),
    "hidden": cfg.get("hidden", 1024),
    "input_size": cfg.get("input_size", 224),
    "tf_mode": cfg.get("tf_mode", ""),
    "preset": cfg.get("preset", "-"),
    "epoch": ck.get("epoch", "-"),
    "mAP": m.get("mAP", "-"),
    "rank1": m.get("rank1", "-"),
}, ensure_ascii=False))
PY
)
echo "$INFO"
read -r BACKBONE OUTDIM CKPT_TF_MODE <<<"$(echo "$INFO" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['backbone'], d['out_dim'], d.get('tf_mode') or '')")"
[ "$BACKBONE" != "?" ] || { echo "в чекпоинте нет имени бэкбона — не могу собрать образ" >&2; exit 1; }

# Dockerfile запекает public pretrained backbone, а full ReID checkpoint
# всё равно содержит собственный encoder. Поэтому для SigLIP2 здесь берём
# совместимый с pinned timm SigLIP1 tag: runtime loader увидит
# args["backbone"]=siglip2_base и загрузит full checkpoint в голую архитектуру.
# Это не влияет на веса full checkpoint и не требует research/ внутри Docker.
case "$BACKBONE" in
  siglip_base|siglip2_base) RESOLVED="vit_base_patch16_siglip_224.webli" ;;
  clip_vitb16)              RESOLVED="vit_base_patch16_clip_224.openai" ;;
  dinov2_vits14)            RESOLVED="vit_small_patch14_dinov2.lvd142m" ;;
  *)                        RESOLVED="$BACKBONE" ;;
esac
if [ "$RESOLVED" != "$BACKBONE" ]; then
  echo "  ключ реестра '$BACKBONE' -> модель timm '$RESOLVED'"
  BACKBONE="$RESOLVED"
fi
ok "бэкбон $BACKBONE, вектор ${OUTDIM}D"

if [ "$DRY" = "1" ]; then
  ok "--dry-run: на этом останавливаюсь"
  exit 0
fi

say "2. кладу веса"
mkdir -p "$MODELS" "$APP/weights"
STAMP=$(date +%Y%m%d-%H%M%S)
cp "$CKPT" "$MODELS/${NAME%.pt}_${STAMP}.pt"
cp "$CKPT" "$MODELS/$NAME"
# Копия в репозитории — обычный файл, не симлинк: симлинк git сохранит как
# симлинк, и в чистом клоне он укажет в пустоту.
cp "$CKPT" "$APP/weights/$NAME"
ok "$MODELS/$NAME и weights/$NAME ($(du -h "$CKPT" | cut -f1))"
echo "  история: $MODELS/${NAME%.pt}_${STAMP}.pt"

say "3. правлю .env"
OLD_DIM=$(grep -E '^EMBED_DIM=' .env | cut -d= -f2 || echo "")
EFFECTIVE_TF_MODE="$TF_MODE_OVERRIDE"
if [ -z "$EFFECTIVE_TF_MODE" ] && [ "$CKPT_TF_MODE" = "timm" -o "$CKPT_TF_MODE" = "squash" ]; then
  EFFECTIVE_TF_MODE="$CKPT_TF_MODE"
fi
python3 - "$APP/.env" "$BACKBONE" "$OUTDIM" "$NAME" "$EFFECTIVE_TF_MODE" "$TTA_OVERRIDE" <<'PY'
import sys, re, pathlib
env_path, backbone, outdim, name, tf_mode, tta_flip = sys.argv[1:]
p = pathlib.Path(env_path); s = p.read_text()
def setkv(s, k, v):
    if re.search(rf"(?m)^{k}=", s):
        return re.sub(rf"(?m)^{k}=.*$", f"{k}={v}", s)
    return s.rstrip("\n") + f"\n{k}={v}\n"
s = setkv(s, "EMBED_MODEL", f"reid:{backbone}")
s = setkv(s, "EMBED_DIM", outdim)
s = setkv(s, "REID_WEIGHTS", f"/models/{name}")
if tf_mode:
    s = setkv(s, "REID_TF_MODE", tf_mode)
if tta_flip:
    s = setkv(s, "EMBED_TTA_FLIP", tta_flip)
p.write_text(s)
print(f"  EMBED_MODEL=reid:{backbone}  EMBED_DIM={outdim}  REID_WEIGHTS=/models/{name}")
if tf_mode:
    print(f"  REID_TF_MODE={tf_mode}")
if tta_flip:
    print(f"  EMBED_TTA_FLIP={tta_flip}")
PY

say "4. пересборка и запуск"
if [ -n "$OLD_DIM" ] && [ "$OLD_DIM" != "$OUTDIM" ]; then
  echo "  размерность вектора изменилась ($OLD_DIM -> $OUTDIM), пересоздаю том базы"
  docker compose down -v
  REIMPORT=1
else
  docker compose down
fi
docker compose build api
docker compose up -d

say "5. жду healthy"
for _ in $(seq 1 40); do
  S=$(docker compose ps --format '{{.Service}}:{{.Health}}' | tr '\n' ' ')
  case "$S" in *api:healthy*) case "$S" in *db:healthy*) case "$S" in *web:healthy*) break;; esac;; esac;; esac
  sleep 10
done
docker compose ps --format '{{.Service}}\t{{.Status}}'
curl -s http://localhost:8000/ready; echo

if [ "$REIMPORT" = "0" ]; then
  say "готово (без переимпорта)"
  echo "  галерея осталась со старыми векторами — поиск будет врать."
  echo "  Когда будешь готов:"
  echo "    curl -X POST localhost:8000/v1/gallery/bulk-import -H 'Content-Type: application/json' \\"
  echo "      -d '{\"images_dir\":\"/data/crops/gallery\",\"csv_path\":\"/data/crops/crops_gallery.csv\",\"truncate\":true}'"
  echo "    docker compose exec api python scripts/make_submission.py"
  exit 0
fi

say "6. переимпорт галереи"
echo "  это ~15 минут на CPU, векторы считаются заново"
curl -s -X POST http://localhost:8000/v1/gallery/bulk-import \
  -H 'Content-Type: application/json' \
  -d '{"images_dir":"/data/crops/gallery","csv_path":"/data/crops/crops_gallery.csv","truncate":true}'
echo

say "7. артефакты сдачи"
docker compose exec -T api python scripts/make_submission.py

say "ГОТОВО"
curl -s http://localhost:8000/ready; echo
echo "  Не забудь про порог отказа: он привязан к модели."
echo "    python research/eval_refusal.py --backbone <...> --heads <тег> --protocol deployment"
echo "  и вписать найденное значение в SEARCH_THRESHOLD."
