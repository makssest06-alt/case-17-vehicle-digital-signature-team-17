#!/bin/bash
# Ставит веса, обученные на Kaggle или в Colab, в прототип — одной командой.
#
#   ./scripts/install_trained_weights.sh ~/best.pt
#   ./scripts/install_trained_weights.sh ~/best_so400m.pt --name reid_so400m.pt
#   ./scripts/install_trained_weights.sh ~/best.pt --no-reimport   # быстрее, без галереи
#   ./scripts/install_trained_weights.sh ~/best.pt --dry-run       # только прочитать чекпоинт
#
# Что делает:
#   1. читает чекпоинт и достаёт конфиг: бэкбон, размерности, метрики;
#   2. кладёт веса и в /opt/case17/models, и в weights/ репозитория —
#      второе обязательно, иначе чистый клон поднимется без нашей модели;
#   3. прописывает EMBED_MODEL, EMBED_DIM, REID_WEIGHTS в .env;
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

while [ $# -gt 0 ]; do
  case "$1" in
    --name)        NAME="$2"; shift 2 ;;
    --no-reimport) REIMPORT=0; shift ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *)             CKPT="$1"; shift ;;
  esac
done

[ -n "$CKPT" ] || { echo "укажи путь к чекпоинту, см. --help" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "нет файла $CKPT" >&2; exit 1; }

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
    "preset": cfg.get("preset", "-"),
    "epoch": ck.get("epoch", "-"),
    "mAP": m.get("mAP", "-"),
    "rank1": m.get("rank1", "-"),
}, ensure_ascii=False))
PY
)
echo "$INFO"
read -r BACKBONE OUTDIM <<<"$(echo "$INFO" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['backbone'], d['out_dim'])")"
[ "$BACKBONE" != "?" ] || { echo "в чекпоинте нет имени бэкбона — не могу собрать образ" >&2; exit 1; }
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
python3 - "$BACKBONE" "$OUTDIM" "$NAME" <<'PY'
import sys, re, pathlib
backbone, outdim, name = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path("/opt/case17/app/.env"); s = p.read_text()
def setkv(s, k, v):
    if re.search(rf"(?m)^{k}=", s):
        return re.sub(rf"(?m)^{k}=.*$", f"{k}={v}", s)
    return s.rstrip("\n") + f"\n{k}={v}\n"
s = setkv(s, "EMBED_MODEL", f"reid:{backbone}")
s = setkv(s, "EMBED_DIM", outdim)
s = setkv(s, "REID_WEIGHTS", f"/models/{name}")
p.write_text(s)
print(f"  EMBED_MODEL=reid:{backbone}  EMBED_DIM={outdim}  REID_WEIGHTS=/models/{name}")
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
