#!/bin/bash
# Ставит веса, обученные на Kaggle, в прототип — одной командой.
#
#   ./scripts/install_trained_weights.sh ~/best.pt
#
# Что делает:
#   1. проверяет чекпоинт и достаёт из него конфиг (бэкбон, размерности);
#   2. кладёт веса в /opt/case17/models;
#   3. прописывает их в .env;
#   4. пересобирает api, поднимает стек и ждёт healthy;
#   5. если изменилась размерность вектора — пересоздаёт том базы.
set -e

CKPT="${1:?укажи путь к чекпоинту: ./scripts/install_trained_weights.sh /путь/best.pt}"
APP=/opt/case17/app
MODELS=/opt/case17/models
[ -f "$CKPT" ] || { echo "нет файла $CKPT"; exit 1; }

echo "=== 1. читаю чекпоинт ==="
cd "$APP"
INFO=$(docker compose run --rm --no-deps -v "$(realpath "$CKPT")":/tmp/ckpt.pt api python - <<'PY'
import torch, json
ck = torch.load("/tmp/ckpt.pt", map_location="cpu", weights_only=False)
cfg = ck.get("cfg") or ck.get("args") or {}
print(json.dumps({
    "backbone": cfg.get("backbone", "?"),
    "out_dim": cfg.get("out_dim", 512),
    "hidden": cfg.get("hidden", 1024),
    "input_size": cfg.get("input_size", 224),
    "mAP": (ck.get("metrics") or {}).get("mAP", "?"),
}, ensure_ascii=False))
PY
)
echo "$INFO"
BACKBONE=$(echo "$INFO" | python3 -c "import sys,json;print(json.load(sys.stdin)['backbone'])")
OUTDIM=$(echo   "$INFO" | python3 -c "import sys,json;print(json.load(sys.stdin)['out_dim'])")

echo "=== 2. кладу веса ==="
mkdir -p "$MODELS"
STAMP=$(date +%Y%m%d-%H%M%S)
cp "$CKPT" "$MODELS/reid_${STAMP}.pt"
ln -sfn "$MODELS/reid_${STAMP}.pt" "$MODELS/reid_current.pt"
echo "  $MODELS/reid_current.pt -> reid_${STAMP}.pt"

echo "=== 3. правлю .env ==="
OLD_DIM=$(grep -E '^EMBED_DIM=' .env | cut -d= -f2 || echo "")
python3 - "$BACKBONE" "$OUTDIM" <<'PY'
import sys, re, pathlib
backbone, outdim = sys.argv[1], sys.argv[2]
p = pathlib.Path("/opt/case17/app/.env"); s = p.read_text()
def setkv(s, k, v):
    if re.search(rf"(?m)^{k}=", s):
        return re.sub(rf"(?m)^{k}=.*$", f"{k}={v}", s)
    return s.rstrip("\n") + f"\n{k}={v}\n"
s = setkv(s, "EMBED_MODEL", f"reid:{backbone}")
s = setkv(s, "EMBED_DIM", outdim)
s = setkv(s, "REID_WEIGHTS", "/models/reid_current.pt")
p.write_text(s)
print("  EMBED_MODEL, EMBED_DIM, REID_WEIGHTS обновлены")
PY

echo "=== 4. пересборка и запуск ==="
if [ -n "$OLD_DIM" ] && [ "$OLD_DIM" != "$OUTDIM" ]; then
  echo "  размерность вектора изменилась ($OLD_DIM -> $OUTDIM), пересоздаю том базы"
  docker compose down -v
else
  docker compose down
fi
docker compose build api
docker compose up -d

echo "=== 5. жду healthy ==="
for i in $(seq 1 40); do
  S=$(docker compose ps --format '{{.Service}}:{{.Health}}' | tr '\n' ' ')
  echo "$S" | grep -q 'api:healthy' && echo "$S" | grep -q 'db:healthy' && echo "$S" | grep -q 'web:healthy' && break
  sleep 10
done
docker compose ps --format '{{.Service}}\t{{.Status}}'
curl -s http://localhost:8000/ready; echo

echo
echo "Готово. Дальше переимпортируй галерею и пересчитай артефакты:"
echo "  curl -X POST localhost:8000/v1/gallery/bulk-import -H 'Content-Type: application/json' \\"
echo "    -d '{\"images_dir\":\"/data/crops/gallery\",\"csv_path\":\"/data/crops/crops_gallery.csv\",\"truncate\":true}'"
echo "  docker compose exec api python scripts/make_submission.py"
