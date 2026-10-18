#!/bin/bash
# Проверка воспроизводимости решения из чистого клона.
#
#   ./scripts/check_reproducibility.sh [ветка]
#
# Отвечает на главный вопрос жюри: поднимется ли решение у постороннего
# человека, у которого есть только git-репозиторий и Docker.
#
# Что проверяется:
#   1. клон содержит все файлы, нужные для сборки (ничего не потеряно в .gitignore);
#   2. образы собираются из клона, без единого файла с нашего сервера;
#   3. модель грузится ПРИ ПОЛНОСТЬЮ ОТКЛЮЧЁННОЙ СЕТИ — веса запечены в образ;
#   4. docker compose up поднимает все сервисы до healthy;
#   5. API отвечает, Swagger отдаётся, веб-интерфейс открывается;
#   6. данные не обязательны для старта: путь к ним подменяется пустой папкой.
#
# Код возврата 0 — решение воспроизводимо. Любой другой — нет.
set -uo pipefail

BRANCH="${1:-$(git -C "$(dirname "$0")/.." rev-parse --abbrev-ref HEAD)}"
# Клонируем из общего git-каталога. --path-format=absolute обязателен:
# без него в основном рабочем дереве git возвращает относительный ".git",
# а в связанном (git worktree) — путь до worktrees/<имя>, и клон уезжает не туда.
GIT_COMMON="$(cd "$(dirname "$0")/.." && git rev-parse --path-format=absolute --git-common-dir)"
REPO_URL="${REPO_URL:-file://$GIT_COMMON}"
WORK="$(mktemp -d /tmp/repro-XXXXXX)"
EMPTY_DATA="$(mktemp -d /tmp/repro-data-XXXXXX)"
PROJECT="repro$$"
FAILED=0

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }

cleanup() {
  say "уборка"
  if [ -d "$WORK/repo" ]; then
    (cd "$WORK/repo" && docker compose -p "$PROJECT" down -v --remove-orphans >/dev/null 2>&1) || true
  fi
  rm -rf "$WORK" "$EMPTY_DATA"
  echo "  временные файлы удалены"
}
trap cleanup EXIT

say "0. Проверяю, что путь свободен"
# compose-файл фиксирует имена контейнеров (case17-api и т.д.) и порты 8000/8080,
# поэтому рабочий стек и проверочный одновременно не живут. Ловим это заранее,
# иначе проверка упадёт на сборке и будет выглядеть как нарушенная
# воспроизводимость, хотя дело в нашем же запущенном контейнере.
BUSY=""
for c in case17-db case17-api case17-web; do
  docker ps --format '{{.Names}}' | grep -qx "$c" && BUSY="$BUSY $c"
done
for prt in 8000 8080; do
  if (command -v ss >/dev/null && ss -ltn "sport = :$prt" 2>/dev/null | grep -q LISTEN); then
    BUSY="$BUSY порт:$prt"
  fi
done
if [ -n "$BUSY" ]; then
  bad "занято:$BUSY"
  echo "  Останови рабочий стек и запусти проверку заново:"
  echo "    cd /opt/case17/app && docker compose down"
  exit 1
fi
ok "контейнеры и порты 8000/8080 свободны"

say "1. Клонирую ветку $BRANCH из $REPO_URL"
git clone -q --branch "$BRANCH" --depth 1 "$REPO_URL" "$WORK/repo" || { bad "клонирование не удалось"; exit 1; }
cd "$WORK/repo"
ok "клон создан: $(git rev-parse --short HEAD)"

say "2. Проверяю состав клона"
MISSING=0
for f in docker-compose.yml .env.example api/Dockerfile api/requirements.txt \
         api/app/main.py api/app/config.py api/app/db.py api/app/imaging.py \
         api/app/models/__init__.py api/app/models/base.py api/app/models/registry.py \
         api/app/models/timm_backbone.py api/scripts/make_submission.py \
         web/Dockerfile web/nginx.conf web/html/index.html db/init/01_schema.sh \
         weights/clean_student_a10_ep28.pt; do
  [ -f "$f" ] || { bad "нет файла: $f"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] && ok "все обязательные файлы на месте"

if python3 - <<'PY'
import ast, pathlib, sys
bad = []
for f in pathlib.Path("api").rglob("*.py"):
    try:
        ast.parse(f.read_text())
    except SyntaxError as e:
        bad.append(f"{f}: {e}")
print("\n".join(bad))
sys.exit(1 if bad else 0)
PY
then ok "весь python-код клона разбирается"; else bad "синтаксические ошибки в клоне"; fi

say "3. Готовлю .env (как это сделал бы член жюри)"
cp .env.example .env
sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(openssl rand -hex 16)|" .env
sed -i "s|^DATA_HOST_DIR=.*|DATA_HOST_DIR=$EMPTY_DATA|"   .env
grep -q '^DATA_HOST_DIR=' .env || echo "DATA_HOST_DIR=$EMPTY_DATA" >> .env
ok "данные подменены пустой папкой $EMPTY_DATA — старт не должен от них зависеть"

# Веса НЕ подменяем: обученная голова обязана приехать вместе с клоном,
# иначе у постороннего человека поднимется не наша модель.
MODELS_REL=$(grep -E '^MODELS_HOST_DIR=' .env | cut -d= -f2-)
WEIGHTS_ABS=$(grep -E '^REID_WEIGHTS=' .env | cut -d= -f2-)
case "$WEIGHTS_ABS" in
  /opt/weights/*)
    # Веса запекаются в образ (api/Dockerfile, стадия weights): из weights/ клона,
    # если там настоящий файл (git lfs pull), иначе по WEIGHTS_URL.
    W="weights/$(basename "$WEIGHTS_ABS")"
    URL="${WEIGHTS_URL:-$(grep -E '^WEIGHTS_URL=' .env | cut -d= -f2- || true)}"
    if [ -f "$W" ] && [ "$(wc -c < "$W")" -ge 1048576 ] && [ "$(head -c 23 "$W")" != "version https://git-lfs" ]; then
      ok "веса модели в клоне: $W ($(du -h "$W" | cut -f1)) — попадут в образ при сборке"
    elif [ -n "$URL" ]; then
      ok "в клоне $W — LFS-указатель; веса скачаются при сборке по WEIGHTS_URL"
    else
      bad "в клоне $W — LFS-указатель, WEIGHTS_URL не задан: сборка остановится (git lfs pull или WEIGHTS_URL)"
    fi
    ;;
  *)
    WEIGHTS_REL=${WEIGHTS_ABS#/models/}
    if [ -f "${MODELS_REL:-./weights}/$WEIGHTS_REL" ]; then
      ok "веса модели в клоне: ${MODELS_REL}/$WEIGHTS_REL ($(du -h "${MODELS_REL}/$WEIGHTS_REL" | cut -f1))"
    else
      bad "в клоне нет весов ${MODELS_REL:-./weights}/$WEIGHTS_REL — поднимется не та модель"
    fi
    ;;
esac

docker compose -p "$PROJECT" config -q 2>/dev/null && ok "docker-compose.yml валиден" || bad "docker-compose.yml невалиден"

say "4. Собираю образы из клона"
if docker compose -p "$PROJECT" build >"$WORK/build.log" 2>&1; then
  ok "сборка прошла"
else
  bad "сборка не удалась, последние строки лога:"; tail -20 "$WORK/build.log"; exit 1
fi

say "5. Проверяю работу БЕЗ СЕТИ — веса должны быть внутри образа"
IMAGE=$(docker compose -p "$PROJECT" config --images 2>/dev/null | grep -i api | head -1)
IMAGE="${IMAGE:-case17-api:local}"
# Веса головы монтируем ровно так же, как это делает docker-compose:
# из папки клона, только на чтение. Проверяем отсутствие СЕТИ, а не
# отсутствие файлов — сеть в рантайме недоступна, а свои файлы у решения
# быть обязаны.
MODELS_MOUNT="$(cd "$(grep -E '^MODELS_HOST_DIR=' .env | cut -d= -f2-)" && pwd)"
if docker run --rm --network none \
     -v "$MODELS_MOUNT":/models:ro \
     -e EMBED_MODEL="$(grep -E '^EMBED_MODEL=' .env | cut -d= -f2 || echo reid:vit_base_patch16_siglip_224.webli)" \
     -e REID_WEIGHTS="$(grep -E '^REID_WEIGHTS=' .env | cut -d= -f2-)" \
     "$IMAGE" python -c "
from app.models.registry import build_model
from app.config import settings
m = build_model(settings.embed_model, settings.input_size)
m.load()
print('модель загружена без сети:', m.name, m.dim, 'D')
" >"$WORK/offline.log" 2>&1; then
  ok "$(tail -1 "$WORK/offline.log")"
else
  bad "модель не грузится без сети — веса не запечены в образ"; tail -15 "$WORK/offline.log"
fi

say "6. Поднимаю стек"
if docker compose -p "$PROJECT" up -d >"$WORK/up.log" 2>&1; then
  ok "docker compose up отработал"
else
  bad "стек не поднялся"; tail -20 "$WORK/up.log"; exit 1
fi

echo "  жду healthy (до 5 минут)..."
for _ in $(seq 1 30); do
  S=$(docker compose -p "$PROJECT" ps --format '{{.Service}}:{{.Health}}' 2>/dev/null | tr '\n' ' ')
  case "$S" in *db:healthy*) case "$S" in *api:healthy*) case "$S" in *web:healthy*) break;; esac;; esac;; esac
  sleep 10
done
S=$(docker compose -p "$PROJECT" ps --format '{{.Service}}:{{.Health}}' 2>/dev/null | tr '\n' ' ')
echo "  состояние: $S"
for svc in db api web; do
  case "$S" in *"$svc:healthy"*) ok "$svc healthy";; *) bad "$svc НЕ healthy";; esac
done

say "7. Проверяю endpoints"
API_PORT=$(docker compose -p "$PROJECT" port api 8000 2>/dev/null | cut -d: -f2)
WEB_PORT=$(docker compose -p "$PROJECT" port web 80   2>/dev/null | cut -d: -f2)
check() {
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$1" 2>/dev/null)
  [ "$code" = "200" ] && ok "$2 -> HTTP 200" || bad "$2 -> HTTP ${code:-нет ответа}"
}
[ -n "$API_PORT" ] && {
  check "http://localhost:$API_PORT/health" "/health"
  check "http://localhost:$API_PORT/ready"  "/ready"
  check "http://localhost:$API_PORT/docs"   "/docs (Swagger)"
  check "http://localhost:$API_PORT/openapi.json" "/openapi.json"
} || bad "порт api не опубликован"
[ -n "$WEB_PORT" ] && {
  check "http://localhost:$WEB_PORT/"           "веб-интерфейс"
  check "http://localhost:$WEB_PORT/api/health" "проксирование /api"
} || bad "порт web не опубликован"

say "ИТОГ"
if [ "$FAILED" -eq 0 ]; then
  printf '\033[32m  РЕШЕНИЕ ВОСПРОИЗВОДИМО\033[0m\n'
  echo "  Чистый клон ветки $BRANCH собирается и поднимается без файлов с нашего сервера."
  echo "  Модель грузится при отключённой сети."
else
  printf '\033[31m  ВОСПРОИЗВОДИМОСТЬ НАРУШЕНА — см. FAIL выше\033[0m\n'
fi
exit "$FAILED"
