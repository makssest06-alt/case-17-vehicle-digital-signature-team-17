#!/bin/sh
# Веса модели в образ без git-lfs у того, кто собирает (стадия weights в api/Dockerfile).
#
#   fetch_weights.sh <файл из контекста сборки> <куда положить> <WEIGHTS_URL> <WEIGHTS_SHA256>
#
# Файл из контекста настоящий (≥ 1 МБ и не начинается с «version https://git-lfs») —
# берётся как есть, ничего не качается. Иначе (LFS-указатель или файла нет) — скачивается
# по WEIGHTS_URL. В обоих случаях sha256 сверяется с WEIGHTS_SHA256; не совпал — сборка падает.
set -eu

SRC="$1"; OUT="$2"; URL="${3:-}"; WANT="${4:-}"
MIN_BYTES=1048576

fail() { echo "ОШИБКА (веса): $*" >&2; exit 1; }

[ -n "$WANT" ] || fail "WEIGHTS_SHA256 пуст — сверять не с чем"

is_real() {
  [ -f "$1" ] && [ "$(wc -c < "$1")" -ge "$MIN_BYTES" ] && [ "$(head -c 23 "$1")" != "version https://git-lfs" ]
}

mkdir -p "$(dirname "$OUT")"
if is_real "$SRC"; then
  echo "веса: $(basename "$SRC") из контекста сборки, $(wc -c < "$SRC") байт — не качаю"
  [ "$SRC" = "$OUT" ] || cp "$SRC" "$OUT"
else
  if [ -f "$SRC" ]; then
    WHY="LFS-указатель ($(wc -c < "$SRC") байт), а не веса"
  else
    WHY="файла нет"
  fi
  [ -n "$URL" ] || fail "weights/$(basename "$SRC") в контексте сборки — $WHY. Укажите WEIGHTS_URL или выполните git lfs pull: \`git lfs install && git lfs pull\` либо \`WEIGHTS_URL=<ссылка> docker compose up -d --build\`"
  echo "веса: в контексте сборки $WHY — качаю ${URL%%\?*}"
  python3 - "$URL" "$OUT.part" <<'PY' || { rm -f "$OUT.part"; fail "не удалось скачать по WEIGHTS_URL"; }
import shutil, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=60) as r, open(sys.argv[2], "wb") as f:
    shutil.copyfileobj(r, f, 1 << 20)
PY
  mv "$OUT.part" "$OUT"
  echo "веса: скачано $(wc -c < "$OUT") байт"
fi

HAVE="$(sha256sum "$OUT" | cut -d' ' -f1)"
if [ "$HAVE" != "$WANT" ]; then
  rm -f "$OUT"
  fail "sha256 $(basename "$OUT") = $HAVE, а ожидается $WANT (WEIGHTS_SHA256) — не те веса или файл повреждён"
fi
echo "веса: sha256 совпадает с WEIGHTS_SHA256 ($HAVE)"
