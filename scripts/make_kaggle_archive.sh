#!/bin/bash
# Собирает архив кропов для заливки на Kaggle.
# Без служебных файлов macOS и без сжатия: JPEG уже сжат, а -0 экономит минуты CPU.
set -e
SRC=/opt/case17/data/crops
OUT=${1:-/opt/case17/data/case17_crops_kaggle.zip}

command -v zip >/dev/null || { echo "нужен zip: sudo apt-get install -y zip"; exit 1; }
[ -d "$SRC" ] || { echo "нет папки $SRC"; exit 1; }

rm -f "$OUT"
cd "$SRC"
zip -0 -q -r "$OUT" . -x '._*' -x '*/._*' -x '.DS_Store' -x '*/.DS_Store'

echo "архив: $OUT"
ls -lh "$OUT" | awk '{print "  размер:", $5}'
echo "  файлов внутри: $(unzip -l "$OUT" | tail -1 | awk '{print $2}')"
echo "  jpg:           $(unzip -l "$OUT" | grep -c '\.jpg$')"
echo "  csv:           $(unzip -l "$OUT" | grep -c '\.csv$')"
