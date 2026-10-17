#!/bin/bash
# Промер прод-модели протоколом honest-eval перед сменой порога.
#
#   scripts/prod_model_check.sh [-n] <данные> <веса.pt> <тег>
#
#   -n         только сверить sha256 и показать команды, ничего не считать
#   <данные>   train.csv, test_*.csv, crops/ (как для research/honest_eval.py)
#   <тег>      имя результатов, например clean_a10
#
# 1. sha256 весов против манифеста: MANIFEST=<файл> или, по умолчанию,
#    research/clean_student/manifest.json из ветки team17/finals-student-gpu-handoff;
# 2. сюита honest_eval: raw и kr на всех галереях (8996 и 10.8k — для таблицы);
# 3. отказ на 2.2k без почти-дублей вне фолда: 0.8025, recall >= 0.6 и >= 0.7;
# 4. таблица research/honest_eval_results/prod_<тег>.md рядом со старым студентом.
#
# Векторы кэшируются, повторный запуск быстрый. Python — $PYTHON или python3.
set -euo pipefail

DRY=0
if [ "${1:-}" = "-n" ]; then DRY=1; shift; fi
[ $# -eq 3 ] || { sed -n '2,16p' "$0"; exit 2; }
DATA="$1"; WEIGHTS="$2"; TAG="$3"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
RES="$ROOT/research/honest_eval_results"

[ -f "$WEIGHTS" ] || { echo "нет файла весов: $WEIGHTS"; exit 1; }
NAME="$(basename "$WEIGHTS")"
if [ -n "${MANIFEST:-}" ]; then
  MAN_TEXT="$(cat "$MANIFEST")"
else
  MAN_TEXT="$(git -C "$ROOT" show origin/team17/finals-student-gpu-handoff:research/clean_student/manifest.json)"
fi
WANT="$(printf '%s' "$MAN_TEXT" | "$PY" -c 'import json, sys; print(json.load(sys.stdin).get("sha256", {}).get(sys.argv[1], ""))' "$NAME")"
if command -v sha256sum >/dev/null; then HAVE="$(sha256sum "$WEIGHTS" | cut -d' ' -f1)"
else HAVE="$(shasum -a 256 "$WEIGHTS" | cut -d' ' -f1)"; fi
[ -n "$WANT" ] || { echo "в манифесте нет sha256 для $NAME"; exit 1; }
[ "$HAVE" = "$WANT" ] || { echo "sha256 $NAME не совпадает с манифестом: $HAVE против $WANT"; exit 1; }
echo "sha256 $NAME совпадает с манифестом: $HAVE"

run() { echo "+ $*"; [ "$DRY" -eq 1 ] || (cd "$ROOT" && "$@"); }
run "$PY" research/honest_eval.py --data-dir "$DATA" --suite --post raw,kr --tag "$TAG" --env "REID_WEIGHTS=$WEIGHTS"
run "$PY" research/calibrate_refusal_folds.py --data-dir "$DATA" --env "REID_WEIGHTS=$WEIGHTS" \
  --out "$RES/refusal_folds_$TAG.json"
run "$PY" research/honest_eval_report.py --prod "$TAG:$TAG" "student_rk:старый студент (main)" --out "$RES/prod_$TAG.md"
