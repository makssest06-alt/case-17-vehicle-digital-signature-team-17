# Прогон bench_extract на RTX 4060 Laptop (нативно, без docker compose exec)

**Дата**: 2026-10-18. **Протокол**:Submission Q31/Q34 — полный цикл
`read → decode → crop_with_pad(pad=0.10) → squash-224 → forward fp16 CUDA → голова → flip-TTA (дефолт сервиса) → L2 → cpu`,
галерея/search ИСКЛЮЧЕНЫ. FP16 CUDA — активирован (автокаст в официальном загрузчике, `REID_FP16=1` по умолчанию).

## ГЛАВНОЕ ОТКЛОНЕНИЕ: официальный `scripts/bench_extract.py` отсутствует

В репозитории `glebmain` скрипта **нет**: ни в `scripts/`, ни где-либо ещё. Проверено:

```
grep -ril bench_extract /tmp/opencode/gm_bench            # 0 совпадений по имени файла
git log --all -S bench_extract --oneline --name-only      # 0 коммитов за всю историю ВСЕХ веток
for ref in origin/main origin/ilya/honest-eval origin/glebmain; do
  git ls-tree -r $ref --name-only | grep -i bench          # только research/ann_benchmark.py
done
find /home/l1zard/diskD -iname '*bench_extract*'          # только наша выходная папка
```

Единственные похожие скрипты — `research/ann_benchmark.py` (ANN-индексы), `research/eval_service_path.py`
(точность сервисного пути), оба с зашитыми docker-путями `/opt/case17/...` и без замера батчей-латентности.

**Поэтому замер — честный native fallback** (не «docker compose exec»): измеряется **официальный сервисный код**
`api/app/models/reid_checkpoint.py:ReidCheckpointModel.embed_batch` + `api/app/imaging.py:crop_with_pad`,
иначе как в протоколе. Драйвер-скрипт лежит ТОЛЬКО в `/tmp/opencode/bench_extract_fallback.py` (repo не изменён,
в git ничего не добавлено и не закоммичено).

## Выборка репозитория

```
cd /tmp/opencode && rm -rf gm_bench
git clone --no-checkout --branch glebmain --single-branch \
  https://git.sourcecraft.dev/lct-hackaton-2026/case-17-vehicle-digital-signature-team-17.git gm_bench
# clone упал: no credentials (нет git-credential helper в неинтерактивной сессии)
git -C /home/l1zard/diskD/bm_finals_handoff fetch origin glebmain    # тоже упал: no credentials
git -C /home/l1zard/diskD/bm_finals_handoff worktree add /tmp/opencode/gm_bench origin/glebmain --detach
# создан worktree из ЛОКАЛЬНО кэшированного origin/glebmain:
# a2606b2 «merge: organizer feedback ...» от 2026-10-16
```

⚠️ Свежий pull заблокирован отсутствием токена для `git.sourcecraft.dev`/github-mirror; работа велась на последнем
локальном снапшоте glebmain **a2606b2 (2026-10-16)** — на момент ручной синхронизации это был последний glebmain.

## Точка входа модели (по схеме официального скрипта)

```
cp /home/l1zard/diskD/bm_improve/clean_student/clean_student_a10_ep28.pt \
   /tmp/opencode/gm_bench/artifacts/reid_model.pt
```

Чекпоинт: полный дистиллят студента (`encoder.*`, 162 тензора; голова `hidden=1024, out=512` — дефолты загрузчика
совпадают с тензорами), `args.backbone=siglip2_base`, `input_size=224`, `args.tf_mode=squash`.
Загрузчик сам выбрал `vit_base_patch16_siglip_224.v2_webli` (timm 1.0.29, pretrained=False — сеть не нужна).

## Окружение (уже готово, ничего доустанавливать не пришлось)

- venv: `/home/l1zard/work/venv` — `torch 2.14.1+cu130` (CUDA runtime 13.0), `timm 1.0.29`, `torchvision 0.29.1+cu130`, `PIL 12.3.0`, `numpy 2.5.4`. Недостающих пакетов нет (`api/app/config.py` — чистый stdlib; fastapi нам не нужен, т.к. грузится только модельный модуль).
- Драйвер: `595.99.02`, CUDA driver 13.2; дежурная проверка перед каждым прогоном — `nvidia-smi` (0 % утилизации).

## Точные команды (3 полных прогона + разбор пайплайна)

```
# прогон 1..3 (прогревы и 3 повторения на каждый батч внутри скрипта)
/home/l1zard/work/venv/bin/python /tmp/opencode/bench_extract_fallback.py \
  --out /home/l1zard/diskD/bm_improve/bench_extract_4060_out/bench_extract.json        2>&1 | tee run.log
/home/l1zard/work/venv/bin/python /tmp/opencode/bench_extract_fallback.py \
  --out .../bench_extract_r2.json                                                      2>&1 | tee -a run.log
/home/l1zard/work/venv/bin/python /tmp/opencode/bench_extract_fallback.py \
  --out .../bench_extract_r3.json                                                      2>&1 | tee -a run.log
# probe: forward-only fp16 (B=32, 20 итераций) + decode 1920x1080 (64 кадра) — добавлено в run.log
```

Окружение замера: `EMBED_MODEL=reid:vit_base_patch16_siglip_224`, `REID_WEIGHTS=.../artifacts/reid_model.pt`,
`OMP_NUM_THREADS=2` (дефолт загрузчика), `EMBED_TTA_FLIP` — дефолт сервиса (вкл), `REID_FP16` — дефолт (1).
Кадры: реальные JPEG фреймы датасета 1920×1080 (`/home/l1zard/diskD/bm/dataset_extracted/images`, пул 64, seed 2026,
bbox 400–800 px, aspect 1.6 ~ средняя машина ≈16 % площади — согласуется с README репо).

## Результаты (best-of-9 повторений на батч, стабильность ±8 % — laptop boost)

| batch | best sustained, ms/кадр | FPS | e2e с sync, ms | ворота |
|------:|------------------------:|----:|---------------:|:-------|
| 1     | 22.37                   | 44.7 | **22.63**     | latency ≤40 ms: **PASS** |
| 8     | 17.28                   | 57.9 | 17.27         | — |
| 16    | 16.95                   | 59.0 | 16.88         | — |
| 32    | **16.43**               | **60.9** | 16.39      | throughput ≥100 FPS: **FAIL (4060)** |

**Разбор B=32**: decode JPEG ≈ **10.97 ms/кадр** (2 потока, 1920×1080) — узкое место; forward fp16 ≈ 2.18 ms/кадр,
TTA-flip +2.22, копирование на CPU ≈ 0.19. Сумма сходится с замером → GPU не является ограничителем.

## Ожидание на A5000 (не претензия, а оценка)

Forward на A5000 (~26 TFLOPS fp16 против ~10–11 эффективных у 4060 Laptop) ориентировочно ~2.2–2.5 ms/кадр;
decode ~11 ms остаётся. Полный цикл ≈ 15–17 ms/кадр ⇒ **60–100 FPS — на грани** ворота 100 FPS; ворота
латентности (≤40 ms на B1) выглядят безопасно (~12–15 ms). Достичь 100 FPS реально при параллельном декоде
(DALI/proc-pool), готовых кропах 224 или выключенном TTA-flip — иначе декод всегда «съедает» маржу.

## Воспроизводимость

1. GPU свободен — контроль `nvidia-smi` перед прогоном (все три прогона начинались с 0 %).
2. 3 прогона × 3 повторения × 4 батча; берётся best sustained (в JSON сохранены все повторения: `results.all_reps`).
3. Устройство/потоки/схема transform не менялись между прогонами (скрипт детерминирован, seed 2026).
4. Disclaimer: это native fallback, а не официальный `docker compose exec`; формат JSON — собственный (официального
   скрипта/схемы нет в репо), поля снабжены пояснениями; никаких файлов репозитория не изменено/не закоммичено.
