# Аудит main: артефакты, документы, сервис

Только факты. Строки указаны по `origin/main` = `0924fb2` (16.10.2026), если
не сказано иное. Проверка каждого пункта — команда из корня репозитория.

Сводная проверка пунктов 1–3 одной командой (нужен только numpy):

```bash
python research/audit_main_artifacts.py --json research/honest_eval_results/main_audit_artifacts.json
```

Её вывод на `0924fb2` сохранён в `main_audit_artifacts.json` рядом.

---

## 1. `artifacts/` собраны не `api/scripts/make_submission.py`

README (`README.md:163`, `docs/SUBMISSION_README.md:192`) называет генератором
артефактов `docker compose exec api python scripts/make_submission.py`.
Закоммиченные файлы этим скриптом получиться не могут.

| Что | `make_submission.py` пишет | В `artifacts/` лежит |
|---|---|---|
| Порог отказа | `--threshold`, по умолчанию `settings.search_threshold` = 0.8025 (`api/scripts/make_submission.py:112`, `api/app/config.py:40`) | 0.094672 |
| `confidence` | сырой косинус `sim`, `round(..., 4)` (`make_submission.py:293`) | скор после qq → k-reciprocal → CSLS, от 0.09471 до 0.505664, до 6 знаков |
| Кандидатов на запрос | до 10, все с косинусом ≥ порога (`make_submission.py:294–295`) | ровно 1 (999 строк на 999 запросов) |
| Поля `postprocess.json` | `pipeline, source, params, reproduce, took_s` (`make_submission.py:240–247`) | `checkpoint, checkpoint_sha256, crop, tta, pipeline, params, refusal, consistency, val_metrics_official_evaluate_py` |
| Параметры постобработки | по умолчанию `qq_k=1`, `rerank_k1=20` (`make_submission.py:126, 139`) | `qq_k=3`, `rerank_k1=10` (`artifacts/postprocess.json:8, 10`) |

Чем собраны на самом деле: `submission.csv` совпадает с
`research/reid_postprocess.rank(...)` при параметрах из `postprocess.json`
на **1110 из 1110** запросов (все 10 позиций). `candidates.csv` = top-1 того же
ранжирования (999 из 999 совпадают по `gallery_id`, |Δconfidence| ≤ 5·10⁻⁷).
Это логика `research/reid_postprocess.py:204–218`, но тот скрипт округляет до
4 знаков (`:218`), а в файле до 6. Скрипта, который пишет поля `refusal`,
`consistency`, `val_metrics_official_evaluate_py`, в репозитории нет
(поиск по всем удалённым веткам
`git grep -l -E 'top1_mismatch_pct|val_metrics_official' $(git branch -r | grep -v HEAD)`
находит только `artifacts/postprocess.json`).

Если собрать артефакты так, как написано в README (`make_submission.py` без
флагов, порог 0.8025 на сыром косинусе), на том же тесте будет принято
**1017** запросов из 1110, а не 999.

Проверка: `python research/audit_main_artifacts.py`, разделы `submission`,
`candidates`, `make_submission_default`.

## 2. Документы противоречат `postprocess.json`

| Документ говорит | Факт |
|---|---|
| `docs/SUBMISSION_README.md:448–455`: «по умолчанию собираем артефакты без пересортировки — топ-1 сходится на 100% проб» | `artifacts/postprocess.json:6`: `"pipeline": "qq_expand -> k_reciprocal_rerank -> csls"`; `:22`: `"top1_mismatch_pct": 20.45`. Пересчёт: top-1 `submission.csv` отличается от top-1 по сырому косинусу `embeddings.npy` у **20.45 %** запросов |
| `README.md:207`, `docs/SUBMISSION_README.md:359`: рабочий порог 0.8025 | `candidates.csv` отсечён порогом 0.094672 другой шкалы (CSLS), см. п. 3 |
| `README.md:264–266`: `submission.csv` воспроизводится из `embeddings.npy` точь-в-точь | верно, но **только** через постобработку с `postprocess.json`; из сырого косинуса — нет (п. 1) |

Проверка: `git show origin/main:artifacts/postprocess.json` и
`python research/audit_main_artifacts.py` → `top1_differs_from_raw_cosine_pct`.

## 3. Порог 0.0947 — 10-й процентиль на самом тесте

`artifacts/postprocess.json:16–19`: `"rule": "10-й процентиль top-1"`,
`"threshold": 0.094672`, `"accepted": 999`, `"refused": 111`.

Источник правила — `research/reid_postprocess.py:208`:
`thr = float(np.quantile(sim.max(axis=1), 0.10))`, где `sim` — матрица
сходства **тестовых** запросов после постобработки. Пересчёт по
`embeddings.npy` даёт ровно 0.094672 и 111 отказов (10.0 % из 1110).

Следствия, без оценок:
- доля отказов на тесте задана заранее (10 %), а не следует из калибровки;
- порог посчитан по тестовым данным;
- шкала порога — CSLS после k-reciprocal; к сырому косинусу сервиса
  (порог 0.8025) не применима, и наоборот.

Проверка: `python research/audit_main_artifacts.py` →
`quantile_0_10_of_postprocess_top1_on_test`, `refused_at_that_quantile`.

## 4. `service_path*.json` и слайды порога — от старой head-only модели

- `research/service_path_extras.py:59, 114` и `research/eval_service_path.py:30`
  грузят `/opt/case17/models/reid_current.pt` и собирают `ReidHead`
  (`service_path_extras.py:117`, `eval_service_path.py:34`) — это head-only
  модель (`weights/reid_current.pt`, 5 МБ), не студент.
- `research/results/service_path_extras.json`: `prenorm.mAP = 64.54`,
  `refusal_deployment.balanced.threshold = 0.5114`, `strict = 0.6354`,
  `auc = 0.972`. Эти же числа в `docs/SUBMISSION_README.md:228` и `:326`
  подписаны как «прежняя head-only модель».
- `research/results/service_path.json`: `ranking.mAP = 61.52`
  (сырой вход в голову, то же, что `SUBMISSION_README.md:267`). На этот файл
  ссылаются `SUBMISSION_README.md:538, 551` как на замер производительности
  решения.
- `research/make_pitch_assets.py:228–236` (слайд `05_refusal.png`) и `:325–326`
  (`12_user_journey.png`) берут пороги из `service_path_extras.json`.
  На закоммиченном `research/pitch_assets/05_refusal.png` подписаны
  **0.5114 / 0.6354, AUC 0.9720**, в README и сервисе — **0.8025 / 0.8465**.
  Последнее изменение обоих файлов — `7e95829` (25.09), до перехода на студента
  (`322a6e5`, 27.09).

Проверка:
`git show origin/main:research/results/service_path_extras.json | python -m json.tool | head -40`.

## 5. `docker compose up` без `.env` поднимает не студента

`docker-compose.yml` (main), значения по умолчанию, если `.env` нет:

| Строка | Переменная | По умолчанию | В `.env.example` |
|---|---|---|---|
| 31, 40 | `EMBED_MODEL` | `resnet18` | `reid:vit_base_patch16_siglip_224.webli` (стр. 18) |
| 43 | `EMBED_TTA_FLIP` | `1` | `0` (стр. 27) |
| 45 | `SEARCH_THRESHOLD` | `0.75` | `0.8025` (стр. 42) |
| 49 | `SEARCH_THRESHOLD_STRICT` | `0.8544` | `0.8465` (стр. 45) |
| 56 | `REID_WEIGHTS` | `/models/reid_current.pt` | `/models/siglip2_distill_ensemble_ep28.pt` (стр. 21) |
| 60 | `REID_TF_MODE` | пусто | `squash` (стр. 22) |
| 63 | `MODELS_HOST_DIR` | `/opt/case17/models` | `./weights` (стр. 61) |

`api/app/config.py:23, 28` повторяет эти умолчания (`resnet18`, TTA включён).
Студент поднимается только после `cp .env.example .env` (`README.md:147`).

В `origin/glebmain` (`a2606b2`) умолчания те же (строки сдвинуты на +8:
48, 51, 53, 64, 68), плюс добавлено обязательное резервирование GPU NVIDIA
(`docker-compose.yml:40`) и проверка CUDA-сборки torch в `api/Dockerfile`.

Проверка:
`git show origin/main:docker-compose.yml | grep -n -E 'EMBED_MODEL|EMBED_TTA_FLIP|SEARCH_THRESHOLD|REID_WEIGHTS|REID_TF_MODE|MODELS_HOST_DIR'`.

## 6. Замер: compose без `.env` против студента на сдаточном тесте

Весь сдаточный тест (1110 query × 750 gallery), CPU, кропы
`crops/{query,gallery}` — те же, из которых собираются артефакты. Обе модели
построены кодом сервиса (`api/app/models/registry.build_model`), окружение
профиля прочитано из `docker-compose.yml` и `.env.example`
(`research/service_embeddings.py`). Сырой косинус, как в `/v1/search`.

| | compose без `.env` | студент (`.env.example`) |
|---|---:|---:|
| Модель | `timm:resnet18` (ImageNet, без ReID-обучения) | `reid:` SigLIP 2 + голова, squash |
| TTA-отражение | нет: `EMBED_TTA_FLIP=1`, но `TimmEmbeddingModel` его не применяет | нет (`EMBED_TTA_FLIP=0`) |
| top-1 косинус: min / p10 / медиана | 0.740 / 0.873 / 0.945 | 0.469 / 0.820 / 0.966 |
| Порог / принято / отказов | 0.75 / **1109** / **1** | 0.8025 / 1015 / 95 |

Расхождение ответов:

| Пара | top-1 совпадает | пересечение top-10 |
|---|---:|---:|
| compose без `.env` ↔ студент | 80.36 % | 0.165 |
| compose без `.env` ↔ `submission.csv` | 66.31 % | 0.153 |
| студент (сервис) ↔ `embeddings.npy` (сырой косинус) | 99.82 % | 0.978 |
| студент (сервис) ↔ `submission.csv` | 79.73 % | 0.669 |

- Корреляция Спирмена строк сходства compose ↔ студент: среднее 0.132.
- Решение «принять / отказать» совпадает у 91.53 % запросов; все 94
  расхождения — compose принимает, студент отказывает. Ещё у 133 запросов
  оба принимают, но с разным top-1.
- Векторы двух профилей напрямую не сравнимы (разные пространства); вектор
  студента из сервиса против `artifacts/embeddings.npy` на тех же кадрах:
  косинус min 0.986197, p01 0.996972, медиана 0.999864; ≥ 0.9999 у 38.4 %
  строк, ≥ 0.999 у 92.9 %. Разбор — шаг 5 (паритет).

Вывод из чисел, без оценки: высокое совпадение top-1 при пересечении top-10
0.165 означает, что top-1 на сдаточном тесте определяют почти-дубли (соседний
кадр той же машины с той же камеры, их находит и ImageNet-resnet18), а
дальше ранжирования расходятся. Профиль compose без `.env` на этом тесте
отказывает 1 раз из 1110, то есть TNR на запросах без пары был бы около нуля.

Проверка (≈3 мин на CPU, векторы кэшируются вне репозитория):

```bash
python research/compare_service_profiles.py --data-dir <папка с test_*.csv и crops/> \
  --compose-ref 0924fb2 --json research/honest_eval_results/service_profiles_test.json
```

`--compose-ref 0924fb2` берёт `docker-compose.yml` из main на момент аудита: в
`origin/glebmain` `6837340` (16.10, 23:28) умолчания compose уже переведены на
студента (`EMBED_MODEL=reid:…`, `EMBED_TTA_FLIP=0`, `SEARCH_THRESHOLD=0.8025`,
`REID_WEIGHTS=/models/siglip2_distill_ensemble_ep28.pt`, `REID_TF_MODE=squash`,
`MODELS_HOST_DIR=./weights`), и без этого флага на свежей ветке оба профиля
совпадут.

```bash
git show origin/glebmain:docker-compose.yml | grep -n -E 'EMBED_MODEL|EMBED_TTA_FLIP|SEARCH_THRESHOLD|REID_WEIGHTS|REID_TF_MODE'
```
