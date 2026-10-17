# Что поднимает сервис без `.env` и объясняет ли это 0.32 у экспертов

Все модели меряются одним протоколом `research/honest_eval.py --suite`: 5 фолдов
по 150 машинам holdout, 25 % машин фолда без пары, метрики — эталонный
`research/official_evaluate.py`, сырой косинус, как отвечает `/v1/search`. Модель
каждый раз строится кодом сервиса (`api/app/models/registry.build_model`) с
окружением соответствующего профиля.

## Какие модели на самом деле поднимаются

| Конфигурация | Что грузит сервис (по коду) | Порог |
|---|---|---:|
| **compose без `.env`** (`docker-compose.yml` main `0924fb2`, так же на момент сдачи — `b5af9af`…`1df255c`) | `EMBED_MODEL=resnet18` → `TimmEmbeddingModel`: ImageNet resnet18 **без головы**. `REID_WEIGHTS=/models/reid_current.pt` и `EMBED_TTA_FLIP=1` в compose есть, но `registry.build_model` передаёт веса только моделям `reid:`, а `TimmEmbeddingModel` отражение не делает | 0.75 |
| **старая голова + TTA** (`.env.example` с 18.09 по 25.09: `55034b6`, `8aadb47`, `fa04254`, `f3992f1`; до сих пор в ветках `feat/demo-ui`, `feat/research-baselines`) | `reid:vit_base_patch16_siglip_224.webli` + `weights/reid_current.pt` (head-only SigLIP, 5 МБ, лежит в git без LFS) + отражение | 0.509–0.5114 |
| **студент** (`.env.example` с 27.09, `b5af9af`) | `reid:` + `weights/siglip2_distill_ensemble_ep28.pt` (Git LFS, 362 МБ), `squash`, без TTA | 0.8025 |

Чем было бы «resnet18 + старая голова + TTA», как в постановке, такой
конфигурации в сервисе нет: умолчания compose дают голый resnet18, а старая
голова с TTA — это `.env.example` 18–25.09.

## Числа

`research/honest_eval_report.py --compare student compose_default old_head_tta`

### mAP@10 (сырой косинус, режим эталона)

| Галерея | студент | compose без `.env` (resnet18) | старая голова + TTA |
|---|---:|---:|---:|
| small (машины фолда, ~54 кадра) | 0.958 ± 0.029 | 0.199 ± 0.049 | 0.838 ± 0.084 |
| 2.2k (+ весь тест) | 0.704 ± 0.093 | 0.030 ± 0.010 | **0.409 ± 0.070** |
| 8996 (состав команды) | 0.553 | 0.018 | **0.251** |
| 10.8k (+ тест и обучающие) | 0.559 ± 0.100 | 0.017 ± 0.009 | **0.236 ± 0.059** |

Старый протокол (361 / 551, одна галерея): студент 0.857, resnet18 0.068.

### Отказ, 2.2k, режим «без почти-дублей»: Recall / TNR

Порог вне скобок — тот, что поднимает конфигурация; для старой головы — оба
порога её истории (`refusal_folds_old_head_tta.json`).

| Модель, порог | Recall | TNR |
|---|---:|---:|
| студент, 0.8025 | 0.340 | 0.877 |
| resnet18, 0.75 | 0.920 | **0.000** |
| старая голова, 0.75 (порог compose) | **0.028** | **0.973** |
| старая голова, 0.8025 | 0.005 | 1.000 |
| старая голова, 0.5114 (`.env.example` 25.09) | 0.339 | 0.657 |

## Вывод

- **compose без `.env` (resnet18) 0.32 не объясняет**: mAP@10 0.02–0.03 на
  больших галереях, на порядок ниже, и TNR = 0 (порог 0.75 не отказывает
  никогда), а эксперты видели высокий TNR.
- **Студент 0.32 тоже не объясняет**: 0.55–0.70 на тех же галереях, recall
  отказа 0.34 при 0.8025.
- **Старая head-only голова + TTA воспроизводит картину экспертов**: mAP@10
  0.24–0.41 на больших «чистых» галереях и recall отказа 3 % при TNR 0.97 на
  пороге 0.75 (0.5 % / 1.0 на 0.8025). Это совпадение по двум независимым
  признакам сразу (mAP ≈ 0.32 и «recall < 10 %, TNR высокий»), но не
  доказательство: подтвердить может только то, какой коммит/ветку и `.env`
  запускали эксперты (в логе сервиса строка `reid_checkpoint: …` называет файл весов).
- Отдельный путь к «recall < 10 %» без смены модели — score-контракт: порог
  сервиса 0.8025, приложенный к CSLS-скору `candidates.csv`, отказывает всем
  (`score_contract.md`).

## История `.env.example` (время московское)

`git log --all -p -- .env.example`, ключи модели:

| Коммит | Когда | Модель | Веса | TTA | Порог |
|---|---|---|---|---:|---:|
| `074a3db` | 17.09 13:48 | resnet18 | — | — | 0.75 |
| `7debdd5` | 18.09 07:21 | resnet18 | — | — | 0.75 |
| `55034b6` | 18.09 12:03 | **reid: SigLIP** | **reid_current.pt** | **1** | 0.525 |
| `8aadb47` | 18.09 14:46 | reid: SigLIP | reid_current.pt | 1 | 0.509 |
| `fa04254` | 25.09 14:40 | reid: SigLIP | reid_current.pt | 1 | 0.509 |
| `f3992f1` | 25.09 16:40 | reid: SigLIP | reid_current.pt | 1 | 0.5114 |
| `b5af9af` | 27.09 09:48 | reid: SigLIP 2 студент | siglip2_distill_ensemble_ep28.pt (LFS) | 0 | 0.8025 |

**Старая голова + TTA стояла в `.env.example` с 18.09 12:03 по 27.09 09:48 МСК.**
С этим `.env.example` до сих пор кончаются ветки `feat/demo-ui` и
`feat/research-baselines` (последние коммиты 25.09). Умолчания
`docker-compose.yml` за всё это время не менялись: resnet18, TTA=1, 0.75.

## Что было сдачей

- **Тегов нет ни одного** (`git tag -l`, `git ls-remote --tags origin` — пусто),
  релизов в репозитории нет.
- Сроки по `docs/TASK_DESCRIPTION.md`: разработка 15–29.09, экспертиза 30.09–14.10.
- Коммиты со словами сдачи: `322a6e5` «артефакты сдачи пересобраны
  дистиллированным студентом» (27.09 09:34), `6639ea4` «веса студента в ветке
  через LFS (оффлайн-сдача)» (27.09 10:07), `69bd80c` «корневой README под
  финальное решение» (29.09 13:11).
- **Последний коммит до дедлайна — `1df255c`, 29.09 21:29 МСК**, ветка
  `team17/structure-and-workspace`. В нём `.env.example` = версия `b5af9af`
  (студент, TTA=0, 0.8025), `docker-compose.yml` без `.env` — resnet18.
- **В `main` решения на дедлайн не было**: `docs/TZ_CHECKLIST.md` в том же
  `1df255c` пишет «репозиторий есть, ветки отправлены; слить в `main` через PR —
  за капитаном». В клоне от 17.09 `main` = `Initial commit`; решение в `main`
  появилось 16.10 (`d87517d` поверх `1df255c`). Что было в `main` с 30.09 по 16.10,
  по git не восстановить — видно только в истории ветки на SourceCraft.

**Вывод одной фразой:** скорее всего, эксперты мерили не сданного студента, а
старую head-only голову SigLIP + TTA из `.env.example` до 27.09 (ветки
`feat/demo-ui` / `feat/research-baselines` или старое состояние репозитория) —
только она даёт mAP ≈ 0.25–0.41 на «чистых» больших галереях; recall < 10 % при
высоком TNR даёт и она (при пороге 0.75–0.80), и порог 0.8025 на CSLS-шкале
`candidates.csv`, — подтвердить можно только у организаторов: коммит, `.env` и
строка `reid_checkpoint: …` из лога сервиса.

Воспроизвести:

```bash
python research/honest_eval.py --data-dir <данные> --suite --profile compose_default --compose-ref 0924fb2 --tag compose_default
python research/honest_eval.py --data-dir <данные> --suite --tag old_head_tta \
  --env EMBED_MODEL=reid:vit_base_patch16_siglip_224.webli --env REID_WEIGHTS=<repo>/weights/reid_current.pt \
  --env EMBED_TTA_FLIP=1 --env REID_TF_MODE= --env SEARCH_THRESHOLD=0.75
python research/honest_eval_report.py --compare student compose_default old_head_tta
```
