# Модель работы команды

## Общий Git/SourceCraft

Использовать один выданный репозиторий как источник истины. `main` защищён; изменения только через PR и минимум один review. Ветки: `feat/data-validator`, `feat/reid-baseline`, `feat/retrieval`, `feat/api`, `feat/infra-docs`.

В ML PR обязательны dataset version, split, seed, config, git SHA и метрики до/после. Dataset, checkpoints, пароли и ключи в Git не хранить.

## Хранилища

| Объект | Место |
|---|---|
| Код/config/docs | SourceCraft Git |
| Dataset | локально + Object Storage, не Git |
| Checkpoints/logs | Object Storage/Model Registry |
| Секреты | Secret Manager/локальный `.env` |
| Финальный release | Git tag + архив |

## Yandex Cloud и аренда GPU

После промокода создать каталог, bucket Object Storage, сервисный аккаунт с минимальными правами, GPU VM/DataSphere, budget alerts и лимиты. Один человек отвечает за billing/GPU shutdown, второй имеет emergency-доступ.

Lifecycle GPU: `allocate → provision → pull immutable dataset → run tracked job → upload checkpoint/logs → benchmark → stop/release`.

Каждый запуск маркировать `RUN_ID`, `DATA_VERSION`, `CONFIG`, `SEED`, `GIT_SHA`; результаты отправлять в Object Storage до освобождения VM.

## Роли

1. ML Lead: split, evaluator, threshold, model selection.
2. Data/GPU: audit, dataloader, augmentations, cloud jobs, storage/cost.
3. Model/Retrieval: backbones, embeddings, FlatIP/HNSW/IVF-PQ, reranking, submission.
4. Backend/UI: FastAPI, Swagger, UI, debug Grad-CAM.
5. MLOps/Docs: CI, Docker, offline test, README, Model Card, presentation.

## Ритм

- ежедневный 15-минутный sync: result/blocker/next;
- два раза в неделю review метрик;
- любая модельная правка начинается с гипотезы;
- дорогой GPU job запускается только с ожидаемым критерием успеха;
- последние 24–48 часов — code freeze и только bugfix/release.
