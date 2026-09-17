# Чеклист SourceCraft

Для настройки через MCP нужна ссылка или пара `org_slug/repo_slug` выданного репозитория: одного профиля недостаточно, чтобы безопасно выбрать нужный проект.

- [ ] Добавить 5 участников.
- [ ] Защитить `main`, включить обязательный CI и PR review.
- [ ] Создать labels P0/P1/ml/data/retrieval/backend/infra/docs.
- [ ] Создать milestones baseline/quality/production/release.
- [ ] Создать issues по `TEAM_TASKS_DETAILED.md`.
- [ ] Добавить README/CONTRIBUTING/SECURITY/.gitignore.

Минимальный CI: lint/type check, unit tests preprocessing/evaluator, tiny retrieval test, submission schema, Docker build и offline health test.

Не хранить в репозитории dataset.zip, изображения, checkpoints до настройки artifact storage, GPU credentials, Yandex keys и пароль.
