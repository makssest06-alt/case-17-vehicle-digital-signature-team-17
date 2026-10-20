# Передача наработок для доработки финального ученика

> **Исторический документ передачи (16.10.2026)** — про прежнего студента `siglip2_distill_ensemble_ep28.pt`.
> Сдаётся soup150s2 (суп студентов l150 и s2, `research/make_soup.py`; корневой `README.md`), запасная — a10
> (`weights/clean_student_a10_ep28.pt`); код обучения
> a10 — `training/` и `TEACHER_PIPELINE_README.md`.

База ветки: `main`, коммит `d87517d` от 16.10.2026. Ветка предназначена для
передачи наработок сокоманднику, который отвечает за интеграцию. Основной код,
конфигурация сервиса и сданные `artifacts/` этой веткой не переключаются.

## План и контракт передачи

1. Опубликовать CUDA/FP16-патч как отдельный файл, без применения к сервису.
2. Передать экспериментальный билдер одной модели, результаты и открытые
   тестовые артефакты. Исправить зависимость confidence от других query.
3. Проверить независимость query, формат файлов и загрузку CUDA-модели локально.
4. Опубликовать отдельную ветку без merge и без обновления сервера.

```json
{
  "base_commit": "d87517d",
  "delivery": "research_handoff_not_active_profile",
  "runtime_student_count": 1,
  "weights": "weights/siglip2_distill_ensemble_ep28.pt",
  "existing_api_mem_limit": "1536m",
  "inputs": {"images": "directory", "query": "CSV image_id,x,y,w,h", "gallery": "CSV image_id,x,y,w,h"},
  "outputs": {
    "submission.csv": "header and exactly 10 unique gallery IDs per query",
    "embeddings.npy": "float32 (N_query+N_gallery,512), query order then gallery order",
    "candidates.csv": "query_id,gallery_id,confidence; missing query means refusal",
    "confidence": "fixed monotonic map (CSLS+4)/8; independent of other queries"
  },
  "experiment": {"gallery_blend": 0.95, "csls_k": 5, "raw_cosine_threshold": 0.7625},
  "status": "not_yet_final_submission"
}
```

## Что получилось

Исторические эксперименты на одном val-сплите (551 query / 361 gallery):

| Вариант | mAP@10 в исследовательском расчёте |
|---|---:|
| Текущий ученик, raw cosine | 0.8581 |
| Одна модель + gallery top-1 blend + gallery-side CSLS | 0.8848 |
| Новый student5, лучшая эпоха 26, та же постобработка | 0.8792 |

Student5 **не заменяет** финального ученика: улучшения не получилось. Его веса
не добавлены. В `results/` сохранены результаты исходных прогонов, включая
неудачные эксперименты и ансамбли только как исследовательские ориентиры.
Runtime-ансамбль не предлагается.

Порог raw cosine 0.7625 в симуляции с ~20% open-set дал objective
`0.7*F1+0.3*TNR = 0.9663`. Это подбор на val, не результат закрытого теста
и не независимое подтверждение переносимости порога.

### Что исправили

- GPU-патч: CUDA torch вместо CPU, автоматический выбор устройства,
  перенос входа и saliency на устройство, опциональный FP16 autocast.
- Билдер формирует эмбеддинги в исходном порядке query, затем gallery,
  дополнительно сохраняет индекс строк и SHA-256 весов.
- Отказ определяется raw cosine; confidence соответствует **CSLS-порядку**,
  поскольку официальный evaluator выбирает верхнего кандидата по confidence.
- Удалён глобальный min/max по всем query: теперь confidence использует
  фиксированную монотонную шкалу. Добавление другого query не меняет ответ.
- Сборщик требует >=10 gallery и проверяет уникальность ID и конечность векторов.

## GPU-патч

`gpu.patch` удалён 18.10: его правки (torch 2.5.1+cu124, FP16, резервирование NVIDIA в compose) уже
в основной ветке, повторно патч не применяется. Без NVIDIA — `docker-compose.cpu.yml` (корневой README).

## Билдер тестовых артефактов (экспериментальный)

В окружении API-зависимостей, из корня проекта:

```bash
python research/finals_handoff/build_final_artifacts.py \
  --images /path/to/images --query /path/to/test_query.csv \
  --gallery /path/to/test_gallery.csv \
  --weights weights/siglip2_distill_ensemble_ep28.pt --out /tmp/final-artifacts
python research/finals_handoff/check_handoff.py
```

Веса `siglip2_distill_ensemble_ep28.pt` (прежний студент) из репозитория убраны 18.10 — взять из истории
(`git show backup/glebmain-2026-10-18:weights/siglip2_distill_ensemble_ep28.pt` + `git lfs pull`). Пример
результата `open_test/` удалён 18.10, чтобы не путался с корневыми `artifacts/` (сдаваемые — там, модель a10).

### Ограничения, обнаруженные после чтения первоисточников

- Q38 явно запрещает query expansion и объясняет это использованием других
  query. Gallery-only blend независим от других query, но отдельного прямого
  разрешения на него нет. Это эксперимент, не безусловно одобренная финальная схема.
- Старый исследовательский расчёт удалял junk из полного ранжирования перед
  top-10. Официальный скрипт удаляет junk только из переданных 10 ID: утраченные
  слоты не заполняются. Поэтому таблица выше должна быть повторно проверена
  **по записанному submission.csv**, а не переноситься в итоговую презентацию.
- Старый внутренний `research/evaluate.py` удаляет всю камеру и не эквивалентен
  официальному. Для новых замеров нужен скрипт организаторов.
- Старый валидатор не завершился успешно: было 20 OK / 5 FAIL, включая
  невозможность воспроизвести постобработку. «Все артефакты полностью валидны»
  было слишком сильным утверждением.
- Ночные измерения не подтверждают отсутствие OOM у CUDA-образа. Исторический
  CPU RSS ученика ~795 МиБ не является замером GPU-профиля.

## Дополнительный одиночный training experiment

Проверен Sub-center ArcFace с двумя центрами на класс для одного студента.
Тесты головы, smoke-тренировка и полный run завершились. По официальному
evaluator на raw-cosine validation mAP@10 просел на 0.0191, Rank-1 на 0.0181;
кандидат отклонён, production baseline не меняется. Детали и command line:
`training/RESULTS.md`, контракт: `training/experiment.json`. Checkpoint кандидата
не включён, поскольку критерий качества не пройден.

## Проверки перед публикацией

- `check_handoff.py` — PASS: single/batch/permutation query-independent,
  gallery 2/5/6/10 конечны, gallery 0/1 отвергаются, формат и отказы корректны.
- Новый полный прогон билдера на открытом тесте через патч на RTX 4060 — PASS,
  1036 accepted / 74 refused; не загружаются дополнительные модели.
- CPU FP32 vs GPU FP16 на 8 crop: min cosine 0.99999857, mean 0.99999905.
- Saliency с переданным target: PASS, 196 конечных значений, сетка 14×14.
  В ночном патче target оставался на CPU — это исправлено в публикуемом патче.
- `py_compile` билдера, проверочного скрипта и патченого загрузчика — PASS.
- Докер-сборка/лимит RAM/скорость A5000 пока не проверены.
- Ревью Python и независимое code-review: блокирующие замечания устранены.
  SHA весов в metadata совпадает с LFS OID уже существующего ученика; новые
  веса не загружаются. Файл `.pt` в новом worktree пока является LFS-указателем.
