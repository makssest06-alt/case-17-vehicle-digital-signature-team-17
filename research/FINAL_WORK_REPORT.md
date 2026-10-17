# Итоговый отчёт: clean retrain, проверка обобщения и передача команде

**Срез: 17.10.2026. Кандидат для интеграции — `clean_student_a10_ep28.pt`.**
Он даёт `0.866609 mAP@10` на малом clean holdout; на expanded proxy raw-метрика
около `0.557–0.558`. Результат на скрытой «чистой» выборке организатора **не
воспроизведён**. Выбор кандидата предварительный, production-интеграция требует
проверки реального evaluator/GPU-пути и согласованного score-контракта.

## 1. Основания и состояние ветки

Источники:

- `research/CLEAN_RETRAIN_REPORT.md`;
- `research/GENERALIZATION_RESULTS.md`;
- `research/finals_handoff/ROOT_CAUSE_FEEDBACK.md`;
- проверенные уточнения к числам, составу proxy, runtime и артефактам,
  предоставленные для этого отчёта. Они имеют приоритет над округлёнными либо
  устаревшими строками исходных документов.

Текущая ветка: `team17/finals-student-gpu-handoff`; проверенный HEAD:
`6bf85da855801453d5177caa31966448a1b51d2f` (`6bf85da`). База Gleb main:
`a2606b2`; это зафиксированная база, live main может уже отличаться.

При подготовке этого документа обучение и новые бенчмарки не запускались.
API/KNN не менялись, ветки не создавались, commit/push не выполнялись.

## 2. Данные, очистка и протокол

| Набор | Изображения/строки | Vehicle IDs | Query / gallery |
|---|---:|---:|---:|
| Исходный source | 9556 | 1541 | — |
| Исторический train | 8644 | 1391 | — |
| Исторический holdout | 912 | 150 | 551 / 361 |
| **Clean train** | **8639** | **1391** | — |
| **Clean holdout** | **907** | **150** | **550 / 357** |

Между train и holdout нет пересечения `vehicle_id`, но аудит JPEG выявил
**5 пар точных побайтовых train↔holdout дубликатов**. Удалены пять train-строк
и пять holdout-строк. Отсутствие пересечения ID само по себе не гарантировало
отсутствие утечки изображений; полная проверка near-duplicates остаётся отдельной
задачей.

В clean cross-camera ranking учитываются **543 query**, ещё **7 исключены**:
для них нет допустимого положительного gallery-объекта после camera/junk-фильтра.
Ни test, ни test labels, ни clean sample организатора для обучения не использовались.

**Source-domain stress proxy: 550 query / 8996 gallery = 357 очищенных
holdout gallery + 8639 clean-training изображений.** Это точный состав из
`/tmp/opencode/cleanish/split.csv`. Здесь также 543 scored / 7 excluded. Добавленные
изображения уже участвовали в обучении, но их vehicle IDs не пересекаются с
holdout query: это отрицательные source-domain distractors, **не данные
организатора**. Proxy проверяет чувствительность к размеру gallery, а не
перенос на независимый неизвестный домен.

## 3. Четыре clean teacher и train-only дистилляция

Все четыре teacher переобучены **из публичных pretrained backbone**, не из старых
ReID checkpoint. Рецепт: ArcFace + batch-hard triplet, P=8/K=4, cross-camera
sampling, GeomAug/NightAndGlare/RandomErasing, context 0.10, 30 эпох, checkpoint
эпохи 28. Seed 42 у SigLIP-вариантов, 7 у CLIP.

Сравнение на одном малом clean holdout, mAP@10:

| Teacher | Старые веса | Clean retrain |
|---|---:|---:|
| SigLIP2 ViT-B/16, squash, single | 0.825381 | 0.813383 |
| SigLIP1 ViT-B/16, squash | 0.796348 | 0.775224 |
| SigLIP1 ViT-B/16, base | 0.728696 | 0.720524 |
| CLIP ViT-B/16, squash | 0.689510 | 0.667420 |
| **Teacher ensemble** | **0.834075** | **0.838386** |

Отдельные clean teacher слабее старых на этой проверке; ансамбль немного сильнее.
Это наблюдение одного split, а не доказательство улучшения на скрытом тесте.

Новые train-only teacher embeddings объединены историческими весами
`0.364/0.232/0.182/0.222` в `clean_teacher_emb.npy`, форма **8639×512**.
Старый `teacher_emb.npy` на 8644 строки для clean-дистилляции не использовался.
Четыре teacher нужны при подготовке soft targets; runtime-кандидат — **один
SigLIP2 student с 512D embedding**, без runtime-ансамбля.

## 4. Student: малый clean holdout

Метрики retrieval: mAP@10 / Rank-1 / Rank-5; 543 scored / 7 excluded.

| Student | mAP@10 | Rank-1 | Rank-5 |
|---|---:|---:|---:|
| Старый student, alpha=25, старый teacher | 0.855694 | 0.819521 | 0.922652 |
| Clean alpha=2 | 0.825932 | 0.773481 | 0.939227 |
| Clean alpha=6 | 0.839983 | 0.791897 | 0.935543 |
| **Clean alpha=10** | **0.866609** | **0.830571** | **0.946593** |
| Clean alpha=25 | 0.861351 | 0.830571 | 0.933702 |

Alpha=10 — лучший из перечисленных clean-кандидатов по mAP@10 и Rank-5 малого
holdout. Преимущество над старым student по mAP@10: `0.010915` (1.0915 п.п.).
Большая alpha не гарантирует лучшего результата.

**Почему student может превзойти teacher:** его objective —
`ArcFace + batch-hard Triplet + alpha × (1 − cosine(student, teacher))`.
Он одновременно учит identity-разделение и hard positive/negative geometry,
а distillation служит регуляризацией, не требованием буквально повторить teacher.
Поэтому `0.866609` student против `0.838386` teacher ensemble не является
противоречием. Это не отменяет необходимости независимой проверки обобщения.

## 5. Expanded gallery: raw и диагностическая абляция

### Сравнительный raw-прогон

550q / 8996g, 543 scored / 7 excluded; mAP@10:

| Student | Raw cosine |
|---|---:|
| Старый student | 0.556711 |
| Clean alpha=2 | 0.530891 |
| Clean alpha=10 | 0.556907 |
| Clean alpha=25 | 0.547565 |

Alpha=10 практически совпадает со старым student (`+0.000196`), а не переносит
на большую gallery весь выигрыш малого holdout. Это существенное ограничение
текущих свидетельств об обобщении.

### Отдельный прогон postprocessing-абляции

Тот же заявленный размер proxy; эти raw-значения получены в отдельном прогоне
и не подменяют raw-строки сравнительной таблицы выше. Точная причина различий
между прогонами здесь не установлена; эффект стадии следует сравнивать внутри
одной таблицы.

| Ranking | Clean alpha=10, mAP@10 | Clean alpha=25, mAP@10 |
|---|---:|---:|
| Raw cosine | 0.558472 | 0.549222 |
| QQ / query expansion | 0.681877 | 0.678484 |
| Gallery-side CSLS | 0.598525 | 0.610094 |
| Joint k-reciprocal rerank | **0.737479** | **0.727481** |
| Старый full pipeline | 0.694322 | 0.698620 |

Full означает историческую цепочку `QQ → joint k-reciprocal → CSLS`.
В этой абляции финальный CSLS ухудшает результат относительно joint rerank.

Диагностические postprocessing-таблицы выше удаляют junk из полной матрицы
score до выбора top-k — именно так работает `research/ablate_postprocess.py`.
Официальный CSV evaluator получает ровно переданные top-10 ID и удаляет junk
только из этих ID, без backfill; он не ранжирует заново всю gallery и не
дополняет результат до top-10.

**Joint postprocessing — диагностика, не автоматически streaming-compliant
решение.** QQ использует другие query (`q @ q.T`); joint rerank строит совместные
query/gallery-соседства. Их offline-выигрыш нельзя обещать независимому онлайн
запросу. Для production нужен явно согласованный per-query, gallery-only путь,
проверенный тем же evaluator. Старую QE-цепочку в финальный сервис не переносить.

## 6. Почему прежние «88–90%» отличаются

1. **Маленький и загрязнённый holdout.** Историческая gallery — всего 361 объект;
   присутствовали пять точных train↔holdout JPEG-пар. В старой абляции raw был
   `0.8581`, CSLS `0.8884`, full pipeline `0.8965 mAP@10` — это другой,
   оптимистичный локальный протокол.
2. **Старый transductive postprocessing.** Числа около 88–90% включали доступ к
   другим query и совместному графу; онлайн raw cosine этому не эквивалентен.
   На расширенной gallery raw падает примерно до 0.557, а full не всегда лучший.
3. **Несовпадение протокола/evaluator и шкал score.** Размер gallery, camera/junk
   filtering, исключение query без positive, различие diagnostic filtering и
   official top-10 без backfill, crop/TTA и конкретные
   checkpoint/profile должны совпасть. Сервис использовал raw cosine pgvector,
   submission — QQ/joint/CSLS, candidates — postprocessed score. Это разные шкалы.

По расшифровке встречи организатор сообщал примерно `0.81 mAP` на одной выборке
и `0.32 mAP` на дополнительной «чистой», а также recall отказа ниже 10%.
Точный состав выборок, числа и timestamps требуют подтверждения. **Мы не
воспроизвели скрытый результат организатора и не установили единственную причину
его падения.** Наш proxy не отделяет domain shift от ошибок representation,
postprocessing или artifact/service parity на неизвестной выборке.

Смешивание raw threshold `0.8025` с CSLS-score (исторически примерно
`−0.04…0.51` у top-1) могло вызвать массовые отказы, **если** оценка шла по такому
смешанному пути. Это гипотеза для refusal, не объяснение mAP само по себе:
ranking `submission.csv` и refusal `candidates.csv` оцениваются отдельно.
Ранее отклонённый Sub-center запуск не использовать как замену clean student;
его неудача не доказывает непригодность самой идеи Sub-center.

### Train-fit — отдельная проверка

| Модель | mAP@10 | Rank-1 | Rank-5 |
|---|---:|---:|---:|
| Clean alpha=10 | 0.900677 | 0.880962 | 0.966966 |
| Старый student | 0.890241 | 0.871970 | 0.962862 |

Это **подгонка на обучающих данных, не generalization accuracy**. Gallery здесь
другая; напрямую сравнивать эти числа с clean holdout, expanded proxy или
организаторским результатом нельзя. Число 0.900677 не является «90% на clean test».

## 7. Refusal: диагностическая калибровка alpha=10

Clean holdout, контролируемый open-set с 50% unknown IDs, target recall 0.70:

| Протокол | Threshold | Recall | TNR | F1 |
|---|---:|---:|---:|---:|
| Cross-camera | 0.669864 | 0.701107 | 0.894161 | 0.775510 |
| Без camera filter | 0.874347 | 0.875458 | 1.000000 | 0.933594 |

Это **диагностические operating points, не production defaults**. Порог нужно
калибровать после интеграции финального ranking/score-пути под его распределение
и целевой recall. Высокий TNR при массовых отказах сам по себе не означает
полезный refusal; raw и postprocessed thresholds взаимозаменять нельзя.

## 8. Runtime, размер и память

Локальный **RTX 4060, clean alpha=10, FP16, без TTA**:

| Batch | Время на изображение, ms | Throughput, FPS |
|---:|---:|---:|
| 1 | 8.03 | 124.5 |
| 8 | 4.88 | 204.8 |
| 16 | 5.18 | 192.9 |
| 32 | 5.45 | 183.5 |

Это локальный embedding smoke/benchmark, не полный organizer JPEG/decode/API/KNN
end-to-end тест и не гарантированный результат A5000. Проверка восьми изображений
из исходного отчёта: GPU FP16 vs CPU FP32 cosine mean `0.999999`, min `0.999998`.

CPU RSS измерен **исторически только для старого alpha=25**: около **2.06 GB**
в контексте `test_gpu`-скрипта либо около **795 MiB** в sampled old benchmark.
Это разные контексты/способы замера; числа не объединять и не приписывать clean
alpha=10. Подтверждённого RSS нового кандидата в production-пути здесь нет.

Кандидат: **94.2M параметров**, checkpoint **379,734,230 bytes** (около 362 MiB).
SHA-256:
`93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce`.

## 9. Локальные артефакты, LFS и передача Глебу

```text
/home/l1zard/diskD/bm_improve/clean_models/
  siglip2_clean_retrain_ep28.pt
  siglip1_clean_ep28.pt
  siglip1_base_clean_ep28.pt
  clip_clean_ep28.pt
  ensemble/clean_teacher_emb.npy
/home/l1zard/diskD/bm_improve/clean_student/
  clean_student_a10_ep28.pt
  clean_student_a25_ep28.pt
/home/l1zard/diskD/bm_finals_handoff/research/FINAL_WORK_REPORT.md
```

**Clean weights на remote отсутствуют:** предыдущая попытка передачи LFS была
отклонена из-за квоты. Зафиксированное использование — **21,292,773,306 /
21,474,836,480 bytes**, остаток всего 182,063,174 bytes, меньше одного checkpoint
кандидата. Наличие локального commit/LFS pointer не означает наличие весового
объекта на сервере. Для весов требуется отдельная согласованная передача с
проверкой SHA либо освобождение/увеличение квоты; текущая работа push не выполняет.

**Что переносить в Gleb main сейчас: отчёт и текстовые выводы/сведения об
артефактах. Production KNN и старую QE не переносить.** Clean alpha=10 предложен
как кандидат, а не как уже подключённая production-модель.

После решения команды об интеграции проверить checkpoint SHA, squash 224,
bbox context 0.10, TTA-профиль, L2/512D, service/offline cosine parity; затем
lifecycle-загрузку, freeze, структурные логи и полный GPU Docker-путь.
Ranking и refusal должны иметь единый явно определённый score-контракт.
Это задачи принимающей стороны, а не изменения API/KNN в рамках данного отчёта.

## 10. Открытые вопросы

- **Точная clean-выборка организатора неизвестна:** нужны состав query/gallery,
  distractors, camera/junk/backfill, evaluator и точный профиль сданной оценки.
- **Alpha=6/50 не прошли полную generalization-оценку:** для alpha=6 известна
  только приведённая малая clean-holdout строка; expanded/postprocessing-оценки
  alpha=6/50 здесь нет, подтверждённых метрик alpha=50 нет.
- **Нет полного A5000 end-to-end benchmark** с JPEG decode, crop, embedding,
  поиском, сериализацией и реальным сервисным lifecycle.
- **LFS-квота блокирует передачу clean weights:** на remote их нет; требуется
  решение команды по квоте/хранилищу.
