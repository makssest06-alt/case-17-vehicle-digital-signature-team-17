# Калибровка режима отказа

У нас есть два разных score, которые нельзя смешивать:

- сервис принимает решение по **raw cosine** (`api/app/main.py`);
- `artifacts/postprocess.json` содержит порог для результата
  `qq_expand → k-reciprocal → CSLS`.

Поэтому порог `0.094672` из артефакта нельзя подставлять в сервис. Организатор
также сообщил, что текущая точка даёт recall менее 10%, поэтому выбирать порог
только по максимальному TNR нельзя.

Скрипт `calibrate_refusal.py`:

1. случайно выбирает часть `vehicle_id` как open-set;
2. удаляет их из gallery;
3. получает top-1 raw cosine для known/unknown query;
4. выбирает максимальный TNR при заданном минимальном recall.

Запуск на holdout:

```bash
python research/calibrate_refusal.py \
  --embeddings artifacts/embeddings.npy \
  --index artifacts/embeddings_index.csv \
  --ground-truth /path/to/test_ground_truth.csv \
  --target-recall 0.70 \
  --json research/results/refusal_calibration.json
```

`--cross-camera` нужен только для сравнения с ranking-протоколом жюри. Для
порогa сервиса без `camera_id` используйте запуск без этого флага.

Результат остаётся диагностическим: выбранный порог нельзя считать финальным,
пока он не проверен на чистой выборке организаторов и на нескольких seed.
