# Абляция постобработки

> **Исторический документ (прежний студент).** Трансдуктивная постобработка (qq-expand → k-reciprocal → CSLS)
> в решении не используется — её запрещает ответ Q38; действующий профиль — корневой `README.md`, раздел 4.2.

Организаторы сообщили о падении примерно `0.81 → 0.32` на чистой расширенной
выборке. Один из первых дешёвых тестов — проверить, не даёт ли локальный
прирост трандуктивный pipeline, который использует всю query/gallery:

```text
raw cosine → query expansion → CSLS → k-reciprocal → CSLS
```

Скрипт `ablate_postprocess.py` сравнивает все этапы на одном ground truth и
использует ту же junk-фильтрацию, что официальный evaluator:

```bash
python research/ablate_postprocess.py \
  --embeddings artifacts/embeddings.npy \
  --index artifacts/embeddings_index.csv \
  --ground-truth /path/to/test_ground_truth.csv \
  --params-json artifacts/postprocess.json \
  --json research/results/postprocess_ablation.json
```

Интерпретация:

- `raw` заметно ниже `full`, а `full` падает на чистом наборе — хрупкость
  постобработки является вероятной причиной;
- `raw` и `full` оба низкие — проблема в эмбеддинге, кропе, доменном сдвиге или
  протоколе данных;
- `qq`/`csls` лучше `raw`, но `rerank` хуже — отключить k-reciprocal;
- сравнение делать на нескольких holdout-сплитах, не оптимизировать параметры по
  единственному набору.
