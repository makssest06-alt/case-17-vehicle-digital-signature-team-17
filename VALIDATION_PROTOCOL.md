# Финальный протокол валидации

## Identity-disjoint split

- 70% vehicle_id — train, 15% — validation, 15% — holdout.
- Identity не пересекаются между частями.
- camera_id используется только для анализа и same-camera exclusion.

## Cross-camera retrieval

В validation query и gallery имеют одинаковый vehicle_id, но релевантными считаются только пары с разными camera_id. Same-camera пары исключаются из AP. В финальном test camera_id отсутствует, поэтому там нельзя фильтровать кандидатов по камере.

## Refusal/calibration

Формируем known-in-gallery queries и unknown-to-gallery queries. Unknown-to-gallery — контролируемая симуляция open-set, а не утверждение о составе скрытого test.

Для каждого query сохраняем `top1_score`, `top2_score`, `margin`, `quality_score`, `is_known`. Сравниваем fixed threshold, threshold по score+margin, logistic и isotonic calibration. Выбираем operating point по F1 с контролем TNR, строим PR curve и фиксируем threshold в config.

## Без leakage

- threshold не обучается на holdout;
- hard-negative mining не использует holdout labels;
- фиксируются `run_id`, `git_sha`, `data_version`, `split_hash`, `seed`.
