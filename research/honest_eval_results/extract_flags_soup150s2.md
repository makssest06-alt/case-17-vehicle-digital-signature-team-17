# Ускорения extract() и качество: soup150s2

Holdout: 546 запросов, 361 кадров галереи, официальный `ranking_metrics`. Полный цикл `api/app/extractor.py` по кадрам из dataset.zip. Косинус — с вектором того же кадра у эталона. `python research/extract_flags.py ...` (докстринг).

| Вариант | Устройство | FP16 | JPEG draft | Эталон | cos min / среднее | mAP@10 raw | mAP@10 DBA | R1 / R5 DBA | Время, с |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| cuda_fp16 | cuda | True | False | — | — | 0.9126 | 0.9265 | 0.910 / 0.958 | 19.3 |
| cuda_fp16_draft2 | cuda | True | True | cuda_fp16 | 0.99387 / 0.999454 | 0.9094 | 0.9211 | 0.903 / 0.956 | 16.8 |
| cuda_fp16_draft | cuda | True | True | — | — | 0.9094 | 0.9211 | 0.903 / 0.956 | 15.5 |
