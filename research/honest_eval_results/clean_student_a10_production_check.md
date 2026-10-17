# Clean student alpha=10 — production check

Дата проверки: 17.10.2026.

## Checkpoint

- файл: `weights/clean_student_a10_ep28.pt`;
- архитектура: полный SigLIP2 ReID checkpoint, 512D BNNeck;
- preprocessing: `squash`, 224×224, bbox pad 0.10;
- TTA: выключен;
- SHA-256: `93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce`.

## Runtime smoke

Модель загружена текущим `api/app/models/reid_checkpoint.py` на pinned runtime
(`torch 2.5.1`, `torchvision 0.20.1`, `timm 1.0.12`, CPU fallback):

- output shape: `(2, 512)`;
- dtype: `float32`;
- L2-нормы: `≈1.0`;
- повторный forward при TTA=0 детерминирован;
- разные изображения дают разные векторы.

Production GPU smoke требует CUDA-хост; локальная CPU-проверка подтверждает
совместимость checkpoint и loader, но не заменяет замер на GPU стенда.

## Clean holdout

Протокол: сохранённый split на 150 vehicle ID, удалены 5 exact
train↔holdout duplicate-пар, 550 query / 357 gallery, raw cosine, junk по
`vehicle_id + camera_id`.

| Метрика | Результат |
|---|---:|
| mAP@10 | 0.8661 |
| Rank-1 | 0.8287 |
| Rank-5 | 0.9448 |

Диагностическая калибровка (не новый production threshold):

- deployment/no-camera: `0.87435`, recall `0.8791`, TNR `1.000`;
- cross-camera: `0.66986`, recall `0.7048`, TNR `0.8942`.

Production-профиль: `DBA k=1 + dedup both (bbox OR background) + z100`;
пороги `4.1592/7.3031`. Raw `0.8025/0.8465` и query-expansion/CSLS в сервис
не включаются.
