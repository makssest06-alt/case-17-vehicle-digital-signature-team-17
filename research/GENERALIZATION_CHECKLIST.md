# Проверка генерализации перед новым обучением

Организаторы зафиксировали падение mAP примерно с `0.81` на сдаточной
выборке до `0.32` на дополнительной чистой. До подбора модели нужно ответить
на два вопроса: наша локальная валидация герметична и измеряем ли мы тем же
путём, что и сервис/организатор.

## 1. Аудит split без утечек

На машине с датасетом:

```bash
python research/audit_generalization.py \
  --source-csv /opt/case17/data/crops/crops_train.csv \
  --split-csv research/val_split.csv \
  --images-dir /opt/case17/data/crops/train \
  --json research/results/generalization_audit.json \
  --strict
```

`OK` означает:

- `vehicle_id` holdout не пересекаются с train;
- каждый `image_id`, `vehicle_id` и `camera_id` split соответствует source CSV;
- нет одинаковых файлов между train и holdout;
- указано число query, у которых есть релевантный кадр **с другой камеры**.

Если обучали не на complement этого split, передайте фактический CSV обучения:

```bash
--training-csv /путь/к/rows_used_for_training.csv
```

При `FAIL` не сравнивать mAP и не запускать новое обучение: сначала исправить
сплит. При warning о query без пары исключать их из mAP/Rank, но оставить для
проверки режима отказа.

## 2. Абляция постобработки

На одном и том же split сравнить raw cosine, query expansion, CSLS и
k-reciprocal. Параметры из `artifacts/postprocess.json` были подобраны на
малой локальной gallery и могут ломаться на чистой расширенной выборке.

Главная гипотеза: рост локальной метрики даёт трандуктивная пересортировка, а
не более устойчивый эмбеддинг. Поэтому нельзя менять параметры postprocess по
одному split или переносить их в онлайн-поиск без отдельной проверки.

## 3. Паритет сервиса и артефактов

Перед любым замером зафиксировать в отчёте:

- SHA-256 checkpoint;
- `REID_TF_MODE`, padding bbox и input size;
- `EMBED_TTA_FLIP`;
- используется ли postprocess;
- версию Git (`git rev-parse HEAD`);
- состав gallery/query и хэш split.

Сравнение модели проводить тем же `ReidCheckpointModel`, который использует
API. Метрика, полученная старым `.npy`, не доказывает качество service path.

## 4. Режим отказа

Отдельный calibration split не должен пересекаться с обучением, подбором
postprocess и итоговым holdout. Цель — выбрать максимум TNR при заранее
зафиксированном минимальном recall, а не максимизировать TNR ценой почти
полного отказа.
