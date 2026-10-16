# Root-cause analysis after organizer feedback

Дата разбора: 16.10.2026. Источники:

- `Загрузки/ЛЦТ2026_кейс17_доработки_по_итогам_встречи.md`;
- `Загрузки/ЛЦТ2026_кейс17_идеи_улучшения.md`;
- read-only audit of root service, artifacts and training logs;
- rejected Sub-center experiment in `training/RESULTS.md`.

## Что подтверждено встречей

Организатор сообщил о двух разных результатах: примерно `0.81 mAP` на выборке,
на которую сдавался сабмит, и примерно `0.32 mAP` на дополнительной «чистой»
выборке без нерелевантных пар. Точные цифры, состав выборок и timestamp требуют
подтверждения: это автоматическая расшифровка, в документе есть несовпадение
таймкодов. Поэтому пока это сигнал о переносимости, а не доказанная причина.

Также сообщено:

- recall режима отказа был ниже 10%, TNR высокий за счёт чрезмерных отказов;
- GPU throughput рос слабо: около 42 FPS при batch 32, ожидается >100 после
  нормальной реализации GPU-пути;
- нужны структурные логи, загрузка модели в lifecycle приложения, а не на module
  import, и проверка freeze весов.

Плюсы, которые сохраняем: одна модель 362 МБ, дистилляция, SigLIP+BNNeck,
двухкомпонентный loss, режим отказа, сильный TNR и запас памяти.

## Критическое расхождение score-контрактов

В root-решении существуют три разных смысла score:

1. **Онлайн-сервис**: raw cosine через pgvector (`1 - cosine distance`).
2. **Сданный submission**: `qq_expand -> joint k-reciprocal -> CSLS`, причём
   `qq_expand` использует `q @ q.T` — другие query.
3. **Сданный candidates**: postprocessed CSLS score и порог около `0.094672`.

При этом сервисный threshold `0.8025` — порог raw cosine. Эти шкалы нельзя
смешивать. Raw threshold `0.8025` на CSLS/top-1 практически откажет от всех:
CSLS top-1 примерно лежит около `-0.04..0.51`. Это правдоподобно объясняет
recall около нуля, **если** финальная оценка запускала такой смешанный путь.

Но отказ не объясняет падение mAP submission сам по себе: официальный evaluator
считает ranking из `submission.csv`, а `candidates.csv` оценивается отдельно.
Падение mAP требует отдельной причины: protocol shift, wrong profile/checkpoint,
offline/online mismatch, batch-only postprocessing или ошибка построения submission.

## Почему Sub-center не доказал причинность

Sub-center кандидат был хуже raw-cos baseline:

| | baseline | candidate |
|---|---:|---:|
| mAP@10 | 0.857729 | 0.838624 |
| Rank-1 | 0.822142 | 0.803993 |
| Rank-5 | 0.923775 | 0.927405 |

Но это не чистая причинная абляция:

- baseline classifier `[1391,512]`, candidate `[1391,2,512]`;
- Xavier инициализация 3D-тензора дала примерно в 19 раз меньший начальный
  prototype std;
- baseline использовал DataLoader workers=6, candidate workers=2;
- Python-аугментации используют worker RNG; seed=7 не гарантирует одинаковые
  batch/augmentation sequence;
- `amax` маршрутизирует gradient только через winning center;
- measured center cosine показывает, что центры разошлись, но не доказывает,
  что именно это испортило retrieval.

Достоверный вывод: **данная реализация и запуск Sub-center отклонены**. Вывод
«сама идея Sub-center плохая» пока делать нельзя.

## Что проверять до нового обучения

### 1. Зафиксировать чистый протокол

Запросить у организаторов: размеры двух выборок, query/gallery construction,
дистракторы, junk/backfill, точный evaluator и использовался ли reranking.
Пока `0.81` и `0.32` нельзя сравнивать с нашими `0.8577` и `0.8961`.

### 2. Проверить artifact/service parity

На одних crop проверить:

- SHA checkpoint;
- `REID_TF_MODE=squash`, 224, bbox pad 0.10;
- TTA;
- L2 и dim;
- cosine service vs offline == 1;
- raw ranking vs submission ranking.

### 3. Проверить leakage

Проверить train/gallery/query по vehicle_id, image hash и near-duplicate кадрам.
Уже подтверждено для текущего local split: train rows 8644 и val rows 912,
пересечение image_id = 0; дополнительно нужен hash/near-duplicate аудит.

### 4. Провести postprocessing matrix

На одной и той же галерее сравнить raw cosine, CSLS, k-reciprocal и разрешённый
per-query reranking. Отдельно проверить маленькую локальную gallery и expanded
gallery. Не подбирать параметры по сообщённой «чистой» выборке.

### 5. Развести refusal и ranking

Для отказа измерять raw top-1, top1-top2, F1, recall, TNR. Порог выбирать по
целевому recall, а не максимуму TNR/Юдена. Для ranking не использовать confidence
threshold и не смешивать CSLS confidence с raw service threshold.

## Если диагностика подтвердит реальное переобучение

Переобучение лечим не увеличением сложности первым делом. Дешёвый порядок:

1. controlled augmentation: отдельные прогоны night/brightness+JPEG+blur и
   viewpoint/perspective; не включать всё сразу;
2. EMA весов при том же seed/loader;
3. distillation alpha 5/10/25 на честном protocol split;
4. умеренный semi-hard mining — batch-hard уже есть, global aggressive mining
   не нужен;
5. masking номера/фон только после подтверждения shortcut анализом карт/ablation;
6. feature-level distillation только если появятся teacher feature maps;
7. higher resolution после устранения protocol/domain проблем.

### Резервный дорогой план

Если диагностика покажет настоящий domain/generalization gap, можно повторить
путь: переобучить четыре открытые учительские модели примерно по часу каждая,
собрать новый train-only teacher и дистиллировать одного ученика примерно час.
Это не первый шаг, но ресурсно реально и соответствует преимуществу команды.
Перед этим нужно сделать протокол parity и leakage audit, иначе мы снова
дистиллируем и оптимизируем неизвестно что.

## Что сейчас не делать

- не принимать Sub-center checkpoint;
- не добавлять runtime ensemble без отдельного решения команды;
- не усиливать query expansion/k-reciprocal для маленькой локальной метрики;
- не подбирать threshold/postprocessing на единственном split;
- не обучаться на выданном test или организаторской clean sample;
- не менять production до получения точного протокола и подтверждения причины.

## Decision gates

| Наблюдение | Следующее действие |
|---|---|
| service/offline parity не совпадает | сначала исправить pipeline |
| есть train/gallery overlap или near duplicates | пересобрать честный split, старые числа отбросить |
| raw cosine стабилен, rerank рушится на expanded gallery | убрать/ослабить rerank |
| threshold даёт recall<10 на raw service score | перекалибровать refusal под target recall |
| raw cosine тоже падает на честной clean-like выборке | controlled augmentation/EMA/alpha, затем новый teacher/student |
| причина не различима | не добавлять сложность; запросить у организаторов артефакты протокола |
