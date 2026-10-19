# Полный retrain после проверки генерализации

> **Исторический план.** Retrain выполнен: веса и метрики — `research/clean_student/`, корневой `README.md`,
> разделы 2.3 и 2.7.

## Цель

Повторить путь команды с нуля на train-only данных без пяти exact duplicate
изображений между train и holdout:

```text
clean train rows
  -> 4 public pretrained teachers
  -> honest clean validation
  -> train-only teacher embeddings
  -> embedding ensemble teacher
  -> one-student distillation
  -> one runtime student
```

Runtime ensemble не планируется. Свободная память используется только во время
обучения и вычисления teacher.

## Clean split

`research/clean_split.py` удаляет:

- все 150 holdout vehicle IDs из train;
- 5 train rows, чьи JPEG bytes совпадают с holdout JPEG bytes.

Ожидаемые файлы:

- `clean_train_rows.csv`: 8639 rows;
- `clean_holdout_split.csv`: 907 rows;
- `manifest.json`: exact removed IDs, counts и zero vehicle overlap.

Важно: train-only teacher embeddings должны быть пересчитаны после удаления
дубликатов. Старый `teacher_emb.npy` использовать нельзя: он имеет 8644 rows.

## Teacher recipes

| Teacher | Backbone | Transform | Init | Seed | TTA |
|---|---|---|---|---:|---|
| T1 | SigLIP2 ViT-B/16 | squash 224 | public/local SigLIP2 | 42 | train recipe |
| T2 | SigLIP1 ViT-B/16 | squash 224 | public pretrained | 42 | train recipe |
| T3 | SigLIP1 ViT-B/16 | timm base crop | public pretrained | 42 | train recipe |
| T4 | CLIP ViT-B/16 | squash 224 | public/local CLIP | 7 | train recipe |

All teachers use the existing ReID head, ArcFace, batch-hard triplet,
P=8/K=4, cross-camera sampling, context 0.10 and existing augmentations unless
an experiment log explicitly says otherwise. No test data or organizer clean
sample enters training.

DINOv2 is optional and only starts after the four-teacher baseline. It is not
automatically included because prior DINOv2 student quality was weak.

## Selection protocol

For every teacher:

1. Extract embeddings on `clean_holdout_split.csv`.
2. Evaluate raw cosine with official junk semantics.
3. Evaluate the expanded-gallery proxy separately, clearly marked as stress test.
4. Record Rank-1, Rank-5, mAP@10, full mAP, checkpoint bytes and inference time.

Teacher selection is based on expanded-gallery stability and clean holdout, not
the old contaminated 0.8961 headline. Postprocessing is not tuned on the
selection split.

## New teacher and student

The new teacher is computed only for `clean_train_rows.csv`:

```text
T = L2(sum(weight_i * L2(embedding_i)))
```

First student candidates:

1. same student architecture, alpha=25;
2. same student architecture, alpha=10;
3. EMA variant only if the first two do not overfit.

Do not launch larger students until the new teacher and clean protocol show
that the representation is the limiting factor. A larger model is a fallback,
not a default.

## Acceptance gates

Reject a teacher or student if:

- it is worse on clean holdout and expanded-gallery proxy;
- it relies on test/organizer data;
- it changes transform/TTA silently;
- checkpoint is not loadable by the runtime loader;
- size exceeds 2 GB or runtime violates the performance budget.

The baseline remains the selected model until a candidate passes both clean
generalization and deployment checks.

## Current blockers

- Public SigLIP1/CLIP pretrained weights download successfully, but are not yet
  converted into the exact local initialization checkpoint format.
- Public weights are downloaded into `/tmp/opencode`; they are not committed.
- Teacher training is intentionally not launched until conversion and clean-row
  alignment are checked.
