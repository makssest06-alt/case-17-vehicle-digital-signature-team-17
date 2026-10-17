# Clean Full Retrain Report

## Status

The full retrain was completed locally on train data with five exact
train/holdout JPEG duplicates removed. No test data, test labels, or organizer
clean sample was used for training.

Runtime selection is provisional: `clean_student_a10_ep28.pt` is the current
winner of the two clean-student variants. It is not yet wired into production.

## Dataset cleaning

Source train: 9556 rows / 1541 vehicle IDs.

Holdout: 912 rows / 150 vehicle IDs.

Removed exact byte duplicates:

- 5 holdout rows;
- 5 corresponding train rows.

Final clean train: **8639 rows / 1391 vehicle IDs**.
Final clean holdout: **907 rows**, 550 queries / 357 gallery.
Vehicle-ID overlap is zero.

The old `teacher_emb.npy` had 8644 rows and was not used.

## Full teacher retrain

All four teachers were retrained from public pretrained backbone initialization,
not from old ReID checkpoints:

| Teacher | Transform | Seed | Clean mAP@10 | R1 | R5 |
|---|---|---:|---:|---:|---:|
| SigLIP2 ViT-B/16 | squash | 42 | 0.8134 | 0.7569 | 0.9263 |
| SigLIP1 ViT-B/16 | squash | 42 | 0.7752 | 0.7293 | 0.8840 |
| SigLIP1 ViT-B/16 | timm/base | 42 | 0.7205 | 0.6611 | 0.8600 |
| CLIP ViT-B/16 | squash | 7 | 0.6674 | 0.6077 | 0.8306 |

Recipe: ArcFace + batch-hard triplet, P=8/K=4, cross-camera sampling,
GeomAug/NightAndGlare/RandomErasing, context 0.10, 30 epochs, checkpoint epoch
28. The four train-only embeddings were combined with the historical weights
`0.364/0.232/0.182/0.222` into `clean_teacher_emb.npy` `(8639,512)`.

The new teacher ensemble scores **0.8384 mAP@10** on clean holdout, versus
**0.8341** for the old teacher ensemble. This is a modest but real improvement
under the corrected protocol.

## Student variants

Both students use the same SigLIP2 initialization, clean train rows and clean
teacher embeddings. Only distillation alpha differs.

| Student | Alpha | Clean mAP@10 | R1 | R5 | Expanded-gallery raw | Expanded k-reciprocal |
|---|---:|---:|---:|---:|---:|---:|
| old student | 25, old teacher | 0.8557 | 0.8195 | 0.9227 | 0.5567 | 0.7187 |
| clean student | 25 | 0.8614 | 0.8306 | 0.9337 | 0.5492 | 0.7275 |
| **clean student** | **10** | **0.8666** | **0.8306** | **0.9466** | **0.5585** | **0.7375** |

The `alpha=10` student is provisionally selected. It retains more independent
identity-learning signal than alpha=25 while still benefiting from the new
teacher. It remains one 512D SigLIP2 student at runtime.

## Expanded-gallery stress test

The proxy contains 550 clean queries and 8996 gallery objects: cleaned holdout
gallery plus 8635 train-only distractors. This is not the hidden organizer test,
but it tests gallery-size/domain shift without training on test.

The old full pipeline is unstable here. For alpha=10:

- raw cosine: `0.5585` mAP@10;
- query expansion: `0.6819`;
- gallery-side CSLS: `0.5985`;
- k-reciprocal: **`0.7375`**;
- old full CSLS pipeline: `0.6943`.

For alpha=25, the corresponding raw/k-reciprocal values are `0.5492/0.7275`.
Therefore the next production ranking experiment should test k-reciprocal without
the old final CSLS stage, subject to streaming-protocol review. Query expansion
remains prohibited for the final service because it uses other queries.

## Why student can exceed teacher

The student is not trained only to copy teacher embeddings. Its objective is:

```text
ArcFace + batch-hard Triplet + alpha * (1 - cosine(student, teacher))
```

The teacher is a fixed soft target; the student also optimizes identity
separation and hard positives/negatives. Distillation can therefore regularize
training while allowing the student to improve retrieval geometry. This is why
clean alpha=10 (`0.8666`) can exceed clean teacher ensemble (`0.8384`).

## Model files

Local files:

```text
/home/l1zard/diskD/bm_improve/clean_models/
  siglip2_clean_retrain_ep28.pt
  siglip1_clean_ep28.pt
  siglip1_base_clean_ep28.pt
  clip_clean_ep28.pt
  ensemble/clean_teacher_emb.npy
/home/l1zard/diskD/bm_improve/clean_student/
  clean_student_a25_ep28.pt
  clean_student_a10_ep28.pt
```

The five epoch-28 files total about 1.8 GB. They were added to the current
SourceCraft branch as LFS candidates, but SourceCraft rejected the LFS push:
private LFS quota is nearly full (`20 GiB` limit, approximately `21.29 GB`
accounted usage). The commit exists locally but the weight objects are not on
the remote. Text reports and manifests can be pushed; weights require quota
increase or cleanup by the team/SourceCraft support.

## Current decision

- Keep old production student until the team explicitly accepts clean alpha=10.
- Do not deploy the rejected Sub-center student.
- Do not use old query-query expansion/CSLS as final ranking without protocol
  approval; expanded-gallery evidence favors k-reciprocal without final CSLS.
- Do not start larger students until alpha=10 is integrated and measured in the
  actual GPU Docker path.
