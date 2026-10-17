# A10 gallery hubness experiment

Research-only result for `clean_student_a10_ep28.pt`. No production files,
checkpoint settings, or SourceCraft branches are changed by this experiment.

## Reproducibility

- Local branch commit: `9b9ac3724d0107fc02230ac0d6e23e9473970310`
- Checkpoint SHA-256: `93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce`
- Preprocessing: SigLIP2 squash crop, input `224`, bbox padding `0.10`, TTA off
- Train CSV SHA-256: `BD1DF45B052AE9AABB7FD356898E244875F8CAD2F111A3F76977C1B90BCE9268`
- Holdout split SHA-256: `2852844AF0F9585AE3EA80F3653E9B6D203557524D1F715DECF1CFB7EF9A3716`
- Labels: clean holdout only; test gallery is used only as unlabeled distractors
- Protocol: official cross-camera junk filtering, 546 clean holdout queries

## Ranking result

The candidate score is:

```text
raw_cosine(query, gallery_item) - beta * gallery_hubness(gallery_item)
```

`gallery_hubness` is the mean cosine similarity to the gallery item's top-k
gallery neighbours, excluding itself. It uses no other query and therefore does
not have query-expansion leakage. For the unseen-domain proxy (holdout gallery
plus the 750 unlabeled `test_gallery` distractors):

| Ranking | mAP@10 | Rank-1 | Rank-5 |
|---|---:|---:|---:|
| raw cosine | 0.7403 | 0.6777 | 0.8901 |
| hubness, k=5, beta=1 | **0.7588** | **0.7033** | **0.9066** |

The small clean gallery changes only from `0.8678` to `0.8731` mAP@10. This
is a proxy, not a hidden-test score. `test_query` is never added to a gallery;
there are no test labels and no test metric is claimed.

## Refusal

The candidate top-1 is selected by the corrected ranking, but refusal uses the
raw cosine of that selected item. Five vehicle-ID folds with 25% of each fold
removed from the gallery gave these diagnostic means:

| Raw threshold | Recall | TNR | F1 |
|---:|---:|---:|---:|
| 0.70 | 0.986 | 0.727 | 0.953 |
| 0.72 | 0.971 | 0.775 | 0.952 |
| 0.75 | 0.954 | 0.836 | 0.952 |
| 0.80 | 0.940 | 0.933 | 0.958 |

These are calibration diagnostics, not a production default. The final
threshold must be selected again after the exact production gallery and service
path are frozen.

## Decision

Do not integrate this into the API yet. It is a promising **optional**
gallery-side artifact, but it has two unresolved production costs:

1. The hubness vector must be rebuilt after gallery mutations.
2. The implementation must be benchmarked with the actual gallery index and
   memory limit; a full dense gallery score is not acceptable for a million-item
   gallery.

The legacy query-query expansion, CSLS, and joint k-reciprocal pipeline remain
rejected for the online service.
