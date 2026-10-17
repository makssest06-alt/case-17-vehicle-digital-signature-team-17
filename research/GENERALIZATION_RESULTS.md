# Generalization audit results

Run on `team17/finals-student-gpu-handoff` after syncing `glebmain`.
No training and no production changes were performed.

## Split and leakage

Command:

```bash
python research/audit_generalization.py \
  --source-csv /path/to/dataset_extracted/train.csv \
  --split-csv /path/to/research/ensemble/val_split.csv \
  --training-csv /path/to/improve_models/distill_train_rows.csv \
  --images-dir /path/to/dataset_extracted/images --strict
```

Results:

- source: 9556 rows, 1541 vehicle IDs;
- actual training CSV: 8644 rows, 1391 vehicle IDs;
- holdout: 912 rows, 150 vehicle IDs, 361 gallery / 551 query;
- training↔holdout `vehicle_id` overlap: 0;
- training↔holdout exact image duplicates: **5**;
- all expected JPEGs exist.

The identity split is correct for `distill_train_rows.csv`, but the five exact
duplicates contaminate the holdout. Local headline scores must be treated as
optimistic until this is removed.

## Small gallery ablation

Existing student embeddings, 551 query / 361 gallery, bbox already extracted:

| Ranking | mAP@10 | Rank-1 | Rank-5 |
|---|---:|---:|---:|
| raw cosine | 0.8581 | 0.8221 | 0.9238 |
| query expansion | 0.8709 | 0.8367 | 0.9401 |
| gallery-side CSLS | 0.8884 | 0.8566 | 0.9456 |
| k-reciprocal | 0.8820 | 0.8403 | 0.9601 |
| old full pipeline | **0.8965** | **0.8639** | **0.9637** |

The last number is the old research pipeline, not a compliant production result:
it includes query-query expansion and joint reranking. It cannot be used as
evidence that the online service will achieve the same score.

## Expanded-gallery stress proxy

This does **not** reproduce the organizer's separate hidden clean dataset. It is
a **source-domain stress proxy** built from available data:

- remove the five exact train↔holdout duplicate pairs;
- keep 550 holdout queries;
- use **357 cleaned holdout gallery rows + 8639 clean-training rows**;
- total gallery: 8996;
- extract embeddings with the unchanged student, squash 224, no TTA.

The 8639 clean-training rows were used during training, but their vehicle IDs
are disjoint from all holdout query vehicle IDs, so they are negative
source-domain distractors rather than positive leakage. This proxy is not
organizer data and cannot establish transfer to the organizer's different
dataset.

Results:

| Ranking | mAP@10 | Rank-1 | Rank-5 |
|---|---:|---:|---:|
| raw cosine | 0.5567 | 0.5267 | 0.6888 |
| query expansion | 0.6820 | 0.6924 | 0.7680 |
| gallery-side CSLS | 0.6072 | 0.6041 | 0.7127 |
| k-reciprocal | **0.7187** | **0.7090** | **0.7993** |
| old full pipeline | 0.7044 | 0.7090 | 0.7716 |

Exactly seven queries have no valid cross-camera positive and are excluded from
ranking metrics (543 scored). The exact score is a source-domain stress-proxy
result, not organizer data or the hidden organizer split, and it shows that the
local score is not stable as gallery size changes. The k-reciprocal row is a
research diagnostic; the implementation joins query/gallery neighborhoods and
is not automatically compliant with the streaming final protocol.

The diagnostic postprocessing tables above remove junk from the full score
matrix before selecting top-k, as implemented by `research/ablate_postprocess.py`.
The official CSV evaluator instead receives exactly the submitted top-10 IDs
and removes junk from those IDs without backfilling; these protocols must not
be conflated.

## Refusal calibration

Using current small validation embeddings and controlled 50% unknown IDs:

| Protocol | target recall | threshold | recall | TNR | F1 |
|---|---:|---:|---:|---:|---:|
| deployment, no camera filter | 0.70 | 0.8616 | 0.8953 | 1.0000 | 0.9448 |
| cross-camera filter | 0.70 | 0.6470 | 0.7004 | 0.8175 | 0.7447 |

The old `0.8025` threshold was calibrated on a different distribution and is not
evidence for the clean-like cross-camera operating point. The cross-camera row
is the relevant warning: threshold calibration must match the score distribution
used by the evaluated service and the desired recall, not maximize TNR alone.

## Current conclusion

The evidence supports a real generalization/gallery-shift problem, but cannot
separate representation failure from postprocessing failure on the hidden set.
The first practical ranking candidate for a larger gallery is raw or k-reciprocal
without the old CSLS stage; this must be tested against the team's declared
streaming constraints before integration. No new model training should start
until Gleb reviews these numbers and the exact hidden-test proxy is agreed.
