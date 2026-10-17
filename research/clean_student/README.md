# Clean retrain artifacts

These files are the result of the full retrain after removing five exact JPEG
duplicates between train and holdout. They are **research handoff artifacts**,
not yet wired into the active service configuration.

## Models

- `siglip2_clean_retrain_ep28.pt`: clean retrained SigLIP2 teacher
- `siglip1_clean_ep28.pt`: clean retrained SigLIP1 squash teacher
- `siglip1_base_clean_ep28.pt`: clean retrained SigLIP1 base/timm teacher
- `clip_clean_ep28.pt`: clean retrained CLIP teacher
- `clean_student_a25_ep28.pt`: one clean student distilled from the new four-model ensemble

The student is the only intended runtime model. Teacher weights are retained for
reproducibility and future ablations; do not load them in the production service.

## Clean holdout metrics

Official evaluator, raw cosine, same cleaned holdout (550 query / 357 gallery,
7 open-set queries excluded from ranking):

| Model | mAP@10 | Rank-1 | Rank-5 |
|---|---:|---:|---:|
| old student | 0.8557 | 0.8195 | 0.9227 |
| clean SigLIP2 teacher | 0.8134 | 0.7569 | 0.9263 |
| clean SigLIP1 squash | 0.7752 | 0.7293 | 0.8840 |
| clean SigLIP1 base | 0.7205 | 0.6611 | 0.8600 |
| clean CLIP | 0.6674 | 0.6077 | 0.8306 |
| old teacher ensemble | 0.8341 | 0.7827 | 0.9411 |
| clean teacher ensemble | 0.8384 | 0.7993 | 0.9319 |
| **clean student alpha=25** | **0.8614** | **0.8306** | **0.9337** |

The student improves the old student on this clean holdout. It is not yet
accepted for production until the expanded-gallery stress protocol and runtime
checks are complete.

## Expanded-gallery stress result

Gallery: 357 cleaned holdout gallery images plus 8635 train-only distractors,
8996 total. No training uses this gallery. The clean student gets raw cosine
mAP@10 `0.5492`; fixed k-reciprocal without the old CSLS stage gets `0.7275`.
The old full CSLS pipeline gets `0.6986`. This confirms the postprocessing stage
is a separate generalization bottleneck.

## Provenance

All models were trained with public pretrained backbones, clean train rows,
identity-disjoint holdout, squash/base transform as named, bbox context 0.10,
ArcFace + batch-hard triplet, and GPU AMP. No test data, test labels or old
teacher embeddings were used.
