# Clean retrain artifacts

These files are the result of the full retrain after removing five exact JPEG
duplicates between train and holdout. The alpha=10 student is the production model of the service
(root `README.md`, sections 2.3 and 2.7); teacher and alpha=25 weights are research artifacts.

The weight files are not stored in this directory: `clean_student_a10_ep28.pt` is
`weights/clean_student_a10_ep28.pt` (Git LFS) and the GitHub Release `weights-a10`; the other
checkpoints are not in the repository (see `manifest.json`).

## Models

- `siglip2_clean_retrain_ep28.pt`: clean retrained SigLIP2 teacher
- `siglip1_clean_ep28.pt`: clean retrained SigLIP1 squash teacher
- `siglip1_base_clean_ep28.pt`: clean retrained SigLIP1 base/timm teacher
- `clip_clean_ep28.pt`: clean retrained CLIP teacher
- `clean_student_a25_ep28.pt`: clean student alpha=25
- `clean_student_a10_ep28.pt`: selected clean student alpha=10 (ветка `glebmain`)
- `clean_student_large_150_ep150.pt`: clean student ViT-L/16@256, alpha=10, 150 эпох, seed 7 — компонент супа (в git нет, 1.21 ГБ; ассет GitHub Release `weights-l150`, sha256 в `manifest.json`). Команда обучения — `research/run_clean_student_large.sh`; clean holdout mAP@10 0.9005 / R1 0.8821 / R5 0.9484, expanded 0.6467 (официальный evaluator; логи вне репо, `bm_improve/large_scale/notes.md`)
- `soup150s2.pt`: **production этой ветки** — model soup l150 (seed 7, 150 эпох) + s2 (seed 42, 75 эпох), рецепт `research/make_soup.py` (в git нет, 1.19 ГБ; ассет GitHub Release `weights-soup150s2`, sha256 в `manifest.json`). clean holdout mAP@10 0.9133 / R1 0.9024 / R5 0.9466, expanded 0.6592 (официальный evaluator)

The alpha=10 student is the production runtime model. Teacher weights are retained for
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
| clean student alpha=25 | 0.8614 | 0.8306 | 0.9337 |
| **clean student alpha=10** | **0.8666** | **0.8306** | **0.9466** |

The student improves the old student on this clean holdout. It is the production
model now (thresholds and protocol — `docs/refusal.md`).

## Expanded-gallery stress result

This **source-domain stress proxy** uses 550 queries and 8996 gallery rows:
**357 cleaned holdout gallery rows + 8639 clean-training rows**. The
clean-training rows were used during training and have vehicle IDs disjoint
from holdout queries; this is not organizer data. Exactly 7 queries are
excluded from ranking (543 scored). The alpha=25 clean student gets raw cosine mAP@10
`0.5492`; fixed k-reciprocal without the old CSLS stage gets `0.7275`. The old
full CSLS pipeline gets `0.6986`. This confirms the postprocessing stage is a
separate generalization bottleneck.

For **alpha=25**, the diagnostic postprocessing values are raw `0.5492`,
k-reciprocal `0.7275`, and full CSLS `0.6986`. The selected **alpha=10** has
raw `0.5585`, k-reciprocal `0.7375`, and full CSLS `0.6943`. Diagnostic tables
remove junk from the full score matrix before top-k, per
`research/ablate_postprocess.py`. The official CSV evaluator receives exactly
top-10 IDs and removes junk without backfilling; it does not rerank the full
gallery and refill top-10.

## Provenance

All models were trained with public pretrained backbones, clean train rows,
identity-disjoint holdout, squash/base transform as named, bbox context 0.10,
ArcFace + batch-hard triplet, and GPU AMP. No test data, test labels or old
teacher embeddings were used.
