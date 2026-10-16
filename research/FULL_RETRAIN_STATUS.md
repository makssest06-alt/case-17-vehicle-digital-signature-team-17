# Full Retrain Status

The full retrain is intentionally staged. The old `teacher_emb.npy` is not used:
it was generated from the old 8644-row training set with five exact train/holdout
duplicate images.

## Prepared

- `research/clean_split.py` produces `clean_train_rows.csv` (8639 rows) and
  `clean_holdout_split.csv` (907 rows), removing all five exact duplicate pairs.
- Public SigLIP1 and CLIP ViT-B/16 pretrained weights downloaded successfully and
  loaded strictly into timm architectures.
- `research/prepare_public_init.py` converts public timm state dicts into the
  local `train_reid.py` init format.
- `research/build_clean_teacher.py` computes a train-only weighted teacher from
  newly retrained checkpoints. It never reads test CSVs.
- The four-teacher recipe is documented in `research/FULL_RETRAIN_PLAN.md`.

## Not yet started

- No clean teacher training has started in this checkpoint. The public weights
  are currently only in `/tmp/opencode`, not in Git.
- No new student has been trained from the clean teacher.

## Guardrails

- Never use the old teacher embeddings after switching to clean rows.
- Evaluate each teacher raw cosine on the clean holdout and expanded-gallery
  stress proxy before selecting it for the new ensemble.
- Keep all runtime output to one student model; the ensemble exists only in the
  training pipeline.
- Do not select checkpoints by the contaminated old 0.8961 score.
