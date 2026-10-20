# SigLIP2 teacher: clean retrain + ensemble embeddings (ТЗ handoff)

Код учителей: `research/finals_handoff/training/` — `train_reid.py` (учитель
SigLIP2/SigLIP1/CLIP ViT-B/16: `ReidHead`, лоссы
`MarginHead` (ArcFace) + `batch_hard_triplet` из `head_model.py`, сэмплер
`PKBatchSampler` P машин x K кадров cross-camera, loaders инлайн:
`CropDataset`/`NightAndGlare`/`GeomAug`/`RandomErasing`, `FullModel` там же).
Отдельного `data.py` нет — train_reid.py импортирует только `backbones.py`
(реестр; `siglip2_base` = `vit_base_patch16_siglip_224.v2_webli`, timm >=
1.0.15, проверено 1.0.29) и `head_model.py`, оба на ветке, лишних локальных
импортов нет; `extract_emb.py` — экспорт из чекпоинта; `tf_mode` по умолчанию из
args чекпоинта (`resolve_tf_mode`, коммит 7779187), явный конфликтующий — ошибка.

Окружение: python `<VENV>/bin/python`, pin `torch 2.14.1+cu130`,
`timm 1.0.29`; ОБЯЗАТЕЛЬНО `export HF_HUB_OFFLINE=1` и
`export PYTHONPATH=research/finals_handoff/training` — веса timm (SigLIP2/
SigLIP1/CLIP pretrained-бэкбоны и init-чекпоинты siglip2/siglip1/clip_vitb16 в локальном HF-кеше заранее).
Входы-плейсхолдеры: `IMAGES_DIR` JPEG (напр. `<DATA>/images`);
`ROWS_CSV` — clean train rows, 8639 строк / 1391 ТС (clean_split.py, 5 дублей удалены);
`BOXES_CSV` — train.csv (image_id,x,y,w,h); `VAL_IDS` — val_ids.txt;
`HOLDOUT_CSV` — clean holdout (550 query / 357 gallery); `CLEAN_MODELS` — dir.

## Шаг 1 — учитель SigLIP2 (alpha НЕ применяется: только ArcFace+triplet)

    python research/finals_handoff/training/train_reid.py \
      --backbone siglip2_base --init-from siglip2_init.pt \
      --images-dir IMAGES_DIR --train-csv ROWS_CSV --val-ids VAL_IDS \
      --models-dir CLEAN_MODELS --tag siglip2_clean_retrain \
      --epochs 30 --save-epochs 28 --lr-head 3e-4 --lr-backbone 1e-5 \
      --wd 5e-4 --P 8 --K 4 --warmup 3 --workers 4 --amp --seed 42 \
      --context 0.10 --geom-aug --cross-cam --tf-mode squash

Выход: `CLEAN_MODELS/siglip2_clean_retrain_ep28.pt` (30 эпох, чекпоинт 28-й;
5190 шагов ≈ 23.5 мин на RTX 4060, лог `bm_improve/clean_logs/`). Остальные 3
учителя — тот же `train_teacher()` (в run_clean_retrain.sh): `siglip1_clean`
(siglip_base, squash+geom+cross-cam, seed 42), `siglip1_base_clean` (siglip_base,
timm-create, без aug, seed 42), `clip_clean` (clip_vitb16, seed 7); после
каждого — `extract_emb.py --ckpt ..._ep28.pt (+ --val-csv HOLDOUT_CSV
--boxes-csv BOXES_CSV --images-dir IMAGES_DIR --context 0.10 --bs 32)` и
`evaluate_pair.py` (артефакты `emb_<tag>_qry/gal.npy`; ансамбль 0.8384 mAP@10).

## Шаг 2 — clean_teacher_emb.npy

Собирает `research/build_clean_teacher.py` из 4 учителей:

    python research/build_clean_teacher.py \
      --config clean_teacher_config.json --rows ROWS_CSV \
      --images-dir IMAGES_DIR --out clean_teacher_emb.npy --batch-size 24

Веса/tf_mode/tta ЧИТАЮТСЯ ИЗ CONFIG (реальный `bm_improve/clean_models/ensemble/`
`clean_teacher_config.json`; пример — `research/clean_teacher_config.example.json`):
0.364 siglip2_clean(squash), 0.232 siglip1_clean(squash),
0.182 siglip1_base_clean(timm), 0.222 clip_clean(squash); tta=true, context 0.10,
L2-норм. Результат (8639,512) float32, 17 692 800 байт, sha256
`4ec0e9dc3c57fd24e841c356f25500d70d63fa5c00784b702ce24859b60e86d1`.

## Чекпоинты: только локально, LFS-quota на remote

Веса ТОЛЬКО в `<MODELS>/`; LFS push на SourceCraft
отклонён по квоте (21 292 773 306 / 21 474 836 480 байт, свободно ~182 MB).

| файл | байт | sha256 |
|---|---|---|
| siglip2_clean_retrain_ep28.pt | 379 735 026 | 82964358a515a40c5ac684265924398cee23b329f4f8fc45b8af1557175225e2 |
| siglip1_clean_ep28.pt | 379 733 498 | 7c201bab224b2e0d348c25218d903132d19555618b8a5802d44f92f4ecdc9e54 |
| siglip1_base_clean_ep28.pt | 379 734 477 | 36deec65c6fb1f3375b097012c198cedda50d7a15aa64d0e7dc3af2707541076 |
| clip_clean_ep28.pt | 351 389 395 | a11ca8cd6eca53616712d014d1574ca48cc7028b6f53b2cc3b830d412f7854f9 |
| ../clean_student/clean_student_a10_ep28.pt | 379 734 230 | 93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce |

## Known deviations

- В коммиченном `run_clean_retrain.sh` только 3 учителя; `siglip2_clean_retrain`
  НЕ прописан, но фактически выполнен тем же `train_teacher()` (`--backbone
  siglip2_base --init-from siglip2_init.pt --seed 42`, tf=squash, geom_aug=True,
  cross_cam=True, лог это подтверждает), вес 0.364. Воспроизводится Шагом 1 —
  см. выше. В 8c1a249 добавлены явные `--tf-mode` (раньше — default); вызов
  `build_clean_teacher.py` не оскриптован (config+результат в
  `bm_improve/clean_models/ensemble/`).

**training code only; a10 weights are in this repo via git LFS (`weights/clean_student_a10_ep28.pt`), учителя — локально/вручную (не в LFS из-за квот).**
