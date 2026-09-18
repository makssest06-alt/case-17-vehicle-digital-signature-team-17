#!/bin/bash
# Сравнение пресетов на замороженных признаках SigLIP.
# Три механизма из четырёх: кросс-камерные позитивы, трудные батчи, центроидный
# лосс. Геометрических аугментаций тут нет — признаки посчитаны заранее.
cd /opt/case17/app/research
. .venv/bin/activate
set -x
python -u train_head.py --backbone siglip_base --preset baseline  --tag p_baseline  --out-dim 512 --loss arcface --epochs 40
python -u train_head.py --backbone siglip_base --preset viewpoint --tag p_viewpoint --out-dim 512 --loss arcface --epochs 40
python -u train_head.py --backbone siglip_base --preset centroid  --tag p_centroid  --out-dim 512 --loss arcface --epochs 40
echo PRESETS_DONE
