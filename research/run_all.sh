#!/bin/bash
cd /opt/case17/app/research
. .venv/bin/activate
set -x
python evaluate.py --model resnet50.a1_in1k                    --tag resnet50 --input-size 224
python evaluate.py --model vit_small_patch14_dinov2.lvd142m    --tag dinov2_vits14 --input-size 224
python evaluate.py --model vit_base_patch16_clip_224.openai    --tag clip_vitb16 --input-size 224
set +x
echo ALL_DONE
