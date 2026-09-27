#!/bin/bash
cd /opt/case17/app/research
. .venv_timm/bin/activate
python -u evaluate.py --model vit_base_patch16_siglip_224.v2_webli --tag siglip2_base --input-size 224
python -u evaluate.py --model vit_so400m_patch14_siglip_224.webli --tag siglip_so400m --input-size 224
python -u evaluate.py --model vit_so400m_patch14_siglip_224.v2_webli --tag siglip2_so400m --input-size 224
echo ZS_DONE
