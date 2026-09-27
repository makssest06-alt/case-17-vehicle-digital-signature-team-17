#!/bin/bash
cd /opt/case17/app/research
. .venv_fastreid/bin/activate
python -u evaluate_fastreid.py --config configs/VeRi/sbs_R50-ibn.yml \
  --weights /opt/case17/models/zoo/veri_sbs_R50-ibn.pth --tag fastreid_veri_sbs_r50ibn --batch 8
python -u evaluate_fastreid.py --config configs/VERIWild/bagtricks_R50-ibn.yml \
  --weights /opt/case17/models/zoo/veriwild_bot_R50-ibn.pth --tag fastreid_veriwild_bot_r50ibn --batch 8
echo FR_ALL_DONE
