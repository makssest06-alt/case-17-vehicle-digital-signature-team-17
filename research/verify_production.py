#!/usr/bin/env python3
"""Проверка, что production-обёртка воспроизводит метрики чекпоинта.

Грузит веса ровно тем кодом, что работает в сервисе
(`api/app/models/reid_checkpoint.py`), считает эмбеддинги его же
`embed_batch` (внутри — усреднение с горизонтальным отражением) и
сверяет mAP / Rank-1 / Rank-5 с результатом `eval_checkpoint.py --flip`.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "api"))

from eval_checkpoint import crop, evaluate, load_split  # noqa: E402
from app.models.reid_checkpoint import ReidCheckpointModel  # noqa: E402

BACKBONE = "vit_base_patch16_siglip_224.webli"
WEIGHTS = str(HERE.parent / "models" / "reid_current.pt")


def main() -> int:
    IMAGES = Path("D:/bm/dataset_extracted/images")
    batch, context = 16, 0.10

    model = ReidCheckpointModel(BACKBONE, WEIGHTS)
    model.load()
    print(f"loaded: name={model.name} dim={model.dim}", flush=True)

    qry, gal = load_split()

    def embed(rows):
        out = []
        for i in range(0, len(rows), batch):
            imgs = [
                crop(Image.open(IMAGES / f"{r['image_id']}.jpg").convert("RGB"), r, context)
                for r in rows[i : i + batch]
            ]
            out.append(model.embed_batch(imgs))
        return np.vstack(out).astype("float32")

    g_emb = embed(gal)
    q_emb = embed(qry)
    res = evaluate(q_emb, g_emb, qry, gal)
    print(f"  mAP {res['mAP']}  Rank-1 {res['rank1']}  Rank-5 {res['rank5']}  "
          f"(queries {res['queries_used']}, dim {q_emb.shape[1]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
