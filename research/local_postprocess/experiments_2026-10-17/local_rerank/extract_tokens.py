"""Extract A10 patch tokens for holdout + test-gallery frames (production crop: pad 0.10, squash 224).

Saves tokens.npz with:
  ids      (N,)            image ids, val_split order then test_gallery order
  glob     (N, 512)        global embedding exactly as embed_batch (parity check)
  head_tok (N, 196, 512)   each patch token through the ReID head (BNNeck), L2-normalised, fp16
  raw_tok  (N, 196, 768)   backbone patch tokens, L2-normalised, fp16
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).parent
TTA = HERE.parent / "tta"
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
CKPT = r"C:\Users\user\Downloads\clean_student_a10_ep28.pt"
os.environ.update(REID_WEIGHTS=CKPT, REID_TF_MODE="squash", EMBED_TTA_FLIP="0",
                  EMBED_MODEL="reid:vit_base_patch16_siglip_224.webli", OMP_NUM_THREADS="4")
sys.dont_write_bytecode = True
sys.path.insert(0, str(TTA / "api"))
from app.imaging import crop_with_pad  # noqa: E402
from app.models.reid_checkpoint import ReidCheckpointModel  # noqa: E402

REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}
PAD = 0.10
BATCH = 32


def rows(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@torch.inference_mode()
def encode(m: ReidCheckpointModel, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.float16):
        tok = m._model.forward_features(x)                 # (B, 196, 768), SigLIP has no CLS token
        pooled = m._model.forward_head(tok)                # attention-pool, same as m._model(x)
        glob = m._head(pooled)[1].float()
        b, n, c = tok.shape
        head_tok = m._head(tok.reshape(b * n, c))[1].float().reshape(b, n, -1)
    return (F.normalize(glob, dim=1), F.normalize(head_tok, dim=2), F.normalize(tok.float(), dim=2))


def main() -> None:
    train = {r["image_id"]: r for r in rows(D / "train.csv")}
    split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
    items = [(r["image_id"], train[r["image_id"]]) for r in split] + \
            [(r["image_id"], r) for r in rows(D / "test_gallery.csv")]
    m = ReidCheckpointModel("vit_base_patch16_siglip_224.webli", CKPT, 224)
    m.load()
    assert m.tf_mode == "squash" and m.input_size == 224
    glob, head_tok, raw_tok = [], [], []
    t0 = time.time()
    for st in range(0, len(items), BATCH):
        ims = []
        for iid, b in items[st:st + BATCH]:
            im = Image.open(D / "dataset" / "images" / f"{iid}.jpg").convert("RGB")
            ims.append(m._tf(crop_with_pad(im, int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]), PAD)))
        g, h, r = encode(m, torch.stack(ims).to(m.device))
        glob.append(g.cpu().numpy()); head_tok.append(h.half().cpu().numpy()); raw_tok.append(r.half().cpu().numpy())
        if (st // BATCH) % 10 == 0:
            print(st + len(ims), "/", len(items), round(time.time() - t0, 1), "s", flush=True)
    np.savez(HERE / "tokens.npz", ids=np.array([i for i, _ in items]), glob=np.vstack(glob),
             head_tok=np.concatenate(head_tok), raw_tok=np.concatenate(raw_tok))
    ref = np.load(TTA / "tta_emb_pad0.10_flip0.npz")
    ref_by = dict(zip(ref["ids"].astype(str), ref["emb"]))
    g_all = np.vstack(glob)
    cos = np.array([g_all[k] @ ref_by[i] for k, (i, _) in enumerate(items)])
    print(f"parity vs embed_batch cache: min cos {cos.min():.6f}  ({len(items)} frames, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
