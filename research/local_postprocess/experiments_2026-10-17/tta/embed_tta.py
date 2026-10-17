"""Embed A10 holdout + test-gallery frames through the repo ReID loader (copied to ./api),
for tta_flip in {0,1} and pad in {0.05,0.10,0.15}. Saves tta_emb_pad{pad}_flip{f}.npz."""
import csv, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
CKPT = r"C:\Users\user\Downloads\clean_student_a10_ep28.pt"
os.environ.update(REID_WEIGHTS=CKPT, REID_TF_MODE="squash", EMBED_TTA_FLIP="0",
                  EMBED_MODEL="reid:vit_base_patch16_siglip_224.webli", OMP_NUM_THREADS="4")
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "api"))
from app.config import settings
from app.models.reid_checkpoint import ReidCheckpointModel
from app.imaging import crop_with_pad

REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}
PADS = [0.10, 0.05, 0.15]

def rows(p):
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def set_flip(v: bool):
    object.__setattr__(settings, "tta_flip", v)

def main():
    train = {r["image_id"]: r for r in rows(D / "train.csv")}
    split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
    tg = rows(D / "test_gallery.csv")
    items = [(r["image_id"], train[r["image_id"]]) for r in split] + [(r["image_id"], r) for r in tg]
    ids = [i for i, _ in items]
    print("frames", len(items), flush=True)
    m = ReidCheckpointModel("vit_base_patch16_siglip_224.webli", CKPT, 224)
    m.load()
    print("device", m.device, "fp16", m._fp16, "tf", m.tf_mode, "size", m.input_size, flush=True)
    bs = 32
    out = {(p, f): [] for p in PADS for f in (0, 1)}
    t0 = time.time()
    for st in range(0, len(items), bs):
        chunk = items[st:st + bs]
        frames = []
        for iid, b in chunk:
            im = Image.open(D / "dataset" / "images" / f"{iid}.jpg").convert("RGB")
            frames.append((im, b))
        for p in PADS:
            ims = [crop_with_pad(im, int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]), p) for im, b in frames]
            for f in (0, 1):
                set_flip(bool(f))
                out[(p, f)].append(m.embed_batch(ims))
        if (st // bs) % 10 == 0:
            print(st + len(chunk), "/", len(items), round(time.time() - t0, 1), "s", flush=True)
    for (p, f), v in out.items():
        np.savez(HERE / f"tta_emb_pad{p:.2f}_flip{f}.npz", ids=np.array(ids), emb=np.vstack(v).astype("float32"))
    print("done", round(time.time() - t0, 1), flush=True)

if __name__ == "__main__":
    main()
