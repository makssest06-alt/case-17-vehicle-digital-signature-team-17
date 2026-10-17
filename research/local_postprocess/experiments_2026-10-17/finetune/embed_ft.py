"""Embed holdout (546q+361g) + 750 test-gallery frames through the service loader for one checkpoint.
usage: embed_ft.py <ckpt> <out.npz>   (pad 0.10, flip 0, squash; same as tta/embed_tta.py)"""
import csv, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
CKPT, OUT = sys.argv[1], sys.argv[2]
TTA = Path(__file__).resolve().parent.parent / "tta"
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
os.environ.update(REID_WEIGHTS=CKPT, REID_TF_MODE="squash", EMBED_TTA_FLIP="0",
                  EMBED_MODEL="reid:vit_base_patch16_siglip_224.webli", OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(TTA / "api"))
from app.config import settings
from app.models.reid_checkpoint import ReidCheckpointModel
from app.imaging import crop_with_pad
REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}
def rows(p):
    with p.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))
train = {r["image_id"]: r for r in rows(D / "train.csv")}
split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
seen, hg, hq = set(), [], []
for r in split:
    k = (r["vehicle_id"], r["camera_id"]); (hq if k in seen else hg).append(r); seen.add(k)
tg = rows(D / "test_gallery.csv")
print("holdout q/g", len(hq), len(hg), "test gallery", len(tg), flush=True)
items = [(r["image_id"], train[r["image_id"]]) for r in hq + hg] + [(r["image_id"], r) for r in tg]
object.__setattr__(settings, "tta_flip", False)
m = ReidCheckpointModel("vit_base_patch16_siglip_224.webli", CKPT, 224); m.load()
print("device", m.device, "tf", m.tf_mode, flush=True)
out, t0 = [], time.time()
for st in range(0, len(items), 32):
    ims = []
    for iid, b in items[st:st + 32]:
        im = Image.open(D / "dataset" / "images" / f"{iid}.jpg").convert("RGB")
        ims.append(crop_with_pad(im, int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]), 0.10))
    out.append(m.embed_batch(ims))
np.savez(OUT, ids=np.array([i for i, _ in items]), emb=np.vstack(out).astype("float32"))
print("done", len(items), round(time.time() - t0, 1), "s", flush=True)
