"""embed_ft.py template; only image loading/cropping swapped. usage: embed_dec.py <ckpt> <out.npz> <full|d224|d160>"""
import csv, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
CKPT, OUT, VAR = sys.argv[1], sys.argv[2], sys.argv[3]
MS = {"full": None, "d224": 224, "d160": 160}[VAR]
TTA = Path(__file__).resolve().parent.parent / "tta"
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
os.environ.update(REID_WEIGHTS=CKPT, REID_TF_MODE="squash", EMBED_TTA_FLIP="0",
                  EMBED_MODEL="reid:vit_base_patch16_siglip_224.webli", OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(TTA / "api")); sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings
from app.models.reid_checkpoint import ReidCheckpointModel
from dlib import load_crop
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
print("device", m.device, "tf", m.tf_mode, "variant", VAR, flush=True)
out, t0, fc = [], time.time(), Counter()
for st in range(0, len(items), 32):
    ims = []
    for iid, b in items[st:st + 32]:
        raw = (D / "dataset" / "images" / f"{iid}.jpg").read_bytes()
        im, f = load_crop(raw, (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])), 0.10, MS)
        fc[f] += 1; ims.append(im)
    out.append(m.embed_batch(ims))
np.savez(OUT, ids=np.array([i for i, _ in items]), emb=np.vstack(out).astype("float32"))
print("done", len(items), round(time.time() - t0, 1), "s", "f dist", dict(sorted(fc.items())), flush=True)
