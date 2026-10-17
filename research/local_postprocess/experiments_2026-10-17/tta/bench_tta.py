import os, sys, time, csv
from pathlib import Path
import numpy as np, torch
from PIL import Image
HERE = Path(__file__).parent
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
CKPT = r"C:\Users\user\Downloads\clean_student_a10_ep28.pt"
os.environ.update(REID_WEIGHTS=CKPT, REID_TF_MODE="squash", EMBED_TTA_FLIP="0", OMP_NUM_THREADS="4")
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "api"))
from app.config import settings
from app.models.reid_checkpoint import ReidCheckpointModel
from app.imaging import crop_with_pad
m = ReidCheckpointModel("vit_base_patch16_siglip_224.webli", CKPT, 224); m.load()
with (D / "train.csv").open(newline="", encoding="utf-8-sig") as f: tr = list(csv.DictReader(f))[:32]
ims = []
for b in tr:
    im = Image.open(D / "dataset" / "images" / f"{b['image_id']}.jpg").convert("RGB")
    ims.append(crop_with_pad(im, int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]), 0.10))
def setf(v): object.__setattr__(settings, "tta_flip", v)
def sync(): torch.cuda.synchronize()
def lat(fn, warm=20, n=200):
    for _ in range(warm): fn()
    sync(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))
print("GPU", torch.cuda.get_device_name(0), "fp16", m._fp16)
x1 = torch.stack([m._tf(ims[0])]).to(m.device)
for flip in (False, True):
    setf(flip)
    p50, p95 = lat(lambda: m.embed_batch([ims[0]]))
    print(f"batch1 end-to-end (PIL->tf->GPU->npy) flip={flip}: p50 {p50:.2f} ms p95 {p95:.2f} ms")
for flip in (False, True):
    def f():
        a = m._encode(x1)
        if flip: a = a + m._encode(torch.flip(x1, dims=[3]))
        return torch.nn.functional.normalize(a, dim=1)
    with torch.inference_mode():
        p50, p95 = lat(f)
    print(f"batch1 GPU-only flip={flip}: p50 {p50:.2f} ms p95 {p95:.2f} ms")
for flip in (False, True):
    setf(flip)
    p50, _ = lat(lambda: m.embed_batch(ims), warm=5, n=30)
    print(f"batch32 end-to-end flip={flip}: p50 {p50:.1f} ms/batch -> {32000/p50:.1f} img/s")
x32 = torch.stack([m._tf(i) for i in ims]).to(m.device)
for flip in (False, True):
    def f():
        a = m._encode(x32)
        if flip: a = a + m._encode(torch.flip(x32, dims=[3]))
        return torch.nn.functional.normalize(a, dim=1)
    with torch.inference_mode():
        p50, _ = lat(f, warm=5, n=30)
    print(f"batch32 GPU-only flip={flip}: p50 {p50:.1f} ms/batch -> {32000/p50:.1f} img/s")
