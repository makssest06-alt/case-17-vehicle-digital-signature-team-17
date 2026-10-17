import os, sys, time, csv
from pathlib import Path
import numpy as np, torch
from PIL import Image
TTA = Path(__file__).parent.parent / "tta"
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
CK = {"A10": r"C:\Users\user\Downloads\clean_student_a10_ep28.pt",
      "old": r"C:\Users\user\PycharmProjects\AIOS\BM\test-clone\repo\weights\siglip2_distill_ensemble_ep28.pt"}
os.environ.update(REID_TF_MODE="squash", EMBED_TTA_FLIP="0", OMP_NUM_THREADS="4")
sys.dont_write_bytecode = True
sys.path.insert(0, str(TTA / "api"))
from app.models.reid_checkpoint import ReidCheckpointModel
from app.imaging import crop_with_pad
with (D / "train.csv").open(newline="", encoding="utf-8-sig") as f: b = list(csv.DictReader(f))[0]
im = Image.open(D / "dataset" / "images" / f"{b['image_id']}.jpg").convert("RGB")
crop = crop_with_pad(im, int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]), 0.10)
def lat(fn, warm=20, n=100):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return np.percentile(ts, 50), np.percentile(ts, 95)
print("GPU", torch.cuda.get_device_name(0))
for name, ck in CK.items():
    os.environ["REID_WEIGHTS"] = ck
    m = ReidCheckpointModel("vit_base_patch16_siglip_224.webli", ck, 224); m.load()
    x = torch.stack([m._tf(crop)]).to(m.device)
    with torch.inference_mode():
        g = lat(lambda: m._encode(x))
    e = lat(lambda: m.embed_batch([crop]))
    print(f"{name}: GPU-only fwd p50 {g[0]:.2f} ms (p95 {g[1]:.2f}) | end-to-end embed_batch([1]) p50 {e[0]:.2f} ms (p95 {e[1]:.2f}) fp16={m._fp16}")
    del m; torch.cuda.empty_cache()
