import csv, io, os, sys, time, json, random
from pathlib import Path
import numpy as np
SCR = Path(__file__).resolve().parent.parent
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
CKPT = r"C:\Users\user\Downloads\clean_student_a10_ep28.pt"
os.environ.update(REID_WEIGHTS=CKPT, REID_TF_MODE="squash", EMBED_TTA_FLIP="0",
                  EMBED_MODEL="reid:vit_base_patch16_siglip_224.webli", OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCR / "tta" / "api")); sys.path.insert(0, str(Path(__file__).parent))
import torch
from app.config import settings
from app.models.reid_checkpoint import ReidCheckpointModel
from dlib import load_crop
object.__setattr__(settings, "tta_flip", False)
m = ReidCheckpointModel("vit_base_patch16_siglip_224.webli", CKPT, 224); m.load()
tf = m._tf
def rows(p):
    with p.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))
allr = rows(D/"test_query.csv") + rows(D/"test_gallery.csv")
random.Random(0).shuffle(allr); sel = allr[:300]
raws = [(open(D/"dataset"/"images"/f"{r['image_id']}.jpg","rb").read(), (int(r['x']),int(r['y']),int(r['w']),int(r['h']))) for r in sel]
VAR = {"full": None, "d224": 224, "d160": 160}
REPS = 3
pc = time.perf_counter
def one(raw, bb, ms):
    t0 = pc(); im, f = load_crop(raw, bb, 0.10, ms); t1 = pc()
    x = tf(im); t2 = pc()
    return t1 - t0, t2 - t1, f
for _ in range(20): one(*raws[0], None)  # warm
res = {}
for name, ms in VAR.items():
    dec = np.full((REPS, len(raws)), np.nan); trf = dec.copy(); fs = [0]*len(raws)
    for rep in range(REPS):
        for i, (raw, bb) in enumerate(raws):
            d, t, f = one(raw, bb, ms); dec[rep, i] = d; trf[rep, i] = t; fs[i] = f
    d = np.median(dec, 0) * 1e3; t = np.median(trf, 0) * 1e3; tot = d + t
    res[name] = dict(decode_crop_p50=np.percentile(d,50), decode_crop_p95=np.percentile(d,95),
        tf_p50=np.percentile(t,50), tf_p95=np.percentile(t,95),
        tot_p50=np.percentile(tot,50), tot_p95=np.percentile(tot,95), tot_mean=tot.mean(),
        dec_frac=d.sum()/tot.sum(), tf_frac=t.sum()/tot.sum(),
        fdist={int(k): fs.count(k) for k in (1,2,4,8)})
    print(name, json.dumps(res[name], default=float), flush=True)
# batch path of 32: load+crop (serial), tf (stack), GPU encode
dev = m.device
def batch(ms, idx):
    t0 = pc(); crops = [load_crop(*raws[i], 0.10, ms)[0] for i in idx]; t1 = pc()
    x = torch.stack([tf(c) for c in crops]); t2 = pc()
    x = x.to(dev, non_blocking=True)
    z = m._encode(x); 
    if dev.type == "cuda": torch.cuda.synchronize()
    z.cpu(); t3 = pc()
    return t1-t0, t2-t1, t3-t2
b = {}
for name, ms in VAR.items():
    runs = []
    for rep in range(5):
        for s in range(0, 288, 32):  # 9 batches of 32
            runs.append(batch(ms, range(s, s+32)))
    if name == "full": [batch(None, range(32)) for _ in range(3)]
    a = np.array(runs[9:]) * 1e3  # drop first rep warm
    b[name] = dict(dec_crop_ms=a[:,0].mean(), tf_ms=a[:,1].mean(), gpu_ms=a[:,2].mean(), total_ms=a.sum(1).mean(), total_p95=np.percentile(a.sum(1),95))
    print("batch32", name, json.dumps(b[name], default=float), flush=True)
json.dump(dict(single=res, batch32=b, device=str(dev)), open(Path(__file__).parent/"timing.json","w"), indent=1, default=float)
