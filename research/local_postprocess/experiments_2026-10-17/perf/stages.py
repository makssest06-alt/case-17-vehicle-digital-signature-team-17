"""Per-stage single-thread CPU timings (ms/image, median over 300 frames x 3 reps) + GPU forward times."""
import io, json, random, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from PIL import Image
from bench_q31 import load_frames
import extractors as E
from dlib import choose_f, pad_box
from app.imaging import crop_with_pad, load_image

pc = time.perf_counter
frames = random.Random(0).sample(load_frames(), 300)
m = E.get_model()
tf = m._tf
raws = [(E.read_bytes(p), bb) for p, bb in frames]
SKIP_CPU=True
rows = {k: [] for k in ["file_read", "open_verify", "decode_full_rgb(load_image total)", "decode_full_only",
                        "decode_draft(chosen f)", "crop_service", "resize224_pil", "to_tensor", "normalize",
                        "tf_total_service", "np_asarray_u8", "base_total", "draft_total", "fused_resize_after_draft"]}
fs = []
for rep in range(0 if SKIP_CPU else 3):
    for i, ((raw, (x, y, w, h)), (p, _)) in enumerate(zip(raws, frames)):
        t0 = pc(); E.read_bytes(p); t1 = pc()
        probe = Image.open(io.BytesIO(raw)); probe.verify(); t2 = pc()
        im0 = Image.open(io.BytesIO(raw)); im0.load(); t3 = pc()
        im = load_image(raw, 20 * 2**20); t4 = pc()
        c = crop_with_pad(im, x, y, w, h, E.PAD); t5 = pc()
        r = c.resize((224, 224), Image.BICUBIC); t6 = pc()
        tt = tf.transforms[1](r); t7 = pc()
        tn = tf.transforms[2](tt); t8 = pc()
        a = np.asarray(r); t9 = pc()
        # draft path
        im2 = Image.open(io.BytesIO(raw)); W, H = im2.size
        box = pad_box(W, H, x, y, w, h, E.PAD); f = choose_f(box, 224)
        td0 = pc()
        if f > 1: im2.draft("RGB", (W // f, H // f))
        im2.load(); td1 = pc()
        sx, sy = im2.width / W, im2.height / H
        cb = tuple(int(round(v * s)) for v, s in zip(box, (sx, sy, sx, sy)))
        r2 = im2.crop(cb).resize((224, 224), Image.BICUBIC); td2 = pc()
        fs.append(f)
        vals = dict(zip(rows, [t1 - t0, t2 - t1, t4 - t3, t3 - t2, td1 - td0, t5 - t4, t6 - t5, t7 - t6, t8 - t7,
                                t8 - t5 + 0 * 1, t9 - t8, 0, 0, 0]))
        vals["tf_total_service"] = (t6 - t5) + (t7 - t6) + (t8 - t7)
        vals["base_total"] = (t1 - t0) + (t4 - t3) + (t2 - t1) + (t5 - t4) + vals["tf_total_service"]
        vals["draft_total"] = (t1 - t0) + (t2 - t1) + (td2 - td0) + (t7 - t6) + (t8 - t7)
        vals["fused_resize_after_draft"] = td2 - td1
        for k, v in vals.items(): rows[k].append(v * 1e3)
out = {k: dict(p50=float(np.median(v)), mean=float(np.mean(v)), p95=float(np.percentile(v, 95))) for k, v in rows.items() if v}
out["f_dist"] = {int(k): int(fs.count(k)) for k in (1, 2, 4, 8)}
print(json.dumps(out, indent=1), flush=True)

# GPU forward-only timings (ms per batch), fp16 autocast eager vs graph, input already on GPU
res = {}
for name in ("gpu16_draft", "gpu16_draft_pre", "gpu16_draft_pre_graph"):
    ex = E.make(name)
    for B in (1, 8, 16, 32):
        u8 = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8, device="cuda")
        buf = torch.randint(0, 255, (32, 224, 224, 3), dtype=torch.uint8).pin_memory()
        ts = []
        for it in range(70):
            torch.cuda.synchronize(); t0 = pc()
            if ex.graph:
                bk = next(b for b in E.BUCKETS if b >= B)
                g, sin, sout = ex.graphs[bk]
                with torch.inference_mode():
                    sin[:B].copy_(buf[:B], non_blocking=True); g.replay(); sout[:B].cpu()
            else:
                with torch.inference_mode():
                    ex._fwd(buf[:B].to("cuda", non_blocking=True)).cpu()
            torch.cuda.synchronize(); ts.append((pc() - t0) * 1e3)
        res[f"{name}_B{B}"] = float(np.median(ts[20:]))
    del ex
print(json.dumps(res, indent=1))
(HERE / "stages.json").write_text(json.dumps(dict(cpu=out, gpu=res), indent=1))
