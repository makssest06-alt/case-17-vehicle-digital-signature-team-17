"""Embedding fidelity vs baseline on 200 frames + determinism (two runs). usage: cmp_emb.py v1 v2 ..."""
import json, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from bench_q31 import load_frames
import extractors

frames = load_frames()
sel = random.Random(1).sample(frames, 200)


def embed(ex, bs):
    return np.concatenate([ex.extract(sel[i:i + bs]) for i in range(0, len(sel), bs)])


def stats(a, b):
    cos = (a * b).sum(1)
    return dict(max_abs_cos_m1=float(np.abs(cos - 1).max()), mean_cos=float(cos.mean()), min_cos=float(cos.min()),
                max_abs_diff=float(np.abs(a - b).max()))


base_ex = extractors.make("baseline")
ref32 = embed(base_ex, 32)
ref1 = embed(base_ex, 1)
res = {"baseline": dict(det32=stats(ref32, embed(base_ex, 32)), det1=stats(ref1, embed(base_ex, 1)),
                         b1_vs_b32=stats(ref1, ref32))}
print("baseline", json.dumps(res["baseline"]), flush=True)
for name in sys.argv[1:]:
    ex = extractors.make(name)
    e32a, e32b = embed(ex, 32), embed(ex, 32)
    e1a, e1b = embed(ex, 1), embed(ex, 1)
    e8 = embed(ex, 8)
    r = dict(vs_base_b32=stats(ref32, e32a), vs_base_b1=stats(ref1, e1a), vs_base_b8=stats(ref32, e8),
             det32=stats(e32a, e32b), det1=stats(e1a, e1b))
    res[name] = r
    print(name, json.dumps(r), flush=True)
    del ex
    torch.cuda.empty_cache()
(HERE / ("cmp_" + "_".join(sys.argv[1:])[:60] + ".json")).write_text(json.dumps(res, indent=1))
