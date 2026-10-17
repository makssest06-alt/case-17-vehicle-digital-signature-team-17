#!/usr/bin/env python3
"""Сколько CPU добавляет штраф за ту же камеру к одному запросу — код сервиса, без модели и базы.

Меряется ровно то, что Postprocess.search делает сверх обычного поиска:
признак фона запроса из уже декодированного кадра (background.service_descriptor),
флаги «та же камера» по --topk кандидатам (их признаки сервис держит в памяти в
float16), пересортировка и оценка отказа. Декод кадра (imaging.load_image) — для
сравнения: его сервис делает и сейчас.

    python scripts/bench_bg_cpu.py --images-dir <кадры> [--topk 100] [--n 300]
    # ограничение CPU как на стенде:
    docker run --cpus=2 ... python scripts/bench_bg_cpu.py ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))


def timed(fn, n: int, warm: int = 10) -> dict:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "mean_ms": float(np.mean(ts))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    from app.imaging import load_image
    from app.pp import background as bgm
    from app.pp import rerank as rr

    raws = [p.read_bytes() for p in sorted(a.images_dir.glob("*.jpg"))[:40]]
    frames = [load_image(r, 50 * 2 ** 20) for r in raws]
    box = lambda im: (im.width // 3, im.height // 3, im.width // 4, im.height // 4)
    rng = np.random.default_rng(0)
    gal = [bgm.l2(rng.standard_normal(bgm.BG_DIM).astype("float32")).astype("float16") for _ in range(a.topk)]
    cos = np.sort(rng.random(a.topk))[::-1]
    i = [0]

    def nxt():
        i[0] += 1
        return frames[i[0] % len(frames)], raws[i[0] % len(raws)]

    def penalty():
        im, _ = nxt()
        flags = rr.camera_flags(bgm.service_descriptor(im, box(im)), gal, 0.68127)
        order = rr.rank(cos, flags)
        return rr.refusal_score(cos[order], flags[order], "raw")

    q0 = bgm.service_descriptor(frames[0], box(frames[0]))
    res = {"cpu_count": os.cpu_count(), "omp_threads": os.getenv("OMP_NUM_THREADS"),
           "cfs_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists()
           else None, "frame": list(frames[0].size), "topk": a.topk,
           "decode_full_frame": timed(lambda: load_image(nxt()[1], 50 * 2 ** 20), a.n // 3),
           "descriptor": timed(lambda: bgm.service_descriptor(*(lambda im: (im, box(im)))(nxt()[0])), a.n),
           "flags_topk": timed(lambda: rr.camera_flags(q0, gal, 0.68127), a.n),
           "penalty_total": timed(penalty, a.n)}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
