"""Organizer protocol (FAQ) benchmark for extract(batch) callables.

latency_b1 : batch=1, 50 warm-up + 300 timed runs, cuda.synchronize before/after each, median + p95.
throughput : batches 1/8/16/32, each run >= 10 s sustained (after a short warm-up), FPS each, best scored.
Cycle: the 1860 test frames (test_query + test_gallery), real JPEGs read from disk each call
(OS file cache is warm after the first pass - noted in the report).

usage: python bench_q31.py <variant> [<variant> ...]   (names understood by extractors.make)
       env BENCH_SECONDS (default 10), BENCH_OUT (json path, default perf/res_<variant>.json)
"""
import csv, json, os, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch

D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
IMG = D / "dataset" / "images"
SECONDS = float(os.environ.get("BENCH_SECONDS", "10"))


def load_frames():
    out = []
    for name in ("test_query.csv", "test_gallery.csv"):
        with (D / "dataset" / name).open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                out.append((str(IMG / f"{r['image_id']}.jpg"), (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))))
    return out


class Cycler:
    def __init__(self, frames, start=0):
        self.frames, self.i = frames, start

    def take(self, n):
        out = [self.frames[(self.i + k) % len(self.frames)] for k in range(n)]
        self.i = (self.i + n) % len(self.frames)
        return out


def sync():
    torch.cuda.synchronize()


def latency_b1(ex, frames, warm=50, runs=300):
    cy = Cycler(frames, 7)
    for _ in range(warm):
        ex.extract(cy.take(1));
    sync()
    ts = []
    for _ in range(runs):
        b = cy.take(1)
        sync(); t0 = time.perf_counter()
        out = ex.extract(b)
        sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return dict(median_ms=statistics.median(ts), p95_ms=float(np.percentile(ts, 95)),
                mean_ms=float(np.mean(ts)), min_ms=min(ts), max_ms=max(ts))


def throughput(ex, frames, B, seconds=SECONDS, warm_s=1.5):
    cy = Cycler(frames, 100 * B)
    t_end = time.perf_counter() + warm_s
    while time.perf_counter() < t_end:
        ex.extract(cy.take(B))
    sync()
    n, t0 = 0, time.perf_counter()
    while True:
        b = cy.take(B)
        ex.extract(b); n += B
        now = time.perf_counter()
        if now - t0 >= seconds:
            break
    sync(); dt = time.perf_counter() - t0
    return n / dt


def run(name, frames):
    import extractors
    ex = extractors.make(name)
    r = dict(variant=name)
    r["latency_b1"] = latency_b1(ex, frames)
    print(name, "latency", json.dumps(r["latency_b1"]), flush=True)
    r["fps"] = {}
    for B in (1, 8, 16, 32):
        r["fps"][B] = throughput(ex, frames, B)
        print(name, "B", B, "fps %.1f" % r["fps"][B], flush=True)
    r["best_fps"] = max(r["fps"].values())
    r["best_B"] = max(r["fps"], key=r["fps"].get)
    return r


if __name__ == "__main__":
    frames = load_frames()
    print("frames", len(frames), "seconds/run", SECONDS, flush=True)
    for name in sys.argv[1:]:
        res = run(name, frames)
        (HERE / f"res_{name}.json").write_text(json.dumps(res, indent=1))
        print("RESULT", json.dumps(res), flush=True)
