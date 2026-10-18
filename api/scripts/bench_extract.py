#!/usr/bin/env python3
"""Замер extract() по протоколу организаторов (ответы Q31, Q32, Q34) — одной командой в контейнере:

    docker compose exec api python scripts/bench_extract.py                 # DATA_DIR: test_query.csv + images/
    docker compose exec api python scripts/bench_extract.py --data /data --out /artifacts/bench_extract.json

  latency_b1  медиана полного цикла extract() при batch=1: 300 прогонов после 50 прогревочных,
              torch.cuda.synchronize до и после каждого;
  throughput  устойчивый FPS при батчах 1 / 8 / 16 / 32, не меньше 10 с на каждом размере; в баллы — лучший;
  ещё         пиковый VRAM, время загрузки весов, суммарный размер весов (расширения из Q37), детерминизм
              (два прогона на одних данных), устройство, sha256 весов, баллы по формуле Q34.
Полный цикл — app/extractor.py: чтение файла с диска, декодирование, кроп по bbox, препроцессинг, forward, L2.
Флаги ускорения — переменные окружения (REID_FP16, EXTRACT_JPEG_DRAFT, EXTRACT_THREADS), см. app/extractor.py.
--smoke — дымовой прогон на 2–3 кадрах (не замер).
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # api/ — пакет app (в контейнере /srv)

from app.config import settings            # noqa: E402
from app.extractor import Extractor        # noqa: E402

WEIGHT_EXT = (".pt", ".pth", ".bin", ".onnx", ".engine", ".plan", ".safetensors", ".ckpt", ".trt", ".pb",
              ".tflite", ".npz")                                   # Q37: чем меряют суммарный размер весов


def sync(model):
    import torch
    dev = getattr(model, "device", None)
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev is not None and dev.type == "mps":
        torch.mps.synchronize()


def weights_size(roots) -> dict:
    files = []
    for root in roots:
        p = Path(root)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in WEIGHT_EXT]
    seen, total = set(), 0
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            total += r.stat().st_size
    return {"roots": [str(r) for r in roots], "files": len(seen), "bytes": total, "MB": round(total / 2 ** 20, 1)}


def score(latency_ms: float, fps: float) -> dict:
    """Q34: latency ≤ 40 мс — полный балл, 40–80 — линейно, > 80 — 0; FPS ≥ 100 — полный, 50–100 — линейно."""
    lat = 1.0 if latency_ms <= 40 else max(0.0, (80 - latency_ms) / 40)
    thr = 1.0 if fps >= 100 else max(0.0, (fps - 50) / 50)
    return {"latency_score": round(lat, 3), "throughput_score": round(thr, 3),
            "performance_pct_of_20": round(10 * lat + 10 * thr, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=settings.data_dir, help="каталог: <csv> и images/ (по умолчанию DATA_DIR)")
    ap.add_argument("--csv", default="test_query.csv", help="image_id,x,y,w,h — какие кадры гонять")
    ap.add_argument("--images", default=None, help="каталог кадров, если не <data>/images")
    ap.add_argument("--frames", type=int, default=256, help="сколько разных кадров циклически перебирать")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--scan", nargs="*", default=None,
                    help="где считать размер весов (Q37); по умолчанию REID_WEIGHTS, HF_HOME, /srv")
    ap.add_argument("--smoke", action="store_true", help="дымовой прогон: 2 прогрева, 3 замера, 1 с на батч")
    ap.add_argument("--out", default=None, help="куда записать JSON")
    a = ap.parse_args()
    if a.smoke:
        a.warmup, a.runs, a.seconds, a.frames = 2, 3, 1.0, min(a.frames, 3)

    data = Path(a.data)
    images = Path(a.images) if a.images else data / "images"
    with (data / a.csv).open(encoding="utf-8", newline="") as f:
        rows = [r for _, r in zip(range(a.frames), csv.DictReader(f))]
    items = [(images / f"{r['image_id']}.jpg", tuple(int(float(r[c])) for c in "xywh")) for r in rows]
    if not items:
        print(f"нет кадров в {data / a.csv}", file=sys.stderr)
        return 1

    import torch
    ex = Extractor()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ex.load()
    m = ex.model
    weights = os.getenv("REID_WEIGHTS", "")
    w_sha = hashlib.sha256(Path(weights).read_bytes()).hexdigest() if weights and Path(weights).is_file() else None

    # latency_b1: полный цикл на одно ТС, синхронизация до и после каждого замера
    for i in range(a.warmup):
        ex.extract(*items[i % len(items)])
    sync(m)
    lat = []
    for i in range(a.runs):
        sync(m)
        t0 = time.perf_counter()
        ex.extract(*items[i % len(items)])
        sync(m)
        lat.append((time.perf_counter() - t0) * 1000)
    lat_ms = statistics.median(lat)

    # throughput: устойчивый FPS при каждом размере батча, не меньше --seconds
    fps = {}
    for B in (int(x) for x in a.batches.split(",")):
        batch = [items[j % len(items)] for j in range(B)]
        ex.extract_batch(batch)                                         # прогрев размера
        sync(m)
        n, t0 = 0, time.perf_counter()
        while True:
            ex.extract_batch(batch)
            sync(m)
            n += B
            if time.perf_counter() - t0 >= a.seconds:
                break
        fps[B] = n / (time.perf_counter() - t0)

    # детерминизм: два прогона на одних данных
    probe = items[: min(len(items), 32)]
    v1, v2 = ex.extract_batch(probe), ex.extract_batch(probe)
    best_b = max(fps, key=fps.get)
    roots = a.scan if a.scan is not None else [p for p in (weights, os.getenv("HF_HOME", ""), "/srv") if p]
    res = {
        **ex.describe(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "weights": Path(weights).name if weights else None, "weights_sha256": w_sha,
        "frames": len(items), "smoke": a.smoke,
        "latency_b1_ms": {"median": round(lat_ms, 2), "p95": round(float(np.percentile(lat, 95)), 2),
                          "runs": a.runs, "warmup": a.warmup},
        "throughput_fps": {str(k): round(v, 1) for k, v in fps.items()}, "best_fps": round(fps[best_b], 1),
        "best_batch": best_b,
        "peak_vram_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else None,
        "determinism_max_abs_diff": float(np.abs(v1 - v2).max()),
        "weights_size_q37": weights_size(roots),
        **score(lat_ms, fps[best_b]),
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
