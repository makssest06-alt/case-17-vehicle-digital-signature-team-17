#!/usr/bin/env python3
"""Замер ровно тех метрик, которые будет мерить жюри.

    docker compose exec api python scripts/benchmark.py
    docker compose exec api python scripts/benchmark.py --quick

Печатает таблицу и пишет benchmark.json. Метрики:
  * latency на одно ТС при батче 1 — p50 / p95 / p99;
  * пропускная способность (FPS) при батчах 1 / 8 / 16 / 32 / 64;
  * пиковая RAM и VRAM;
  * суммарный размер весов;
  * стабильность памяти на длительном прогоне — не течёт ли.
"""
import argparse
import csv
import gc
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/srv")

from app.config import settings                # noqa: E402
from app.models.registry import build_model    # noqa: E402

CROPS = Path(settings.data_dir) / "crops"


def rss_mb() -> float:
    """Текущий RSS процесса в МБ, без внешних зависимостей."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2
    except Exception:
        return float("nan")


def vram_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:
        pass
    return None


def weights_mb(model) -> float:
    # у ONNX-модели нет torch-модулей: вес это сам файл графа
    path = getattr(model, "path", None)
    if getattr(model, "_sess", None) is not None and path and os.path.exists(path):
        return os.path.getsize(path) / 1024 ** 2
    total = 0.0
    for attr in ("_model", "_head"):
        m = getattr(model, attr, None)
        if m is not None:
            total += sum(p.numel() * p.element_size() for p in m.parameters())
    return total / 1024 ** 2


def load_images(n: int):
    ids = [r["image_id"] for r in csv.DictReader((CROPS / "crops_query.csv").open())][:n]
    return [Image.open(CROPS / "query" / f"{i}.jpg").convert("RGB") for i in ids]


def pct(values, q):
    values = sorted(values)
    k = min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))
    return values[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="короткий прогон, без теста стабильности")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n-latency", type=int, default=50)
    ap.add_argument("--batches", default="1,8,16,32,64")
    ap.add_argument("--soak-seconds", type=int, default=300)
    ap.add_argument("--out", default="/artifacts/benchmark.json")
    a = ap.parse_args()

    device = "cpu"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    print(f"устройство: {device}")
    t0 = time.perf_counter()
    model = build_model(settings.embed_model, settings.input_size)
    model.load()
    load_s = time.perf_counter() - t0
    print(f"модель {model.name}: загружена за {load_s:.1f} с, {model.dim}D, "
          f"веса {weights_mb(model):.0f} МБ")

    batches = [int(b) for b in a.batches.split(",")]
    pool = load_images(max(batches + [a.n_latency]))

    print(f"\nпрогрев ({a.warmup} итераций)...")
    for _ in range(a.warmup):
        model.embed_batch(pool[:1])

    # --- latency при батче 1, то что меряет жюри ---
    lat = []
    for i in range(a.n_latency):
        img = pool[i % len(pool)]
        t = time.perf_counter()
        model.embed_batch([img])
        lat.append((time.perf_counter() - t) * 1000)

    res = {
        "model": model.name,
        "device": device,
        "dim": model.dim,
        "weights_mb": round(weights_mb(model), 1),
        "load_s": round(load_s, 1),
        "latency_batch1_ms": {
            "p50": round(pct(lat, 50), 2),
            "p95": round(pct(lat, 95), 2),
            "p99": round(pct(lat, 99), 2),
            "mean": round(st.mean(lat), 2),
            "min": round(min(lat), 2),
            "n": len(lat),
        },
        "throughput_fps": {},
    }

    # --- FPS по батчам ---
    for bs in batches:
        imgs = [pool[i % len(pool)] for i in range(bs)]
        for _ in range(2):
            model.embed_batch(imgs)
        reps = max(3, min(20, 200 // bs))
        t = time.perf_counter()
        for _ in range(reps):
            model.embed_batch(imgs)
        dt = time.perf_counter() - t
        res["throughput_fps"][str(bs)] = round(reps * bs / dt, 2)

    res["peak_rss_mb"] = round(rss_mb(), 1)
    v = vram_mb()
    if v is not None:
        res["peak_vram_mb"] = round(v, 1)

    # --- стабильность памяти ---
    if not a.quick:
        print(f"\nтест стабильности памяти, {a.soak_seconds} с...")
        gc.collect()
        start_rss = rss_mb()
        samples, processed, t_end = [], 0, time.perf_counter() + a.soak_seconds
        bs = 8
        imgs = [pool[i % len(pool)] for i in range(bs)]
        while time.perf_counter() < t_end:
            model.embed_batch(imgs)
            processed += bs
            samples.append(rss_mb())
        end_rss = rss_mb()
        res["memory_stability"] = {
            "seconds": a.soak_seconds,
            "images_processed": processed,
            "rss_start_mb": round(start_rss, 1),
            "rss_end_mb": round(end_rss, 1),
            "rss_growth_mb": round(end_rss - start_rss, 1),
            "rss_max_mb": round(max(samples), 1),
            "verdict": "стабильна" if end_rss - start_rss < 50 else "РОСТ ПАМЯТИ",
        }
        res["peak_rss_mb"] = round(max(res["peak_rss_mb"], max(samples)), 1)

    # --- печать ---
    L = res["latency_batch1_ms"]
    print("\n" + "=" * 62)
    print(f"{'МЕТРИКА':<38}{'ЗНАЧЕНИЕ':>24}")
    print("=" * 62)
    print(f"{'модель':<38}{res['model']:>24}")
    print(f"{'устройство':<38}{res['device']:>24}")
    print(f"{'размерность вектора':<38}{res['dim']:>24}")
    print(f"{'размер весов, МБ':<38}{res['weights_mb']:>24}")
    print("-" * 62)
    print(f"{'latency батч=1, p50, мс':<38}{L['p50']:>24}")
    print(f"{'latency батч=1, p95, мс':<38}{L['p95']:>24}")
    print(f"{'latency батч=1, p99, мс':<38}{L['p99']:>24}")
    print("-" * 62)
    for bs, fps in res["throughput_fps"].items():
        print(f"{'FPS, батч=' + bs:<38}{fps:>24}")
    print("-" * 62)
    print(f"{'пиковая RAM, МБ':<38}{res['peak_rss_mb']:>24}")
    if "peak_vram_mb" in res:
        print(f"{'пиковая VRAM, МБ':<38}{res['peak_vram_mb']:>24}")
    if "memory_stability" in res:
        m = res["memory_stability"]
        print(f"{'память за ' + str(m['seconds']) + ' с, МБ':<38}"
              f"{str(m['rss_start_mb']) + ' -> ' + str(m['rss_end_mb']):>24}")
        print(f"{'вердикт по памяти':<38}{m['verdict']:>24}")
    print("=" * 62)

    limit = 2048
    print(f"\nлимит жюри по весам: {limit} МБ — "
          f"{'влезаем' if res['weights_mb'] <= limit else 'НЕ ВЛЕЗАЕМ'} "
          f"({res['weights_mb']:.0f} МБ)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
