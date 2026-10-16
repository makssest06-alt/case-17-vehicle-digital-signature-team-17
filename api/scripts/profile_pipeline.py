#!/usr/bin/env python3
"""Разложить путь запроса на этапы и замерить каждый.

Замеряется один прямой проход. Сервис на каждый кадр делает два (оригинал и
отражение, результат усредняется), поэтому абсолютная задержка в бою вдвое
выше — её меряет benchmark.py. Здесь важны доли этапов, а не абсолют.

    docker compose exec api python scripts/profile_pipeline.py

Даёт таблицу p50/p95 по этапам — на что реально уходит время между
«пришёл кадр» и «ушёл ответ». Нужна, чтобы оптимизировать по делу,
а не по интуиции.

Этапы:
  1. чтение файла с диска
  2. декодирование JPEG
  3. кроп по bbox
  4. ресайз
  5. нормализация (в тензор)
  6. модель
  7. постобработка (L2)
  8. поиск в галерее
"""
import argparse
import csv
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/srv")

from app import db                              # noqa: E402
# Быстрые пути есть только на ветке feat/fast-inference. Если их нет —
# работает эталонная реализация на Pillow и поиске в pgvector. Профиль
# меряет то, что в этой сборке реально исполняется, поэтому ДО и ПОСЛЕ
# сравниваются целиком, а не по одной только модели.
try:
    from app.fastimg import to_batch, decode as fast_decode, crop_then_resize  # noqa: E402
    HAVE_FAST_IMG = True
except ImportError:
    HAVE_FAST_IMG = False

    def to_batch(arrays, mean, std):
        b = np.stack(arrays).astype(np.float32, copy=False)
        b /= 255.0; b -= mean; b /= std
        return np.ascontiguousarray(b.transpose(0, 3, 1, 2))

try:
    from app.gallery_index import GalleryIndex  # noqa: E402
    HAVE_FAST_INDEX = True
except ImportError:
    HAVE_FAST_INDEX = False
from app.config import settings                 # noqa: E402
from app.models.registry import build_model     # noqa: E402

CROPS = Path(settings.data_dir) / "crops"


class Timer:
    def __init__(self):
        self.stages = {}

    def add(self, name, ms):
        self.stages.setdefault(name, []).append(ms)

    def report(self):
        out = {}
        for k, v in self.stages.items():
            v = sorted(v)
            out[k] = {
                "p50": round(v[len(v) // 2], 3),
                "p95": round(v[min(len(v) - 1, int(0.95 * (len(v) - 1)))], 3),
                "mean": round(st.mean(v), 3),
                "n": len(v),
            }
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="/artifacts/profile_before.json")
    ap.add_argument("--label", default="before")
    a = ap.parse_args()

    rows = list(csv.DictReader((CROPS / "crops_query.csv").open()))[: a.n + a.warmup]
    model = build_model(settings.embed_model, settings.input_size)
    t0 = time.perf_counter()
    model.load()
    print(f"модель {model.name} загружена за {time.perf_counter() - t0:.1f} с")

    # доступ к трансформу модели, чтобы разделить ресайз и нормализацию
    tf = getattr(model, "_tf", None)

    have_db = True
    try:
        db.init_pool()
        gallery_size = db.count()
    except Exception as e:
        have_db = False
        gallery_size = 0
        print(f"база недоступна ({e}) — этап поиска пропускаю")

    index = None
    if HAVE_FAST_INDEX and have_db and gallery_size:
        index = GalleryIndex()
        t_ix = time.perf_counter()
        index.rebuild(db.fetch_all())
        print(f"in-memory индекс собран за {(time.perf_counter() - t_ix) * 1000:.0f} мс, "
              f"{index.size} векторов")

    print(f"быстрое декодирование: {'да' if HAVE_FAST_IMG else 'нет'} | "
          f"in-memory индекс: {'да' if index is not None else 'нет'}")

    T = Timer()
    for idx, r in enumerate(rows):
        path = CROPS / "query" / f"{r['image_id']}.jpg"
        w, h = int(r["out_w"]), int(r["out_h"])
        warm = idx < a.warmup

        t = time.perf_counter(); raw = path.read_bytes()
        d_read = (time.perf_counter() - t) * 1000

        if HAVE_FAST_IMG:
            t = time.perf_counter(); arr = fast_decode(raw)
            d_decode = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            resized = crop_then_resize(arr, (0, 0, w, h), settings.input_size)
            d_crop_resize = (time.perf_counter() - t) * 1000
            crop = None
        else:
            t = time.perf_counter(); img = Image.open(io.BytesIO(raw)); img = img.convert("RGB")
            d_decode = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            crop = img.crop((0, 0, w, h))
            resized = crop.resize((settings.input_size, settings.input_size), Image.BILINEAR)
            d_crop_resize = (time.perf_counter() - t) * 1000

        # Нормализация у двух реализаций своя: torch-путь идёт через transform
        # из timm, ONNX-путь — через векторный to_batch. Меряем ту, которая
        # реально работает в этой сборке, иначе этап выпадает из сравнения.
        t = time.perf_counter()
        if tf is not None:
            x = tf(crop if crop is not None else Image.fromarray(resized)).unsqueeze(0)
        else:
            x = torch.from_numpy(to_batch([np.asarray(resized)], model._mean, model._std))
        d_norm = (time.perf_counter() - t) * 1000

        # Путь через torch (ветка research-baselines) и через ONNX Runtime
        # (ветка feat/fast-inference) меряются одним и тем же кодом, иначе
        # сравнение ДО/ПОСЛЕ ничего не стоит.
        t = time.perf_counter()
        if getattr(model, "_sess", None) is not None:          # ONNX Runtime
            raw = model._sess.run([model._out], {model._in: x.numpy()})[0]
        else:                                                   # torch
            with torch.inference_mode():
                feats = model._model(x)
                # у обученного чекпоинта поверх бэкбона стоит голова: без неё
                # вектор остаётся сырым (768D вместо 512D) и поиск в галерее
                # падает на несовпадении размерности
                head = getattr(model, "_head", None)
                if head is not None:
                    out = head(feats)
                    feats = out[1] if isinstance(out, (tuple, list)) else out
            raw = feats.numpy()
        d_model = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        vec = raw[0].astype("float32")
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        d_post = (time.perf_counter() - t) * 1000

        d_search = 0.0
        if index is not None:
            t = time.perf_counter(); index.search(vec, 10)
            d_search = (time.perf_counter() - t) * 1000
        elif have_db and gallery_size:
            t = time.perf_counter(); db.search(vec, 10)
            d_search = (time.perf_counter() - t) * 1000

        if warm:
            continue
        T.add("1. чтение с диска", d_read)
        T.add("2. декодирование JPEG", d_decode)
        T.add("3. кроп и ресайз", d_crop_resize)
        T.add("4. нормализация", d_norm)
        T.add("5. модель", d_model)
        T.add("6. постобработка", d_post)
        if index is not None or (have_db and gallery_size):
            T.add("7. поиск в галерее", d_search)

    rep = T.report()
    total_p50 = sum(v["p50"] for v in rep.values())

    print(f"\n{'ЭТАП':<28}{'p50, мс':>10}{'p95, мс':>10}{'доля':>10}")
    print("-" * 58)
    for k in sorted(rep):
        v = rep[k]
        print(f"{k:<28}{v['p50']:>10.2f}{v['p95']:>10.2f}{v['p50']/total_p50*100:>9.1f}%")
    print("-" * 58)
    print(f"{'ИТОГО (сумма p50)':<28}{total_p50:>10.2f}")
    print(f"\nгалерея: {gallery_size} записей")

    out = {"label": a.label, "model": model.name, "gallery_size": gallery_size,
           "fast_decode": HAVE_FAST_IMG, "memory_index": index is not None,
           "total_p50_ms": round(total_p50, 2), "stages": rep}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
