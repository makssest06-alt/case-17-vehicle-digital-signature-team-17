#!/usr/bin/env python3
"""FPS и латентность конвейера сервиса с постобработкой выключенной и включённой.

Конвейер — тот же код, что /v1/search, без HTTP: чтение и декод кадра
(imaging.load_image), кроп, embed_batch модели сервиса (батч b), поиск в pgvector,
постобработка (app.pp: DBA k=1 + штраф за ту же камеру) и решение об отказе.
Режимы:
  off  флаги PP_* выключены — db.search как сейчас;
  on   PP_DBA_K=1, PP_CAMERA_PENALTY=on, оценка --refusal-score.
Модель — EMBED_MODEL / REID_WEIGHTS контейнера (как у сервиса), --model — только
для пробного прогона.

Галерея строится в ОТДЕЛЬНОЙ схеме Postgres (--schema, по умолчанию bench_pp) той
же базы и в конце удаляется: галерея сервиса не трогается. Кадры галереи — первые
--gallery-limit строк --gallery-csv (полные кадры: признак фона считается из них),
запросы — --query-csv по кругу.

Обычно запускается через scripts/bench_fps.sh на машине с GPU.
"""
import argparse
import dataclasses
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "scripts"))
import benchmark_reid as br  # noqa: E402  (чтение CSV, синхронизация CUDA, сводка латентности)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, required=True, help="полные кадры <image_id>.jpg")
    ap.add_argument("--query-csv", type=Path, required=True, help="image_id,x,y,w,h запросов")
    ap.add_argument("--gallery-csv", type=Path, required=True, help="image_id,x,y,w,h кадров галереи")
    ap.add_argument("--gallery-limit", type=br.positive, default=10847)
    ap.add_argument("--samples", type=br.positive, default=256, help="запросов в круговом пуле")
    ap.add_argument("--batches", type=br.batch_sizes, default=br.batch_sizes("1,8"))
    ap.add_argument("--modes", default="off,on")
    ap.add_argument("--refusal-score", default="raw", choices=("raw", "z100"))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iterations", type=br.positive, default=100, help="батчей на точку")
    ap.add_argument("--schema", default="bench_pp")
    ap.add_argument("--keep-schema", action="store_true", help="не удалять схему с галереей после замера")
    ap.add_argument("--model", default=None, help="вместо EMBED_MODEL — только для пробного прогона")
    ap.add_argument("--json", type=Path, help="куда записать отчёт")
    ap.add_argument("--require-cuda", action="store_true")
    return ap


def with_schema(dsn: str, schema: str) -> str:
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}options=-csearch_path%3D{schema},public"


def build_gallery(model, pp, rows, batch: int) -> int:
    """Кадры галереи через app.pp (embedding, embedding_dba, bg) — так их добавил бы /v1/gallery."""
    from app import config
    from app.imaging import crop_with_pad, load_image, validate_bbox

    done = 0
    for s in range(0, len(rows), batch):
        items, crops = [], []
        for r in rows[s:s + batch]:
            frame = load_image(r.path.read_bytes(), config.settings.max_upload_bytes)
            validate_bbox(frame, *r.bbox)
            crops.append(crop_with_pad(frame, *r.bbox, config.settings.bbox_pad))
            items.append([None, r.path.name, list(r.bbox), None, frame])
        vecs = model.embed_batch(crops)
        for it, v in zip(items, vecs):
            it[3] = v
        pp.add_many([tuple(it) for it in items])
        done += len(items)
    return done


def one_batch(model, samples, mode: str, pp, settings):
    """Один батч запросов целиком: декод → кроп → модель → поиск → постобработка → отказ."""
    from app import db
    from app.imaging import crop_with_pad, load_image, validate_bbox

    frames, crops = [], []
    for smp in samples:
        frame = load_image(smp.path.read_bytes(), settings.max_upload_bytes)
        validate_bbox(frame, *smp.bbox)
        frames.append(frame)
        crops.append(crop_with_pad(frame, *smp.bbox, settings.bbox_pad))
    vecs = model.embed_batch(crops)
    refused = 0
    for smp, frame, vec in zip(samples, frames, vecs):
        if mode == "off":
            found = db.search(vec, settings.search_top_n)
            best = found[0]["confidence"] if found else 0.0
        else:
            found, best = pp.search(vec, settings.search_top_n, frame, smp.bbox)
        refused += best < settings.search_threshold
    return refused


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    import torch
    from app import config, db
    from app.models.registry import build_model
    from app.pp import state as pp_state

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("--require-cuda: CUDA недоступна")
    base = config.settings
    model = build_model(args.model or base.embed_model, base.input_size)
    model.load()
    device = getattr(model, "device", torch.device("cpu"))
    queries = br.read_samples(args.query_csv, args.images_dir, args.samples)
    gallery = br.read_samples(args.gallery_csv, args.images_dir, args.gallery_limit)

    on = dataclasses.replace(base, pp_dba_k=1, pp_camera_penalty="on", refusal_score=args.refusal_score,
                             db_dsn=with_schema(base.db_dsn, args.schema))
    config.settings = db.settings = on
    db.init_pool()
    try:
        with db.pool().connection() as conn, conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {args.schema} CASCADE")
            cur.execute(f"CREATE SCHEMA {args.schema}")
            conn.commit()
        db.ensure_schema(model.dim)
        pp = pp_state.Postprocess(on, model.dim, db)
        t0 = time.perf_counter()
        n_gal = build_gallery(model, pp, gallery, max(args.batches))
        build_s = time.perf_counter() - t0
        print(f"галерея: {n_gal} кадров за {build_s:.1f} с (схема {args.schema})", flush=True)

        results = []
        for mode in args.modes.split(","):
            st = on if mode == "on" else dataclasses.replace(base, db_dsn=on.db_dsn)
            for size in args.batches:
                batch = lambda i: [queries[(i * size + o) % len(queries)] for o in range(size)]
                for i in range(args.warmup):
                    one_batch(model, batch(i), mode, pp, st)
                times, refused = [], 0
                for i in range(args.iterations):
                    r, ms = br.timed(device, lambda: one_batch(model, batch(i), mode, pp, st))
                    times.append(ms)
                    refused += r
                lat = br.latency_summary(times)
                res = {"mode": mode, "batch_size": size, "p50_batch_ms": lat["p50_batch_ms"],
                       "p95_batch_ms": lat["p95_batch_ms"], "p50_per_image_ms": lat["p50_batch_ms"] / size,
                       "p95_per_image_ms": lat["p95_batch_ms"] / size,
                       "images_per_second": size * len(times) * 1000 / sum(times),
                       "refusal_share_at_container_threshold": refused / (size * len(times)), "batch_ms": times}
                results.append(res)
                print(f"{mode:3s} b{size:<3d} p50 {res['p50_batch_ms']:7.2f} мс  p95 {res['p95_batch_ms']:7.2f} мс  "
                      f"на кадр p50 {res['p50_per_image_ms']:6.2f} мс  FPS {res['images_per_second']:7.1f}", flush=True)
    finally:
        if not args.keep_schema:
            try:
                with db.pool().connection() as conn, conn.cursor() as cur:
                    cur.execute(f"DROP SCHEMA IF EXISTS {args.schema} CASCADE")
                    conn.commit()
            except Exception as exc:                     # схема останется — скажем об этом
                print(f"не удалось удалить схему {args.schema}: {exc}", file=sys.stderr)
        db.close_pool()
        config.settings = db.settings = base

    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=API_DIR, capture_output=True, text=True,
                             timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        rev = None
    report = {"metadata": {"python": platform.python_version(), "torch": torch.__version__,
                           "device": str(device),
                           "device_name": torch.cuda.get_device_name(device) if device.type == "cuda"
                           else platform.processor() or "CPU",
                           "model": model.name, "embed_model": args.model or base.embed_model,
                           "tta_flip": base.tta_flip, "top_n": base.search_top_n, "threshold": base.search_threshold,
                           "refusal_score_on": args.refusal_score, "camera_tau": on.pp_camera_tau,
                           "camera_topk": on.pp_camera_topk, "gallery_size": n_gal, "gallery_build_s": build_s,
                           "query_pool": len(queries), "torch_threads": torch.get_num_threads(), "git_revision": rev,
                           "protocol": "Батч запросов целиком: чтение, декод, кроп, embed_batch, поиск, постобработка, "
                                       "отказ; без HTTP. Галерея — во временной схеме той же базы."},
              "results": results}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
