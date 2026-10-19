#!/usr/bin/env python3
"""Эмбеддинги тем же кодом, которым отвечает сервис, с кэшем на диске.

Модель строится через api/app/models/registry.build_model — тот же путь,
что в api/app/main.py и api/scripts/make_submission.py. Своей копии
кропа, трансформа или головы здесь нет: если сервис поменяется, замер
поменяется вместе с ним.

Профиль — это набор переменных окружения сервиса:

  compose_default   docker compose up без .env: умолчания ${VAR:-...}
                    из docker-compose.yml;
  env_example       после cp .env.example .env: ключи модели из .env.example.

Значения читаются из самих файлов, а не вписаны сюда. Каждый профиль
считается в отдельном процессе: app/config.py читает окружение один раз
при импорте, и два профиля в одном процессе смешались бы.

Кэш: <cache>/<profile>-<ключ>.npz, ключ = sha256(профиль, sha256 весов,
список image_id, папка кропов). По умолчанию кэш вне репозитория:
~/lct-case17/cache/honest_eval (переопределяется HONEST_EVAL_CACHE).

    python research/service_embeddings.py --profile env_example \\
        --images-dir /opt/case17/data/crops/gallery --ids-csv test_gallery.csv
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
API = ROOT / "api"
DEFAULT_CACHE = Path(os.getenv("HONEST_EVAL_CACHE", Path.home() / "lct-case17" / "cache" / "honest_eval"))

# Переменные, от которых зависит вектор и решение об отказе. Пароли БД и
# прочее из .env.example сюда не попадают и нигде не печатаются.
MODEL_KEYS = ("EMBED_MODEL", "EMBED_DIM", "INPUT_SIZE", "EMBED_TTA_FLIP", "REID_WEIGHTS",
              "REID_TF_MODE", "SEARCH_THRESHOLD", "SEARCH_THRESHOLD_STRICT", "MODELS_HOST_DIR")
PROFILES = ("compose_default", "env_example")


def compose_defaults(ref: str = "") -> dict:
    """Умолчания вида KEY: ${KEY:-value} — то, что получит сервис без .env.

    ref — коммит, из которого взять docker-compose.yml (например 0924fb2);
    пусто — рабочее дерево. Нужен, чтобы замер по старому main оставался
    воспроизводимым после того, как умолчания в ветке поправили.
    """
    if ref:
        text = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:docker-compose.yml"],
                              check=True, capture_output=True, text=True).stdout
    else:
        text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    out = {}
    for key in MODEL_KEYS:
        m = re.search(rf"\$\{{{key}:-([^}}]*)\}}", text)
        if m:
            out[key] = m.group(1)
    return out


def env_example(path: Path = ROOT / ".env.example") -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in MODEL_KEYS:
            out[key.strip()] = value.strip()
    return out


def profile_env(name: str, compose_ref: str = "", overrides: dict | None = None) -> dict:
    """Окружение сервиса для профиля, с путём /models/* переведённым на хост.

    В контейнере веса лежат в /models, а это том ${MODELS_HOST_DIR}. Для
    compose_default в main 0924fb2 это /opt/case17/models (на чистом клоне
    его нет), для env_example — ./weights репозитория.
    """
    if name == "compose_default":
        env = compose_defaults(compose_ref)
    elif name == "env_example":
        env = env_example()
    else:
        raise ValueError(f"неизвестный профиль {name!r}, есть: {PROFILES}")
    # Новая модель меряется тем же путём: профиль env_example плюс, например,
    # REID_WEIGHTS=/путь/к/весам.pt. Переопределения входят в ключ кэша.
    env.update(overrides or {})
    host = Path(env.pop("MODELS_HOST_DIR", "/opt/case17/models"))
    if not host.is_absolute():
        host = (ROOT / host).resolve()
    w = env.get("REID_WEIGHTS", "")
    if w.startswith("/models/"):
        env["REID_WEIGHTS"] = str(host / w[len("/models/"):])
    return env


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_sha(env: dict) -> str:
    """sha256 весов, которые реально загрузятся; '' если модель их не читает."""
    if not env.get("EMBED_MODEL", "").startswith("reid:"):
        return ""
    p = Path(env["REID_WEIGHTS"])
    return _sha256(p) if p.exists() else "missing"


def _api_files() -> list:
    return sorted((API / "app").rglob("*.py"))


def code_hash() -> str:
    """sha256 кода сервиса api/app/**/*.py: другой код — другой кэш."""
    h = hashlib.sha256()
    for p in _api_files():
        h.update(str(p.relative_to(API)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def service_env_keys() -> set:
    """Все переменные окружения, которые читает код сервиса (getenv и _i/_f/_b
    в app/config.py). Из окружения дочернего процесса они вычищаются, чтобы
    случайная переменная оболочки не попала в замер мимо ключа кэша."""
    keys = {"OMP_NUM_THREADS"}
    for p in _api_files():
        text = p.read_text(encoding="utf-8")
        keys |= set(re.findall(r'getenv\(\s*"([A-Z0-9_]+)"', text))
        keys |= set(re.findall(r'\b_[ifb]\(\s*"([A-Z0-9_]+)"', text))
    return keys


def _key(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def cache_path(profile: str, env: dict, ids: list, images_dir: Path, cache_dir: Path,
               frames: str = "") -> Path:
    spec = {"profile": profile, "env": env, "weights_sha": weights_sha(env),
            "images_dir": str(images_dir), "ids": ids, "code": code_hash()}
    if frames:
        spec["frames"] = frames
    return cache_dir / f"{profile}-{_key(spec)}.npz"


def _legacy_cache_path(profile: str, env: dict, ids: list, images_dir: Path, cache_dir: Path,
                       frames: str = "") -> Path:
    """Ключ кэша до 17.10, без хэша кода."""
    spec = {"profile": profile, "env": env, "weights_sha": weights_sha(env),
            "images_dir": str(images_dir), "ids": ids}
    if frames:
        spec["frames"] = frames
    return cache_dir / f"{profile}-{_key(spec)}.npz"


def _adopt_legacy(legacy: Path, out: Path) -> bool:
    """Старый кэш принимается, только если создан ПОСЛЕ последнего изменения
    кода сервиса (mtime файлов api/app): тогда считал тот же код. Иначе —
    пересчёт. Старый файл не удаляется."""
    if not legacy.exists():
        return False
    if legacy.stat().st_mtime <= max(p.stat().st_mtime for p in _api_files()):
        return False
    z = np.load(legacy, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    meta.update(code=code_hash(), adopted_from=legacy.name)
    np.savez(out, emb=z["emb"], ids=z["ids"], meta=json.dumps(meta, ensure_ascii=False))
    print(f"  кэш {legacy.name} принят как {out.name}: создан после последнего изменения api/app", flush=True)
    return True


def embed(profile: str, images_dir: Path, ids: list, cache_dir: Path = DEFAULT_CACHE,
          batch: int = 16, threads: int = 0, compose_ref: str = "",
          overrides: dict | None = None, frames_zip: Path | None = None,
          boxes: dict | None = None) -> np.ndarray:
    """Векторы (len(ids), dim) float32, L2-нормированы — как отдаёт сервис.

    Источник кадра — готовые кропы images_dir/<id>.jpg (так читает
    make_submission.py) или, при frames_zip, полный кадр из dataset.zip,
    обрезанный функциями сервиса app.imaging.load_image → validate_bbox →
    crop_with_pad (так отвечает /v1/embed); при EXTRACT_JPEG_DRAFT=1/2 кроп идёт
    через app.extractor.Extractor.crop — тот же draft-декод, что у прод-сервиса.
    boxes: {image_id: (x, y, w, h)}.
    Кадр, чей bbox сервис отверг, получает вектор из NaN: сервис такой запрос
    не принял бы, и вызывающий код обязан его выкинуть, а не считать нулём.
    """
    env = profile_env(profile, compose_ref, overrides)
    frames = f"zip:{frames_zip}" if frames_zip else ""
    if frames_zip:        # draft меняет векторы — входит в ключ кэша и в окружение воркера
        frames += f":draft={os.getenv('EXTRACT_JPEG_DRAFT', '0')}"
    out = cache_path(profile, env, ids, images_dir, cache_dir, frames)
    if not out.exists():
        _adopt_legacy(_legacy_cache_path(profile, env, ids, images_dir, cache_dir, frames), out)
    if out.exists():
        z = np.load(out, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if list(z["ids"]) == list(ids) and meta.get("code") == code_hash() and meta.get("profile") == profile:
            return z["emb"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(ids))
        ids_file = f.name
    scrub = service_env_keys()
    child = {k: v for k, v in os.environ.items() if k not in scrub}
    child.update(env)
    # Число потоков меняет только скорость (и порядок суммирования ~1e-7),
    # не результат; сервис в контейнере работает на двух.
    child["OMP_NUM_THREADS"] = str(threads or os.cpu_count() or 2)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--profile", profile,
           "--images-dir", str(images_dir), "--ids-file", ids_file, "--out", str(out),
           "--batch", str(batch)]
    boxes_file = None
    if frames_zip:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({i: list(boxes[i]) for i in ids}, f)
            boxes_file = f.name
        cmd += ["--frames-zip", str(frames_zip), "--boxes-file", boxes_file]
    try:
        subprocess.run(cmd, env=child, check=True)
    finally:
        os.unlink(ids_file)
        if boxes_file:
            os.unlink(boxes_file)
    return np.load(out, allow_pickle=False)["emb"]


def _worker(a) -> int:
    """Выполняется в дочернем процессе с окружением профиля."""
    from PIL import Image

    sys.path.insert(0, str(API))
    from app.config import settings  # noqa: E402  читает окружение профиля
    from app.models.registry import build_model  # noqa: E402

    ids = Path(a.ids_file).read_text().split()
    model = build_model(settings.embed_model, settings.input_size)
    model.load()
    emb = np.zeros((len(ids), model.dim), dtype="float32")
    rejected = []
    if a.frames_zip:
        import zipfile

        from app.imaging import ImageError, crop_with_pad, load_image, validate_bbox  # noqa: E402

        zf = zipfile.ZipFile(a.frames_zip)
        boxes = json.loads(Path(a.boxes_file).read_text())

        if os.getenv("EXTRACT_JPEG_DRAFT", "0").strip().lower() not in ("0", "false", "no", "off"):
            from app.extractor import Extractor  # noqa: E402

            _ex = Extractor(model=model, threads=1)

            def load(i):
                """Путь прод-сервиса с draft: байты кадра -> Extractor.crop (декод с уменьшением)."""
                return _ex.crop(zf.read(f"images/{i}.jpg"), [int(float(v)) for v in boxes[i]])
        else:
            def load(i):
                """Ровно путь /v1/embed: байты кадра -> load_image -> validate_bbox -> crop_with_pad."""
                img = load_image(zf.read(f"images/{i}.jpg"), settings.max_upload_bytes)
                x, y, w, h = (int(float(v)) for v in boxes[i])
                try:
                    validate_bbox(img, x, y, w, h)
                except ImageError as exc:
                    rejected.append((i, str(exc)))
                    return None
                return crop_with_pad(img, x, y, w, h, settings.bbox_pad)
    else:
        def load(i):
            return Image.open(Path(a.images_dir) / f"{i}.jpg").convert("RGB")
    t0 = time.perf_counter()
    for s in range(0, len(ids), a.batch):
        chunk = ids[s:s + a.batch]
        imgs = [load(i) for i in chunk]
        keep = [k for k, im in enumerate(imgs) if im is not None]
        if keep:
            emb[[s + k for k in keep]] = model.embed_batch([imgs[k] for k in keep])
        emb[[s + k for k, im in enumerate(imgs) if im is None]] = np.nan
        done = s + len(chunk)
        if done % (a.batch * 20) == 0 or done == len(ids):
            print(f"  [{a.profile}] {done}/{len(ids)}  "
                  f"{done / (time.perf_counter() - t0):.1f} кадров/с", flush=True)
    meta = {"profile": a.profile, "model": model.name, "dim": int(model.dim),
            "tta_flip": bool(settings.tta_flip), "tf_mode": getattr(model, "tf_mode", None),
            "env": {k: os.environ.get(k, "") for k in MODEL_KEYS},
            "source": f"zip:{a.frames_zip}" if a.frames_zip else str(a.images_dir),
            "code": code_hash(),
            "rejected_bbox": rejected,
            "seconds": round(time.perf_counter() - t0, 1)}
    if rejected:
        print(f"  [{a.profile}] сервис отверг bbox у {len(rejected)} кадров, "
              f"например {rejected[0]}", flush=True)
    np.savez(a.out, emb=emb, ids=np.array(ids), meta=json.dumps(meta, ensure_ascii=False))
    print(f"  [{a.profile}] -> {a.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--ids-csv", type=Path, help="CSV с колонкой image_id")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--compose-ref", default="", help="коммит для docker-compose.yml; пусто — рабочее дерево")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL поверх профиля, например REID_WEIGHTS=...")
    ap.add_argument("--show-env", action="store_true", help="напечатать окружение профиля и выйти")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--ids-file", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--frames-zip", help=argparse.SUPPRESS)
    ap.add_argument("--boxes-file", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return _worker(a)
    overrides = dict(kv.split("=", 1) for kv in a.env)
    if a.show_env:
        print(json.dumps(profile_env(a.profile, a.compose_ref, overrides), ensure_ascii=False, indent=2))
        return 0
    if not a.ids_csv:
        ap.error("нужен --ids-csv")
    ids = [r["image_id"] for r in csv.DictReader(a.ids_csv.open(encoding="utf-8"))]
    e = embed(a.profile, a.images_dir, ids, a.cache_dir, a.batch, compose_ref=a.compose_ref,
              overrides=overrides)
    print(f"{a.profile}: {e.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
