"""Сервис цифрового признака ТС — кейс 17, ЛЦТ 2026.

Контур: картинка + bbox -> вектор -> поиск ближайших в pgvector -> кандидаты или отказ.
"""
import csv
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from . import db
from .config import settings
from .imaging import ImageError, crop_with_pad, load_image, validate_bbox
from .models.registry import build_model
from .schemas import (
    BulkImportRequest,
    BulkImportResponse,
    EmbedResponse,
    GalleryAddResponse,
    GalleryListResponse,
    HealthResponse,
    ReadyResponse,
    SearchResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("case17")


def _validate_runtime_config() -> None:
    """Проверить опасные настройки до загрузки тяжёлой модели."""
    errors = []
    if settings.input_size < 32:
        errors.append(f"INPUT_SIZE должен быть не меньше 32, получено {settings.input_size}")
    if settings.embed_dim < 1:
        errors.append(f"EMBED_DIM должен быть положительным, получено {settings.embed_dim}")
    if settings.batch_size < 1:
        errors.append(f"BATCH_SIZE должен быть положительным, получено {settings.batch_size}")
    if not 1 <= settings.search_top_n <= 100:
        errors.append(f"SEARCH_TOP_N должен быть от 1 до 100, получено {settings.search_top_n}")
    for name, value in (
        ("SEARCH_THRESHOLD", settings.search_threshold),
        ("SEARCH_THRESHOLD_STRICT", settings.search_threshold_strict),
    ):
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} должен быть от 0 до 1, получено {value}")
    if settings.search_threshold_strict < settings.search_threshold:
        errors.append(
            "SEARCH_THRESHOLD_STRICT должен быть не меньше SEARCH_THRESHOLD"
        )

    if settings.embed_model.startswith("reid:"):
        weights = Path(os.getenv("REID_WEIGHTS", "/models/reid_current.pt"))
        if not weights.is_file():
            errors.append(f"файл REID_WEIGHTS не найден: {weights}")
        else:
            with weights.open("rb") as fh:
                prefix = fh.read(128)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                errors.append(
                    f"REID_WEIGHTS — это Git LFS pointer, а не веса: {weights}; "
                    "нужно скачать LFS-объект перед запуском"
                )

    if errors:
        raise RuntimeError("ошибка конфигурации: " + "; ".join(errors))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_runtime_config()
    t0 = time.time()
    log.info("загружаю модель %s ...", settings.embed_model)
    log.info(
        "конфигурация: input=%d dim=%d batch=%d tta_flip=%s threshold=%.4f strict=%.4f",
        settings.input_size,
        settings.embed_dim,
        settings.batch_size,
        settings.tta_flip,
        settings.search_threshold,
        settings.search_threshold_strict,
    )
    model = build_model(settings.embed_model, settings.input_size)
    model.load()
    app.state.model = model
    app.state.model_loaded = False
    log.info("модель загружена за %.1f с, размерность %d", time.time() - t0, model.dim)

    if model.dim != settings.embed_dim:
        log.warning(
            "EMBED_DIM=%d не совпадает с реальной размерностью модели %d, использую реальную",
            settings.embed_dim, model.dim,
        )
    db.init_pool()
    db.ensure_schema(model.dim)
    stored = db.stored_dim()
    if stored is not None and stored != model.dim:
        log.error(
            "В базе вектор размерности %d, а модель даёт %d. "
            "Нужно пересоздать том базы (docker compose down -v) после смены модели.",
            stored, model.dim,
        )
    app.state.model_loaded = True
    log.info("сервис готов")
    yield
    app.state.model_loaded = False
    del app.state.model
    db.close_pool()


app = FastAPI(
    title="Кейс 17 — цифровой признак ТС",
    description=(
        "Сопоставление снимков одного автомобиля с разных камер без госномера.\n\n"
        "Модель спрятана за интерфейсом EmbeddingModel и меняется переменной EMBED_MODEL."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Логировать метрики запроса без содержимого изображений и секретов."""
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("запрос завершился исключением: %s %s", request.method, request.url.path)
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info(
        "запрос %s %s -> %d за %.1f мс",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


def _model():
    """Вернуть модель, загруженную lifespan-ом приложения."""
    model = getattr(app.state, "model", None)
    if model is None or not getattr(app.state, "model_loaded", False):
        raise HTTPException(503, "Модель ещё не загружена.")
    return model


@app.exception_handler(ImageError)
async def image_error_handler(request, exc: ImageError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def _read_upload(image: UploadFile) -> bytes:
    raw = await image.read()
    if len(raw) > settings.max_upload_bytes:
        raise ImageError(
            f"Файл слишком большой: {len(raw) / 1024 / 1024:.1f} МБ, лимит {settings.max_upload_mb} МБ."
        )
    return raw


def _prepare(raw: bytes, x: int, y: int, w: int, h: int) -> Image.Image:
    img = load_image(raw, settings.max_upload_bytes)
    validate_bbox(img, x, y, w, h)
    return crop_with_pad(img, x, y, w, h, settings.bbox_pad)


@app.get("/health", response_model=HealthResponse, tags=["служебные"])
def health() -> HealthResponse:
    """Живой ли процесс. Не трогает базу."""
    model = getattr(app.state, "model", None)
    return HealthResponse(
        status="ok",
        model=model.name if model is not None else settings.embed_model,
        dim=model.dim if model is not None else settings.embed_dim,
    )


@app.get("/ready", response_model=ReadyResponse, tags=["служебные"])
def ready() -> ReadyResponse:
    """Готов ли сервис обслуживать запросы: модель загружена и база отвечает."""
    db_ok, size, detail = True, None, None
    try:
        size = db.count()
    except Exception as exc:
        db_ok, detail = False, str(exc)
    ok = getattr(app.state, "model_loaded", False) and db_ok
    return ReadyResponse(
        status="ready" if ok else "not_ready",
        model_loaded=getattr(app.state, "model_loaded", False),
        db_ok=db_ok,
        gallery_size=size,
        detail=detail,
    )


@app.get("/v1/config", tags=["служебные"])
def service_config() -> dict:
    """Рабочие точки порога и что они означают.

    Интерфейс берёт пороги отсюда, а не хранит у себя: они привязаны к
    конкретной модели, и при смене весов не должно оставаться зашитых чисел.
    """
    return {
        "model": _model().name,
        "dim": _model().dim,
        "explain_available": hasattr(_model(), "patch_saliency"),
        "modes": {
            "balanced": {
                "threshold": settings.search_threshold,
                "title": "сбалансированный",
                "note": "Обе ошибки равноценны. Точка максимума индекса Юдена: "
                        "F1 0.946, точность 0.962, полнота 0.931, TNR 0.964.",
            },
            "strict": {
                "threshold": settings.search_threshold_strict,
                "title": "строгий",
                "note": "Ложное совпадение дороже пропуска: TNR 0.984, "
                        "точность 0.982, полнота 0.893. Из шестидесяти чужих "
                        "запросов ошибочно принимается примерно один.",
            },
        },
    }


@app.post("/v1/explain", tags=["поиск"])
async def explain(
    image: UploadFile = File(...),
    x: int = Form(...), y: int = Form(...), w: int = Form(...), h: int = Form(...),
    gallery_id: Optional[int] = Form(None),
) -> dict:
    """Показать, какие участки кадра дали сходство.

    Без gallery_id карта показывает, что формирует сам признак. С gallery_id —
    что именно совпало с конкретным кандидатом из галереи.

    Величина на карте — тот же косинус, по которому принимается решение,
    посчитанный по участкам кадра. Поэтому это не догадка о том, «куда
    смотрит модель», а разложение самой оценки сходства.
    """
    if not hasattr(_model(), "patch_saliency"):
        raise HTTPException(
            501,
            f"Модель {_model().name} не умеет объяснять: карта строится по токенам "
            f"трансформера, а этот бэкбон их не отдаёт.",
        )
    t0 = time.perf_counter()
    crop = _prepare(await _read_upload(image), x, y, w, h)

    target = None
    if gallery_id is not None:
        target = db.fetch_vector(gallery_id)
        if target is None:
            raise HTTPException(404, f"В галерее нет записи {gallery_id}.")

    weights, side = _model().patch_saliency(crop, target)
    lo, hi = float(weights.min()), float(weights.max())
    return {
        "grid": side,
        "weights": [round(float(v), 4) for v in weights],
        "min": round(lo, 4),
        "max": round(hi, 4),
        "gallery_id": gallery_id,
        "model": _model().name,
        "took_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.post("/v1/embed", response_model=EmbedResponse, tags=["признак"])
async def embed(
    image: UploadFile = File(..., description="JPEG, PNG, WEBP или BMP"),
    x: int = Form(...), y: int = Form(...), w: int = Form(...), h: int = Form(...),
) -> EmbedResponse:
    """Посчитать цифровой признак автомобиля по кадру и его bbox."""
    t0 = time.perf_counter()
    crop = _prepare(await _read_upload(image), x, y, w, h)
    vec = _model().embed_batch([crop])[0]
    return EmbedResponse(
        model=_model().name, dim=_model().dim,
        embedding=[round(float(v), 6) for v in vec],
        took_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/v1/gallery", response_model=GalleryAddResponse, status_code=201, tags=["галерея"])
async def gallery_add(
    image: UploadFile = File(...),
    x: int = Form(...), y: int = Form(...), w: int = Form(...), h: int = Form(...),
    vehicle_id: Optional[str] = Form(None),
) -> GalleryAddResponse:
    """Добавить автомобиль в галерею известных."""
    t0 = time.perf_counter()
    crop = _prepare(await _read_upload(image), x, y, w, h)
    vec = _model().embed_batch([crop])[0]
    new_id = db.insert_one(vehicle_id, image.filename or "upload", [x, y, w, h], vec)
    return GalleryAddResponse(
        id=new_id, vehicle_id=vehicle_id, dim=_model().dim,
        took_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/v1/gallery/bulk-import", response_model=BulkImportResponse, tags=["галерея"])
def gallery_bulk_import(req: BulkImportRequest) -> BulkImportResponse:
    """Массовый импорт: папка с кропами + csv с метаданными.

    Пути обязаны лежать внутри DATA_DIR — наружу файловой системы не выпускаем.
    """
    data_root = Path(settings.data_dir).resolve()
    images_dir = Path(req.images_dir).resolve()
    csv_path = Path(req.csv_path).resolve()
    for p in (images_dir, csv_path):
        if not str(p).startswith(str(data_root)):
            raise HTTPException(400, f"Путь {p} вне разрешённой папки {data_root}.")
    if not images_dir.is_dir():
        raise HTTPException(404, f"Папка не найдена: {images_dir}")
    if not csv_path.is_file():
        raise HTTPException(404, f"CSV не найден: {csv_path}")

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if req.limit:
        rows = rows[: req.limit]
    if "image_id" not in (rows[0] if rows else {}):
        raise HTTPException(422, "В CSV нет колонки image_id.")

    if req.truncate:
        with db.pool().connection() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE gallery RESTART IDENTITY")
            conn.commit()

    t0 = time.perf_counter()
    imported, skipped, errors = 0, 0, []
    batch_imgs, batch_meta = [], []

    def flush():
        nonlocal imported, batch_imgs, batch_meta
        if not batch_imgs:
            return
        vecs = _model().embed_batch(batch_imgs)
        imported += db.insert_many(
            [(m[0], m[1], m[2], vecs[i]) for i, m in enumerate(batch_meta)]
        )
        batch_imgs, batch_meta = [], []

    for row in rows:
        path = images_dir / f"{row['image_id']}.jpg"
        if not path.is_file():
            skipped += 1
            if len(errors) < 10:
                errors.append(f"нет файла {path.name}")
            continue
        try:
            img = Image.open(path).convert("RGB")
            if not req.already_cropped:
                bx = [int(row[k]) for k in ("x", "y", "w", "h")]
                validate_bbox(img, *bx)
                img = crop_with_pad(img, *bx, settings.bbox_pad)
            bbox = [int(row.get(k, 0) or 0) for k in ("x", "y", "w", "h")]
            batch_imgs.append(img)
            batch_meta.append((row.get("vehicle_id") or None, str(path), bbox))
            if len(batch_imgs) >= settings.batch_size:
                flush()
        except Exception as exc:
            skipped += 1
            if len(errors) < 10:
                errors.append(f"{path.name}: {exc}")
    flush()

    took = time.perf_counter() - t0
    return BulkImportResponse(
        imported=imported, skipped=skipped, errors=errors,
        took_s=round(took, 2),
        images_per_sec=round(imported / took, 2) if took > 0 else 0.0,
    )


@app.post("/v1/search", response_model=SearchResponse, tags=["поиск"])
async def search(
    image: UploadFile = File(...),
    x: int = Form(...), y: int = Form(...), w: int = Form(...), h: int = Form(...),
    top_n: Optional[int] = Form(None),
    threshold: Optional[float] = Form(None),
) -> SearchResponse:
    """Найти кандидатов на то же ТС.

    Если самый похожий кандидат не дотягивает до порога — честный отказ
    вместо выдумывания ответа. Это требование кейса: лучше сказать
    «уверенного совпадения нет», чем подсунуть чужую машину.
    """
    model = _model()
    t0 = time.perf_counter()
    n = top_n or settings.search_top_n
    thr = settings.search_threshold if threshold is None else threshold
    if n < 1 or n > 100:
        raise HTTPException(422, "top_n должен быть от 1 до 100.")
    if not 0.0 <= thr <= 1.0:
        raise HTTPException(422, "threshold должен быть от 0 до 1.")

    crop = _prepare(await _read_upload(image), x, y, w, h)
    vec = _model().embed_batch([crop])[0]

    if db.count() == 0:
        took = round((time.perf_counter() - t0) * 1000, 1)
        log.info(
            "поиск: gallery=0 threshold=%.4f refusal=gallery_empty took_ms=%.1f",
            thr,
            took,
        )
        return SearchResponse(
            refusal=True, reason="gallery_empty",
            message="Галерея пуста, сравнивать не с чем.",
            threshold=thr, top_n=n, model=model.name, took_ms=took,
        )

    found = db.search(vec, n)
    took = round((time.perf_counter() - t0) * 1000, 1)
    best = found[0]["confidence"] if found else 0.0

    if best < thr:
        log.info(
            "поиск: returned=%d threshold=%.4f best=%.4f refusal=low_similarity "
            "candidates=0 took_ms=%.1f",
            len(found),
            thr,
            best,
            took,
        )
        return SearchResponse(
            refusal=True, reason="low_similarity",
            message=(
                f"Уверенного совпадения нет: лучший кандидат {best:.3f} "
                f"при пороге {thr:.3f}."
            ),
            threshold=thr, top_n=n, model=model.name, took_ms=took, candidates=[],
        )

    log.info(
        "поиск: returned=%d threshold=%.4f best=%.4f refusal=false candidates=%d took_ms=%.1f",
        len(found),
        thr,
        best,
        len(found),
        took,
    )
    return SearchResponse(
        refusal=False, threshold=thr, top_n=n, model=model.name,
        took_ms=took, candidates=found,
    )


@app.get("/v1/gallery", response_model=GalleryListResponse, tags=["галерея"])
def gallery_list(limit: int = 50, offset: int = 0) -> GalleryListResponse:
    """Что лежит в галерее."""
    if limit < 1 or limit > 500:
        raise HTTPException(422, "limit должен быть от 1 до 500.")
    return GalleryListResponse(
        total=db.count(), limit=limit, offset=offset, items=db.list_items(limit, offset)
    )


@app.delete("/v1/gallery/{item_id}", tags=["галерея"])
def gallery_delete(item_id: int):
    """Удалить запись из галереи."""
    if not db.delete_one(item_id):
        raise HTTPException(404, f"Записи с id={item_id} нет.")
    return {"deleted": item_id}
