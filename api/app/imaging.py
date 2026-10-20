"""Загрузка и валидация картинок, вырезание по bbox.

crop_frame — единственный кроп модели: им режут extract() (app/extractor.py), офлайн-сборка
(scripts/make_submission.py), сервис (/v1/*) и калибровка (research/service_embeddings.py), поэтому
векторы сервиса, artifacts/ и замера скорости (Q31) совпадают. JPEG декодируется сразу с уменьшением
(PIL draft, EXTRACT_JPEG_DRAFT, по умолчанию не больше 1/2).
"""
import io
import os
from typing import Tuple

from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_PIXELS = 80_000_000  # защита от decompression bomb


class ImageError(ValueError):
    """Понятная человеку ошибка входных данных."""


def open_image(raw: bytes, max_bytes: int) -> Image.Image:
    """Проверки load_image без декодирования пикселей: кадр открыт, draft ещё возможен (crop_frame)."""
    if not raw:
        raise ImageError("Файл пустой.")
    if len(raw) > max_bytes:
        raise ImageError(
            f"Файл слишком большой: {len(raw) / 1024 / 1024:.1f} МБ, "
            f"лимит {max_bytes / 1024 / 1024:.0f} МБ."
        )
    try:
        probe = Image.open(io.BytesIO(raw))
        fmt = (probe.format or "").upper()
        probe.verify()
    except UnidentifiedImageError:
        raise ImageError("Не удалось распознать изображение. Нужен JPEG, PNG, WEBP или BMP.")
    except Exception as exc:
        raise ImageError(f"Битое изображение: {exc}")

    if fmt not in ALLOWED_FORMATS:
        raise ImageError(f"Формат {fmt or 'неизвестный'} не поддерживается. Нужен JPEG, PNG, WEBP или BMP.")

    image = Image.open(io.BytesIO(raw))
    if image.width * image.height > MAX_PIXELS:
        raise ImageError("Изображение слишком большое по числу пикселей.")
    return image


def load_image(raw: bytes, max_bytes: int) -> Image.Image:
    """Полный кадр RGB после проверок — для признака фона (app/pp/background.py калибровался по нему)."""
    return open_image(raw, max_bytes).convert("RGB")


def validate_bbox(image: Image.Image, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    if w <= 0 or h <= 0:
        raise ImageError(f"Ширина и высота bbox должны быть больше нуля, получено w={w}, h={h}.")
    if x < 0 or y < 0:
        raise ImageError(f"Координаты bbox не могут быть отрицательными, получено x={x}, y={y}.")
    if x + w > image.width or y + h > image.height:
        raise ImageError(
            f"bbox выходит за границы кадра: bbox=({x},{y},{w},{h}), "
            f"кадр {image.width}x{image.height}."
        )
    return x, y, w, h


MIN_BBOX_SIDE = 2          # px: рамка уже этого после обрезки по кадру — берётся весь кадр
GRAY = (128, 128, 128)


def fit_bbox(width: int, height: int, x: int, y: int, w: int, h: int,
             min_side: int = MIN_BBOX_SIDE) -> Tuple[Tuple[int, int, int, int], str]:
    """Рамка в границах кадра — для офлайн-сборки и extract(), где одна строка CSV не должна остановить прогон.

    -> ((x, y, w, h), статус): "ok" — рамка внутри кадра, как есть; "clipped" — обрезана по границам кадра;
    "full" — после обрезки пустая или уже min_side px, берётся весь кадр. API (/v1/*) по-прежнему отвечает
    на такую рамку ошибкой (validate_bbox): там её видит и исправляет клиент.
    """
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return (0, 0, width, height), "full"
    box = (x0, y0, x1 - x0, y1 - y0)
    return box, ("ok" if box == (x, y, w, h) else "clipped")


def gray_frame(size: int) -> Image.Image:
    """Замена нечитаемого кадра: серый size×size. Эмбеддинг по нему держит строку в embeddings.npy (Q27)."""
    return Image.new("RGB", (size, size), GRAY)


def crop_with_pad(image: Image.Image, x: int, y: int, w: int, h: int, pad: float) -> Image.Image:
    """Вырезать bbox с запасом pad от его размера с каждой стороны, не выходя за кадр.

    Тем же способом нарезаны кропы в /opt/case17/data/crops, поэтому
    картинка из API и картинка из галереи попадают в модель одинаково.
    """
    px, py = int(round(w * pad)), int(round(h * pad))
    left = max(0, x - px)
    top = max(0, y - py)
    right = min(image.width, x + w + px)
    bottom = min(image.height, y + h + py)
    return image.crop((left, top, right, bottom))


DRAFT_SCALES = (8, 4, 2)                # во сколько раз JPEG умеет уменьшать при декодировании


def jpeg_draft_max(value=None) -> int:
    """EXTRACT_JPEG_DRAFT -> наибольшее уменьшение JPEG при декодировании.

    "2"/"half" (по умолчанию) — не больше 1/2; "0"/"off" — 1, полный декод; иное «включено» ("1") — до 1/8.
    """
    raw = (os.getenv("EXTRACT_JPEG_DRAFT", "2") if value is None else str(value)).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return 1
    return 2 if raw in ("2", "half") else 8


def crop_frame(image: Image.Image, x: int, y: int, w: int, h: int, pad: float, input_size: int,
               draft_max: int = None) -> Image.Image:
    """Кроп bbox с запасом pad — как crop_with_pad, но JPEG декодируется сразу с уменьшением.

    image — открытый и ещё не декодированный кадр (open_image / Image.open), рамка — в его координатах и
    уже в кадре (fit_bbox или validate_bbox). Уменьшение — наибольшее из DRAFT_SCALES, не больше draft_max
    (по умолчанию — EXTRACT_JPEG_DRAFT), при котором кроп с запасом остаётся не меньше input_size по обеим
    сторонам; запас и границы считаются в координатах исходного кадра. Без уменьшения (не JPEG, мелкий кроп,
    draft_max=1) результат пиксель в пиксель равен crop_with_pad полного кадра.
    """
    draft_max = jpeg_draft_max() if draft_max is None else draft_max
    W, H = image.size
    px, py = int(round(w * pad)), int(round(h * pad))
    box = (max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py))
    side = min(box[2] - box[0], box[3] - box[1])
    r = next((s for s in DRAFT_SCALES if s <= draft_max and side / s >= input_size), 1)
    if r > 1 and image.format == "JPEG":
        # floor, не ceil: PIL берёт scale = min(W // reqW, H // reqH) — с ceil любая
        # нечётная сторона кадра давала бы scale=1 и draft молча не срабатывал
        image.draft("RGB", (max(1, W // r), max(1, H // r)))
    img = image.convert("RGB")
    if img.size == (W, H):
        return img.crop(box)
    sx, sy = img.width / W, img.height / H
    return img.crop((int(round(box[0] * sx)), int(round(box[1] * sy)),
                     int(round(box[2] * sx)), int(round(box[3] * sy))))
