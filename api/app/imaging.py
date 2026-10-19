"""Загрузка и валидация картинок, вырезание по bbox."""
import io
from typing import Tuple

from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_PIXELS = 80_000_000  # защита от decompression bomb


class ImageError(ValueError):
    """Понятная человеку ошибка входных данных."""


def load_image(raw: bytes, max_bytes: int) -> Image.Image:
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
    return image.convert("RGB")


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
