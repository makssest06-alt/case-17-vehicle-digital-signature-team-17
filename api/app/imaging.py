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
