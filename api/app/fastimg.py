"""Быстрый путь картинки: декодирование, кроп, ресайз, нормализация.

Три отличия от наивного пути через PIL + torchvision:

1. **TurboJPEG** декодирует в 2-4 раза быстрее PIL и умеет декодировать
   сразу в уменьшенном масштабе — если кроп всё равно ужимается до 224,
   незачем разворачивать полный кадр 1920x1080.
2. **Сначала кроп, потом ресайз.** Наивный путь ресайзит весь кадр,
   а потом вырезает — это работа впустую над пикселями, которые выбросим.
3. **Нормализация векторно в numpy**, одним массивом на весь батч,
   вместо torchvision-трансформа по одной картинке.

Если TurboJPEG в системе нет, всё честно работает через PIL — просто медленнее.
"""
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from turbojpeg import TJPF_RGB, TurboJPEG

    _TJ: Optional["TurboJPEG"] = TurboJPEG()
    HAVE_TURBO = True
except Exception:  # библиотеки нет или не нашлась libturbojpeg
    _TJ = None
    HAVE_TURBO = False

# TurboJPEG умеет только эти доли — берём ближайшую сверху к нужной
_SCALES: Tuple[Tuple[int, int], ...] = (
    (1, 8), (1, 4), (3, 8), (1, 2), (5, 8), (3, 4), (7, 8), (1, 1),
)


def pick_scale(src_long: int, need_long: int) -> Tuple[int, int]:
    """Наименьший масштаб, при котором картинка ещё не мельче нужного."""
    for num, den in _SCALES:
        if src_long * num / den >= need_long:
            return num, den
    return 1, 1


def decode(raw: bytes, target_long: Optional[int] = None) -> np.ndarray:
    """JPEG -> RGB uint8 HxWx3. target_long включает декодирование с уменьшением."""
    if HAVE_TURBO:
        try:
            if target_long:
                w, h, _, _ = _TJ.decode_header(raw)
                num, den = pick_scale(max(w, h), target_long)
                if (num, den) != (1, 1):
                    return _TJ.decode(raw, pixel_format=TJPF_RGB, scaling_factor=(num, den))
            return _TJ.decode(raw, pixel_format=TJPF_RGB)
        except Exception:
            pass  # битый JPEG или не JPEG — падаем на PIL
    import io

    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def crop_then_resize(arr: np.ndarray, box: Tuple[int, int, int, int], size: int) -> np.ndarray:
    """Сначала вырезать, потом ужать — а не наоборот."""
    x0, y0, x1, y1 = box
    h, w = arr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    sub = arr[y0:y1, x0:x1]
    if sub.size == 0:
        sub = arr
    return np.asarray(Image.fromarray(sub).resize((size, size), Image.BILINEAR))


def to_batch(arrays: List[np.ndarray], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Список HxWx3 uint8 -> NCHW float32, нормализованный. Одной векторной операцией."""
    batch = np.stack(arrays).astype(np.float32, copy=False)
    batch /= 255.0
    batch -= mean
    batch /= std
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
