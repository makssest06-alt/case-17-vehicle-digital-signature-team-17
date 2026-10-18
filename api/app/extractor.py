"""extract(): признак одного ТС полным циклом — так, как его меряют организаторы (ответы Q31, Q32).

    from app.extractor import Extractor
    ex = Extractor(); ex.load()
    v = ex.extract("images/<id>.jpg", (x, y, w, h))           # (D,) float32, L2
    V = ex.extract_batch([(path, bbox), ...])                  # (N, D)

Цикл: чтение файла с диска -> декодирование -> кроп по bbox с запасом BBOX_PAD (как app.imaging) ->
препроцессинг (squash 224) -> forward -> L2. Поиск по галерее сюда не входит (Q31).

Ускорения (разрешены ответом Q36), переменные окружения:
  REID_FP16=1           forward в FP16 (autocast) — на CUDA по умолчанию включён (app/models/reid_checkpoint.py);
  EXTRACT_JPEG_DRAFT=1  JPEG декодируется сразу с уменьшением (PIL draft: 1/2, 1/4, 1/8) — до наименьшего
                        масштаба, при котором кроп не меньше INPUT_SIZE по обеим сторонам; по умолчанию выкл.;
  EXTRACT_THREADS=N     декодирование и кроп батча в N потоков (по умолчанию 4); на результат не влияет.
Без EXTRACT_JPEG_DRAFT кроп совпадает с сервисом и офлайн-скриптом пиксель в пиксель.
"""
import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .config import settings
from .imaging import crop_with_pad, validate_bbox
from .models.registry import build_model

Source = Union[str, Path, bytes]
DRAFT_SCALES = (8, 4, 2)                       # во сколько раз JPEG умеет уменьшать при декодировании


def _flag(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower()


class Extractor:
    def __init__(self, model=None, draft: bool = None, threads: int = None):
        self.model = model or build_model(settings.embed_model, settings.input_size)
        self.draft = _flag("EXTRACT_JPEG_DRAFT", "0") in ("1", "true", "yes", "on") if draft is None else draft
        self.threads = max(1, int(_flag("EXTRACT_THREADS", "4")) if threads is None else threads)
        self.load_s = None

    def load(self) -> float:
        """Загрузить веса; -> секунды (время загрузки весов, Q31 — справочно)."""
        t0 = time.perf_counter()
        self.model.load()
        self.load_s = time.perf_counter() - t0
        return self.load_s

    # ------------------------------------------------------------------ кадр -> кроп
    def crop(self, src: Source, bbox: Sequence[int]) -> Image.Image:
        raw = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
        im = Image.open(io.BytesIO(raw))
        W, H = im.size
        x, y, w, h = (int(v) for v in bbox)
        if not self.draft or im.format != "JPEG":
            img = im.convert("RGB")
            validate_bbox(img, x, y, w, h)
            return crop_with_pad(img, x, y, w, h, settings.bbox_pad)
        validate_bbox(im, x, y, w, h)
        px, py = int(round(w * settings.bbox_pad)), int(round(h * settings.bbox_pad))
        box = (max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py))
        side = min(box[2] - box[0], box[3] - box[1])
        r = next((s for s in DRAFT_SCALES if side / s >= settings.input_size), 1)
        if r > 1:
            im.draft("RGB", (math.ceil(W / r), math.ceil(H / r)))
        img = im.convert("RGB")
        sx, sy = img.width / W, img.height / H
        return img.crop((int(round(box[0] * sx)), int(round(box[1] * sy)),
                         int(round(box[2] * sx)), int(round(box[3] * sy))))

    def crops(self, items: Iterable[Tuple[Source, Sequence[int]]]) -> List[Image.Image]:
        items = list(items)
        if self.threads == 1 or len(items) == 1:
            return [self.crop(s, b) for s, b in items]
        with ThreadPoolExecutor(max_workers=min(self.threads, len(items))) as pool:
            return list(pool.map(lambda it: self.crop(*it), items))

    # ------------------------------------------------------------------ кроп -> вектор
    def extract(self, src: Source, bbox: Sequence[int]) -> np.ndarray:
        """Полный цикл на одно ТС. -> (D,) float32, L2-нормирован."""
        return self.model.embed_batch([self.crop(src, bbox)])[0]

    def extract_batch(self, items: Iterable[Tuple[Source, Sequence[int]]]) -> np.ndarray:
        """Батч: декодирование и кроп в потоках, forward одним батчем. -> (N, D) float32."""
        return self.model.embed_batch(self.crops(items))

    def describe(self) -> dict:
        m = self.model
        return {"model": getattr(m, "name", settings.embed_model), "dim": getattr(m, "dim", None),
                "device": str(getattr(m, "device", "cpu")), "fp16": bool(getattr(m, "_fp16", False)),
                "jpeg_draft": self.draft, "threads": self.threads, "tf_mode": getattr(m, "tf_mode", None),
                "load_s": self.load_s}
