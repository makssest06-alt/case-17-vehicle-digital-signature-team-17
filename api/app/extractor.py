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
  EXTRACT_JPEG_DRAFT=2  то же, но уменьшение не больше 1/2 — медленнее "1", но почти не трогает векторы;
  EXTRACT_THREADS=N     декодирование и кроп батча в N потоков (по умолчанию 4); на результат не влияет.
Без EXTRACT_JPEG_DRAFT кроп совпадает с сервисом и офлайн-скриптом пиксель в пиксель.

Как офлайн-сборка (scripts/make_submission.py), кадр не роняет extract(): рамка за кадром обрезается по
границам, пустая или уже 2 px — весь кадр, нечитаемый кадр — серый INPUT_SIZE×INPUT_SIZE (imaging.fit_bbox).
Каждый такой кадр — одна строка WARNING / ERROR (один раз на кадр, замер гоняет кадры по кругу); счёт —
в describe()["frame_fixes"].
"""
import io
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .config import settings
from .imaging import crop_with_pad, fit_bbox, gray_frame
from .models.registry import build_model

log = logging.getLogger("case17.extractor")
Source = Union[str, Path, bytes]
DRAFT_SCALES = (8, 4, 2)                       # во сколько раз JPEG умеет уменьшать при декодировании


def _flag(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower()


class Extractor:
    def __init__(self, model=None, draft=None, threads: int = None):
        self.model = model or build_model(settings.embed_model, settings.input_size)
        # draft: "0" — выкл; "1" — максимальное уменьшение (до 1/8); "2"/"half" — не больше 1/2
        raw = (_flag("EXTRACT_JPEG_DRAFT", "0") if draft is None else str(draft)).strip().lower()
        self.draft = raw not in ("0", "false", "no", "off")
        self.draft_max = 2 if raw in ("2", "half") else 8
        self.threads = max(1, int(_flag("EXTRACT_THREADS", "4")) if threads is None else threads)
        self.load_s = None
        self.fixes = {"clipped": 0, "full": 0, "unread": 0}          # кадров, не повторов
        self._seen, self._lock = set(), threading.Lock()

    def load(self) -> float:
        """Загрузить веса; -> секунды (время загрузки весов, Q31 — справочно)."""
        t0 = time.perf_counter()
        self.model.load()
        self.load_s = time.perf_counter() - t0
        d = self.describe()
        log.info("extract(): модель %s загружена за %.1f с — устройство %s, FP16 %s, JPEG draft %s, потоков %d",
                 d["model"], self.load_s, d["device"], "да" if d["fp16"] else "нет",
                 "да" if d["jpeg_draft"] else "нет", d["threads"])
        return self.load_s

    # ------------------------------------------------------------------ кадр -> кроп
    def _note(self, kind: str, name: str, msg: str, *args) -> None:
        """Счёт и одна строка лога на кадр: clipped / full — WARNING, unread — ERROR."""
        with self._lock:
            if (kind, name) in self._seen:
                return
            self._seen.add((kind, name))
            self.fixes[kind] += 1
        log.log(logging.ERROR if kind == "unread" else logging.WARNING, "extract(): %s: " + msg, name, *args)

    def _note_fit(self, status: str, name: str, bbox, box, W: int, H: int) -> None:
        """Отметить обрезку рамки — после того, как кадр декодирован (битый кадр — только «не прочитан»)."""
        if status == "clipped":
            self._note("clipped", name, "рамка %s выходит за кадр %d×%d — обрезана до %s", bbox, W, H, box)
        elif status == "full":
            self._note("full", name, "рамка %s после обрезки по кадру %d×%d пустая или уже 2 px — беру весь кадр",
                       bbox, W, H)

    def crop(self, src: Source, bbox: Sequence[int]) -> Image.Image:
        name = "<bytes>" if isinstance(src, (bytes, bytearray)) else Path(src).stem
        bbox = tuple(int(v) for v in bbox)
        try:
            raw = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
            im = Image.open(io.BytesIO(raw))
            if self.draft and im.format == "JPEG":
                return self._crop_draft(im, bbox, name)
            img = im.convert("RGB")
        except Exception as exc:                          # битый файл, не тот формат, файла нет
            img = gray_frame(settings.input_size)
            self._note("unread", name, "кадр не прочитан (%s) — эмбеддинг по серому кадру %d×%d", exc,
                       img.width, img.height)
            return crop_with_pad(img, 0, 0, img.width, img.height, settings.bbox_pad)
        box, status = fit_bbox(img.width, img.height, *bbox)
        self._note_fit(status, name, bbox, box, img.width, img.height)
        return crop_with_pad(img, *box, settings.bbox_pad)

    def _crop_draft(self, im: Image.Image, bbox: Tuple[int, int, int, int], name: str) -> Image.Image:
        """JPEG декодируется сразу с уменьшением: кроп считается в координатах исходного кадра."""
        W, H = im.size
        (x, y, w, h), status = fit_bbox(W, H, *bbox)
        px, py = int(round(w * settings.bbox_pad)), int(round(h * settings.bbox_pad))
        box = (max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py))
        side = min(box[2] - box[0], box[3] - box[1])
        r = next((s for s in DRAFT_SCALES if s <= self.draft_max and side / s >= settings.input_size), 1)
        if r > 1:
            im.draft("RGB", (math.ceil(W / r), math.ceil(H / r)))
        img = im.convert("RGB")
        self._note_fit(status, name, bbox, (x, y, w, h), W, H)
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
                "jpeg_draft": self.draft, "draft_max": self.draft_max, "threads": self.threads,
                "tf_mode": getattr(m, "tf_mode", None),
                "load_s": self.load_s, "frame_fixes": dict(self.fixes)}
