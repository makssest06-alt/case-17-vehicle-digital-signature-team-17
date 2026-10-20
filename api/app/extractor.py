"""extract(): признак одного ТС полным циклом — так, как его меряют организаторы (ответы Q31, Q32).

    from app.extractor import Extractor
    ex = Extractor(); ex.load()
    v = ex.extract("images/<id>.jpg", (x, y, w, h))           # (D,) float32, L2
    V = ex.extract_batch([(path, bbox), ...])                  # (N, D)

Цикл: чтение файла с диска -> декодирование и кроп по bbox с запасом BBOX_PAD (app.imaging.crop_frame —
тот же код, что у сервиса и scripts/make_submission.py: векторы совпадают) -> препроцессинг (squash,
INPUT_SIZE) -> forward -> L2. Поиск по галерее сюда не входит (Q31).

Ускорения (разрешены ответом Q36), переменные окружения:
  REID_FP16=1           forward в FP16 (autocast) — на CUDA по умолчанию включён (app/models/reid_checkpoint.py);
  EXTRACT_JPEG_DRAFT=1  JPEG декодируется сразу с уменьшением (PIL draft: 1/2, 1/4, 1/8) — до наименьшего
                        масштаба, при котором кроп не меньше INPUT_SIZE по обеим сторонам; флаг общий для
                        extract(), сервиса и make_submission (app.imaging.jpeg_draft_max);
  EXTRACT_JPEG_DRAFT=2  то же, но уменьшение не больше 1/2 — медленнее "1", но почти не трогает векторы;
                        по умолчанию включён этот режим (bench на A5000 + Threadripper 3955WX: 141 FPS
                        против 86 без draft, качество −0.003/−0.005 mAP@10 — extract_flags_soup150s2);
  EXTRACT_JPEG_DRAFT=0  полный декод;
  EXTRACT_THREADS=N     декодирование и кроп батча в N потоков (по умолчанию 32 — у жюри 2× Xeon Gold 6338
                        на 64 ядра, Q30; на результат не влияет, только скорость).

Как офлайн-сборка (scripts/make_submission.py), кадр не роняет extract(): рамка за кадром обрезается по
границам, пустая или уже 2 px — весь кадр, нечитаемый кадр — серый INPUT_SIZE×INPUT_SIZE (imaging.fit_bbox).
Каждый такой кадр — одна строка WARNING / ERROR (один раз на кадр, замер гоняет кадры по кругу); счёт —
в describe()["frame_fixes"].
"""
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .config import settings
from .imaging import crop_frame, crop_with_pad, fit_bbox, gray_frame, jpeg_draft_max, open_image
from .models.registry import build_model

log = logging.getLogger("case17.extractor")
Source = Union[str, Path, bytes]


def _flag(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower()


class Extractor:
    def __init__(self, model=None, draft=None, threads: int = None):
        self.model = model or build_model(settings.embed_model, settings.input_size)
        # draft: "0" — выкл; "1" — максимальное уменьшение (до 1/8); "2"/"half" — не больше 1/2 (прод-дефолт)
        self.draft_max = jpeg_draft_max(draft)
        self.draft = self.draft_max > 1
        self.threads = max(1, int(_flag("EXTRACT_THREADS", "32")) if threads is None else threads)
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
        """Кадр -> кроп: open_image -> fit_bbox -> crop_frame, как scripts/make_submission.py (Frames.load)."""
        name = "<bytes>" if isinstance(src, (bytes, bytearray)) else Path(src).stem
        bbox = tuple(int(v) for v in bbox)
        try:
            raw = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
            im = open_image(raw, settings.max_upload_bytes)
            W, H = im.size
            box, status = fit_bbox(W, H, *bbox)
            crop = crop_frame(im, *box, settings.bbox_pad, settings.input_size, self.draft_max)
        except Exception as exc:                          # битый файл, не тот формат, файла нет
            img = gray_frame(settings.input_size)
            self._note("unread", name, "кадр не прочитан (%s) — эмбеддинг по серому кадру %d×%d", exc,
                       img.width, img.height)
            return crop_with_pad(img, 0, 0, img.width, img.height, settings.bbox_pad)
        self._note_fit(status, name, bbox, box, W, H)
        return crop

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
