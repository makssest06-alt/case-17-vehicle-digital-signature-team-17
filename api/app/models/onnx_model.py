"""Инференс через ONNX Runtime с автовыбором провайдера.

Порядок предпочтения: TensorRT -> CUDA -> CPU. Что доступно, то и берётся,
поэтому один образ работает и на карте жюри, и на нашем CPU-сервере
без правки конфигурации.

Плюсы против PyTorch на CPU:
  * графовые оптимизации (свёртка слоёв, устранение лишних узлов);
  * фиксированные формы входа — память раскладывается заранее;
  * меньше накладных расходов Python на вызов.
"""
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from .base import EmbeddingModel

PREFERRED = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")


def pick_providers() -> List[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    chosen = [p for p in PREFERRED if p in available]
    return chosen or ["CPUExecutionProvider"]


class OnnxEmbeddingModel(EmbeddingModel):
    """EMBED_MODEL=onnx:<путь к .onnx>."""

    def __init__(self, path: str, input_size: int = 224) -> None:
        self.path = path
        self.input_size = input_size
        self.name = f"onnx:{Path(path).name}"
        self.dim = 0
        self._sess = None
        self._in = None
        self._out = None
        self._mean = None
        self._std = None
        self.providers: List[str] = []

    def load(self) -> None:
        import json

        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = int(os.getenv("OMP_NUM_THREADS", "2"))

        self.providers = pick_providers()
        self._sess = ort.InferenceSession(self.path, sess_options=so, providers=self.providers)
        self._in = self._sess.get_inputs()[0].name
        self._out = self._sess.get_outputs()[0].name

        shape = self._sess.get_outputs()[0].shape
        self.dim = int(shape[-1]) if isinstance(shape[-1], int) else 0

        # нормализация рядом с моделью — чтобы в рантайме не требовался timm
        meta = Path(self.path).with_suffix(".json")
        if meta.exists():
            m = json.loads(meta.read_text())
            self._mean = np.array(m["mean"], dtype="float32")
            self._std = np.array(m["std"], dtype="float32")
            self.input_size = int(m.get("input_size", self.input_size))
            self.dim = int(m.get("dim", self.dim))
        else:
            self._mean = np.array([0.485, 0.456, 0.406], dtype="float32")
            self._std = np.array([0.229, 0.224, 0.225], dtype="float32")

        if not self.dim:
            probe = np.zeros((1, 3, self.input_size, self.input_size), dtype="float32")
            self.dim = int(self._sess.run([self._out], {self._in: probe})[0].shape[-1])

    def _run_tta(self, batch: np.ndarray) -> np.ndarray:
        """Прогон с усреднением оригинала и горизонтального отражения."""
        from ..config import settings

        feats = self._sess.run([self._out], {self._in: batch})[0].astype("float32")
        if settings.tta_flip:
            flipped = np.ascontiguousarray(batch[:, :, :, ::-1])
            feats = feats + self._sess.run([self._out], {self._in: flipped})[0].astype("float32")
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms

    def _run(self, batch: np.ndarray) -> np.ndarray:
        feats = self._sess.run([self._out], {self._in: batch})[0].astype("float32")
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms

    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        from ..fastimg import to_batch

        if self._sess is None:
            raise RuntimeError("сессия не создана, сначала load()")
        if not images:
            return np.zeros((0, self.dim), dtype="float32")
        arrs = [
            np.asarray(im.convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR))
            for im in images
        ]
        return self._run_tta(to_batch(arrs, self._mean, self._std))

    def embed_arrays(self, arrays: List[np.ndarray]) -> np.ndarray:
        """Быстрый путь: уже вырезанные и отресайзенные HxWx3 uint8."""
        from ..fastimg import to_batch

        if not arrays:
            return np.zeros((0, self.dim), dtype="float32")
        return self._run_tta(to_batch(arrays, self._mean, self._std))
