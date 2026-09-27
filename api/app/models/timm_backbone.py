"""Заглушка на время прототипа: лёгкий бэкбон из timm, обученный на ImageNet.

Это НЕ ReID-модель, она не обучалась различать конкретные машины.
Задача — поднять весь контур целиком; веса меняются через EMBED_MODEL.
"""
import os
from typing import List

import numpy as np
import timm
import torch
from PIL import Image

from .base import EmbeddingModel


class TimmEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str, input_size: int = 224) -> None:
        self.model_name = model_name
        self.input_size = input_size
        self.name = f"timm:{model_name}"
        self.dim = 0
        self._model = None
        self._tf = None

    def load(self) -> None:
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        model.eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        cfg["input_size"] = (3, self.input_size, self.input_size)
        self._tf = timm.data.create_transform(**cfg, is_training=False)
        self._model = model
        self.dim = int(model.num_features)

    @torch.inference_mode()
    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("модель не загружена, сначала load()")
        if not images:
            return np.zeros((0, self.dim), dtype="float32")
        batch = torch.stack([self._tf(im.convert("RGB")) for im in images])
        feats = self._model(batch)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().astype("float32")
