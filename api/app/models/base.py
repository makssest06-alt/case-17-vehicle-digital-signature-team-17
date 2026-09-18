"""Интерфейс модели эмбеддингов.

Любая модель обязана уметь ровно две вещи: загрузиться и посчитать батч.
Всё остальное (какой бэкбон, откуда веса) прячется за реализацией,
поэтому подменить заглушку на обученную модель = поменять одну строку в конфиге.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL import Image


class EmbeddingModel(ABC):
    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def load(self) -> None:
        """Загрузить веса в память. Вызывается один раз при старте сервиса."""

    @abstractmethod
    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Вернуть матрицу (len(images), dim), float32, L2-нормированную."""
