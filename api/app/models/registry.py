"""Выбор модели по имени из конфига."""
import os

from .base import EmbeddingModel
from .timm_backbone import TimmEmbeddingModel


def build_model(name: str, input_size: int = 224) -> EmbeddingModel:
    """Поддерживает три формы:

      resnet18                -> бэкбон timm как есть (заглушка)
      timm:resnet18           -> то же самое, явно
      reid:<timm-имя>         -> обученная ReID-модель, веса из REID_WEIGHTS

    Третья форма и есть путь для модели, дообученной на GPU: ставится
    переменными окружения, без единой правки кода сервиса.
    """
    key = name.strip()

    if key.startswith("reid:"):
        from .reid_checkpoint import ReidCheckpointModel

        weights = os.getenv("REID_WEIGHTS", "/models/reid_current.pt")
        return ReidCheckpointModel(key.split(":", 1)[1], weights, input_size)

    if key.startswith("timm:"):
        key = key.split(":", 1)[1]
    return TimmEmbeddingModel(key, input_size=input_size)
