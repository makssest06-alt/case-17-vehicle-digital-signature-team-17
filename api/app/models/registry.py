"""Выбор модели по имени из конфига."""
import os
from pathlib import Path

from .base import EmbeddingModel
from .timm_backbone import TimmEmbeddingModel


def build_model(name: str, input_size: int = 224) -> EmbeddingModel:
    """Поддерживает формы:

      resnet18                -> бэкбон timm как есть (заглушка)
      timm:resnet18           -> то же самое, явно
      reid:<timm-имя>         -> обученная ReID-модель, веса из REID_WEIGHTS

    Форма reid: — путь для модели, дообученной на GPU: ставится переменными
    окружения, без единой правки кода сервиса.
    """
    key = name.strip()

    if key.startswith("reid:"):
        from .reid_checkpoint import ReidCheckpointModel

        weights = os.getenv("REID_WEIGHTS")
        if not weights:
            weights = "/opt/weights/clean_student_a10_ep28.pt"
            # Прямой запуск из checkout без Docker/.env. В образе REID_WEIGHTS —
            # /opt/weights/clean_student_a10_ep28.pt (api/Dockerfile).
            local_weights = Path(__file__).resolve().parents[3] / "weights" / Path(weights).name
            if local_weights.is_file():
                weights = str(local_weights)
        return ReidCheckpointModel(key.split(":", 1)[1], weights, input_size)

    if key.startswith("timm:"):
        key = key.split(":", 1)[1]
    return TimmEmbeddingModel(key, input_size=input_size)
