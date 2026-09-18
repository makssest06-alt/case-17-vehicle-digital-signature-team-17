"""Обученная ReID-модель: бэкбон timm + голова с BNNeck, загруженные из чекпоинта.

Появляется после дообучения на GPU. От заглушки отличается тем, что
веса лежат в файле, а не скачиваются из хаба, и что эмбеддинг —
среднее оригинала и горизонтального отражения.
"""
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import EmbeddingModel


class _ReidHead(nn.Module):
    """Та же голова, что обучалась: Linear -> BN -> ReLU -> Dropout -> Linear -> BNNeck."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.bnneck = nn.BatchNorm1d(out_dim)
        self.bnneck.bias.requires_grad_(False)

    def forward(self, x):
        h = self.drop(F.relu(self.bn1(self.fc1(x))))
        feat = self.fc2(h)
        return feat, self.bnneck(feat)


class ReidCheckpointModel(EmbeddingModel):
    """EMBED_MODEL=reid:<timm-имя>, веса из REID_WEIGHTS."""

    def __init__(self, backbone: str, weights: str, input_size: int = 224) -> None:
        self.backbone_name = backbone
        self.weights_path = weights
        self.input_size = input_size
        self.name = f"reid:{backbone}"
        self.dim = 0
        self._model = None
        self._head = None
        self._tf = None

    def load(self) -> None:
        import timm

        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
        ck = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg") or ck.get("args") or {}
        hidden = int(cfg.get("hidden", 1024))
        out_dim = int(cfg.get("out_dim", 512))
        dropout = float(cfg.get("dropout", 0.3))
        self.input_size = int(cfg.get("input_size", self.input_size))

        state = ck.get("model", ck)
        enc = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        hd = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}

        # Чекпоинт бывает двух видов, и это важно различать:
        #  * полное дообучение — внутри есть веса бэкбона (ключи encoder.*);
        #  * только голова (train_head.py) — бэкбон был заморожен и в файл не попал,
        #    значит его надо взять предобученным из образа, иначе он останется
        #    случайно инициализированным и модель выдаст мусор.
        head_only = not enc
        backbone = timm.create_model(self.backbone_name, pretrained=head_only, num_classes=0)
        head = _ReidHead(backbone.num_features, hidden, out_dim, dropout)

        if head_only:
            hd = ck.get("head", hd)
        else:
            backbone.load_state_dict(enc, strict=True)
        head.load_state_dict(hd, strict=True)

        backbone.eval()
        head.eval()
        cfgd = timm.data.resolve_data_config({}, model=backbone)
        cfgd["input_size"] = (3, self.input_size, self.input_size)
        self._tf = timm.data.create_transform(**cfgd, is_training=False)
        self._model, self._head = backbone, head
        self.dim = out_dim

    def patch_saliency(self, image: Image.Image, target: np.ndarray = None):
        """Вклад участков кадра в итоговый вектор — для кнопки «показать почему».

        Берём токены последнего слоя бэкбона (по одному на участок 16x16),
        прогоняем каждый через ту же голову, что и общий вектор, и считаем
        косинус с целью. Цель — либо сам итоговый вектор кадра (тогда карта
        показывает, что формирует признак), либо вектор найденного кандидата
        (тогда карта показывает, что именно совпало).

        Это не Grad-CAM: градиенты не считаются вовсе, один прямой проход.
        Зато метод опирается ровно на ту величину, по которой принимается
        решение, — на косинусное сходство, а не на её приближение.
        """
        if self._model is None:
            raise RuntimeError("модель не загружена, сначала load()")
        x = self._tf(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            tokens = self._model.forward_features(x)
            n = tokens.shape[1]
            side = int(round(n ** 0.5))
            if side * side != n:               # впереди служебные токены (CLS и т.п.)
                side = int(round((n - 1) ** 0.5))
                tokens = tokens[:, n - side * side :, :]
            pooled = self._model.forward_head(self._model.forward_features(x))
            v_pool = F.normalize(self._head(pooled)[1], p=2, dim=1)
            v_tok = F.normalize(self._head(tokens[0])[1], p=2, dim=1)
            tgt = v_pool if target is None else F.normalize(
                torch.from_numpy(np.asarray(target, dtype="float32")).unsqueeze(0), p=2, dim=1)
            sim = (v_tok @ tgt.T).squeeze(1)
        return sim.cpu().numpy().astype("float32"), side

    @torch.inference_mode()
    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("модель не загружена, сначала load()")
        if not images:
            return np.zeros((0, self.dim), dtype="float32")
        from ..config import settings

        x = torch.stack([self._tf(im.convert("RGB")) for im in images])
        a = self._encode(x)
        b = self._encode(torch.flip(x, dims=[3])) if settings.tta_flip else None
        feats = F.normalize(a if b is None else (a + b) / 2, p=2, dim=1)
        return feats.cpu().numpy().astype("float32")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Бэкбон -> L2 -> голова.

        Нормировка перед головой обязательна: голова обучалась на выходах
        бэкбона, которые extract_features.py нормирует перед сохранением.
        Без неё голова получает вектор другой длины, чем видела при обучении,
        и это стоит 3 пункта mAP и вдвое больше катастрофических промахов
        (61.52 против 64.54 на валидации, research/results/service_path_extras.json).
        """
        return self._head(F.normalize(self._model(x), p=2, dim=1))[1]
