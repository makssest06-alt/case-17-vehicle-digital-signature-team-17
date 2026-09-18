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

        backbone = timm.create_model(self.backbone_name, pretrained=False, num_classes=0)
        head = _ReidHead(backbone.num_features, hidden, out_dim, dropout)

        state = ck.get("model", ck)
        enc = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        hd = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
        if not enc:  # чекпоинт только головы (train_head.py)
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

    @torch.inference_mode()
    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("модель не загружена, сначала load()")
        if not images:
            return np.zeros((0, self.dim), dtype="float32")
        x = torch.stack([self._tf(im.convert("RGB")) for im in images])
        a = self._head(self._model(x))[1]
        b = self._head(self._model(torch.flip(x, dims=[3])))[1]
        feats = F.normalize((a + b) / 2, p=2, dim=1)
        return feats.cpu().numpy().astype("float32")
