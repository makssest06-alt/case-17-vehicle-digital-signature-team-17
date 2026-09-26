"""Обученная ReID-модель: бэкбон timm + голова с BNNeck, загруженные из чекпоинта.

Появляется после дообучения на GPU. От заглушки отличается тем, что
веса лежат в файле, а не скачиваются из хаба, и что эмбеддинг —
среднее оригинала и горизонтального отражения.
"""
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import EmbeddingModel


# Ключи реестра analysis/backbones.py -> имена timm. Чекпоинт хранит свой
# бэкбон в args["backbone"], и это надёжнее переменной окружения: если
# EMBED_MODEL расходится с чекпоинтом, модель построится не под те веса.
BACKBONE_TIMM = {
    "siglip_base": "vit_base_patch16_siglip_224.webli",
    "siglip2_base": "vit_base_patch16_siglip_224.v2_webli",
    "clip_vitb16": "vit_base_patch16_clip_224.openai",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
}


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
        # analysis/train_reid.py пишет разрешение в args["size"] (0 = родное),
        # старые чекпоинты — в "input_size"; ноль трактуем как «не задано».
        self.input_size = int(cfg.get("input_size") or cfg.get("size") or self.input_size)

        state = ck.get("model", ck)
        enc = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        hd = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}

        # Чекпоинт бывает двух видов, и это важно различать:
        #  * полное дообучение — внутри есть веса бэкбона (ключи encoder.*);
        #  * только голова (train_head.py) — бэкбон был заморожен и в файл не попал,
        #    значит его надо взять предобученным из образа, иначе он останется
        #    случайно инициализированным и модель выдаст мусор.
        head_only = not enc
        # Бэкбон берём из чекпоинта, если он там объявлен: EMBED_MODEL — это
        # настройка окружения, а args["backbone"] — факт о весах. Расхождение
        # означает, что собран не тот чекпоинт, о котором думают.
        ck_backbone = str(cfg.get("backbone") or "")
        name = BACKBONE_TIMM.get(ck_backbone, self.backbone_name)
        if ck_backbone and name != self.backbone_name:
            print(f"  reid_checkpoint: EMBED_MODEL задан как {self.backbone_name}, "
                  f"но чекпоинт обучен на {ck_backbone} -> беру {name}", flush=True)
        backbone = timm.create_model(name, pretrained=head_only,
                                     num_classes=0, img_size=self.input_size,
                                     dynamic_img_size=True)
        head = _ReidHead(backbone.num_features, hidden, out_dim, dropout)

        if head_only:
            hd = ck.get("head", hd)
        else:
            backbone.load_state_dict(enc, strict=True)
        head.load_state_dict(hd, strict=True)

        # Нормировать выход бэкбона перед головой нужно ТОЛЬКО для head-only:
        # такие головы учились на признаках из research/extract_features.py,
        # а тот скрипт нормирует перед сохранением. Головы полного дообучения
        # (analysis/train_reid.py, analysis/train_distill.py) учились как
        # head(encoder(x)) — нормировка на инференсе меняет распределение входа
        # и стоит 1.2-5.2 mAP (analysis/service_path_grid.py).
        self._normalize_before_head = head_only
        # Режим кропа. Приоритет: переменная окружения > args чекпоинта > timm.
        #
        # Дефолт именно timm, потому что это прежнее поведение сервиса и оно
        # верно для production-весов: reid_current.pt (= geom_ep30) и
        # siglip_vitb16_reid_ep29.pt обучены на base-кропе и args["tf_mode"]
        # не содержат. Угадывать squash для всех full-чекпоинтов нельзя —
        # сломает их.
        #
        # Squash-модели (analysis/train_reid.py --tf-mode squash,
        # analysis/train_distill.py) пишут tf_mode в args сами. Для чекпоинтов,
        # которые его не пишут (дистиллят из team17/clip-ensemble), режим
        # задаётся переменной REID_TF_MODE — это явное решение оператора,
        # а не догадка кода.
        self.tf_mode = os.getenv("REID_TF_MODE") or str(cfg.get("tf_mode") or "timm")
        if not cfg.get("tf_mode") and not os.getenv("REID_TF_MODE"):
            print(f"  reid_checkpoint: {Path(self.weights_path).name} не объявил "
                  f"args[tf_mode] и REID_TF_MODE не задан — беру 'timm' "
                  f"(прежнее поведение). Если модель обучена на squash-кропе, "
                  f"поставьте REID_TF_MODE=squash, иначе метрики просядут.",
                  flush=True)
        if self.tf_mode not in ("timm", "squash"):
            raise ValueError(f"REID_TF_MODE/args[tf_mode] = {self.tf_mode!r}, "
                             f"ожидалось 'timm' или 'squash'")

        backbone.eval()
        head.eval()
        self._tf = self._build_tf(timm, backbone)
        self._model, self._head = backbone, head
        self.dim = out_dim

    def _build_tf(self, timm, backbone):
        cfgd = timm.data.resolve_data_config({}, model=backbone)
        mean, std = tuple(cfgd["mean"]), tuple(cfgd["std"])
        if self.tf_mode == "squash":
            from torchvision import transforms as T

            return T.Compose([
                T.Resize((self.input_size, self.input_size),
                         interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        cfgd = dict(cfgd)
        cfgd["input_size"] = (3, self.input_size, self.input_size)
        return timm.data.create_transform(**cfgd, is_training=False)

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
        """Бэкбон -> (L2 для head-only) -> голова.

        Нормировка обязательна только для head-only чекпоинтов: их голова
        училась на выходах бэкбона, которые extract_features.py нормирует
        перед сохранением. Без неё голова получает вектор другой длины, чем
        видела при обучении, и это стоит 3 пункта mAP и вдвое больше
        катастрофических промахов (61.52 против 64.54 на валидации,
        research/results/service_path_extras.json).

        Для чекпоинтов полного дообучения (ключи encoder.*) нормировка
        вредит: они учились как head(encoder(x)). Замер на наших моделях —
        от -1.18 до -5.21 mAP (analysis/service_path_grid.py).
        """
        z = self._model(x)
        if self._normalize_before_head:
            z = F.normalize(z, p=2, dim=1)
        return self._head(z)[1]
