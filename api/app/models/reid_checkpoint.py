"""Обученная ReID-модель: бэкбон timm + голова с BNNeck, загруженные из чекпоинта.

Появляется после дообучения на GPU. От заглушки отличается тем, что
веса лежат в файле, а не скачиваются из хаба, и что эмбеддинг —
среднее оригинала и горизонтального отражения.
"""
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import EmbeddingModel

log = logging.getLogger("case17.model")

# Ключи реестра analysis/backbones.py -> имена timm. Чекпоинт хранит свой
# бэкбон в args["backbone"], и это надёжнее переменной окружения: если
# EMBED_MODEL расходится с чекпоинтом, модель построится не под те веса.
#
# Тег предобученных весов (`.webli`, `.v2_webli`) в списке стоит первым, но
# разрешается он не во всех версиях timm: `.v2_webli` (SigLIP 2) требует
# timm >= 1.0.15, а сервис запинен на 1.0.12. Поэтому create_model вызывается
# по очереди имён и берётся первое сработавшее. Для чекпоинтов полного
# дообучения это безопасно: pretrained=False, весь энкодер приходит из
# чекпоинта, а тег нужен только чтобы скачать чужие веса. Проверено на
# timm 1.0.12 и 1.0.29: эмбеддинги студента совпадают до 7.4e-07
# (analysis/student_on_timm.py, probe_student_timm112.json).
#
# Осторожно: для head-only чекпоинтов (pretrained=True) голое имя
# `vit_base_patch16_siglip_224` тянет SigLIP 1 (timm/ViT-B-16-SigLIP), а не
# SigLIP 2. Сегодня это не проблема — единственный head-only чекпоинт
# (weights/reid_current.pt) обучен на siglip_base.
BACKBONE_TIMM = {
    "siglip_base": ["vit_base_patch16_siglip_224.webli",
                    "vit_base_patch16_siglip_224"],
    "siglip2_base": ["vit_base_patch16_siglip_224.v2_webli",
                     "vit_base_patch16_siglip_224"],
    "clip_vitb16": ["vit_base_patch16_clip_224.openai",
                    "vit_base_patch16_clip_224"],
    "dinov2_vits14": ["vit_small_patch14_dinov2.lvd142m",
                      "vit_small_patch14_dinov2"],
}


# Подпись семейства весов для лога: «бэкбон: vit_base_patch16_siglip_224 (веса SigLIP2 из чекпоинта)».
FAMILY = {"siglip_base": "SigLIP", "siglip2_base": "SigLIP2", "clip_vitb16": "CLIP", "dinov2_vits14": "DINOv2"}


def timm_names(ck_backbone: str, fallback: str) -> List[str]:
    """Имена timm для бэкбона из args["backbone"] чекпоинта; не объявлен — имя из EMBED_MODEL."""
    names = BACKBONE_TIMM.get(ck_backbone, [fallback])
    return [names] if isinstance(names, str) else list(names)


def checkpoint_model_name(backbone: str, weights: str) -> str:
    """reid:<имя timm бэкбона, на котором обучен чекпоинт> — без построения сети (для подписи в отчётах)."""
    try:
        ck = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):         # старый формат torch.save — без mmap
        ck = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or ck.get("args") or {}
    return f"reid:{timm_names(str(cfg.get('backbone') or ''), backbone)[0]}"


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
        names = timm_names(ck_backbone, self.backbone_name)
        if ck_backbone and names[0] != self.backbone_name:
            log.debug("EMBED_MODEL задан как %s, чекпоинт обучен на %s -> беру %s",
                      self.backbone_name, ck_backbone, names[0])
        # Имя модели — по чекпоинту: оно попадает в /health, ответы API, postprocess.json и JSON замера.
        self.name = f"reid:{names[0]}"
        backbone, used_name = self._create_backbone(timm, names, head_only)
        if len(names) > 1 and used_name != names[0]:
            log.debug("тег %s на timm %s не разрешается — архитектура %s; веса целиком из чекпоинта, "
                      "pretrained не используется", names[0], timm.__version__, used_name)
        log.info("бэкбон: %s (%s)", used_name,
                 f"предобученный {names[0]}, голова из чекпоинта" if head_only
                 else f"веса {FAMILY.get(ck_backbone, ck_backbone or used_name)} из чекпоинта")
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
            log.warning("%s не объявил args[tf_mode] и REID_TF_MODE не задан — беру 'timm' (прежнее "
                        "поведение). Если модель обучена на squash-кропе, поставьте REID_TF_MODE=squash, "
                        "иначе метрики просядут.", Path(self.weights_path).name)
        if self.tf_mode not in ("timm", "squash"):
            raise ValueError(f"REID_TF_MODE/args[tf_mode] = {self.tf_mode!r}, "
                             f"ожидалось 'timm' или 'squash'")

        # Веса только для инференса: eval() (BatchNorm, Dropout), forward — под inference_mode (градиенты не
        # считаются). requires_grad_(False) не ставим: на CPU у nn.Linear с замороженным весом другое ядро,
        # векторы сдвигаются до 1.7e-7, и embeddings.npy / candidates.csv перестают совпадать побайтно.
        backbone.eval()
        head.eval()
        # A CUDA-enabled torch build is required to use an available GPU.
        # REID_DEVICE: auto (по умолчанию: cuda, если есть, иначе cpu) | cpu | cuda | mps — для замеров и проверок.
        want = os.getenv("REID_DEVICE", "auto").strip().lower() or "auto"
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if want == "auto" else want)
        if os.getenv("REID_REQUIRE_CUDA", "0") == "1" and self.device.type != "cuda":
            raise RuntimeError("REID_REQUIRE_CUDA=1 but CUDA is unavailable")
        # FP16 (autocast) — на CUDA по умолчанию, REID_FP16=0 выключает; на MPS — только явно (REID_FP16=1).
        fp16_env = os.getenv("REID_FP16", "1" if self.device.type == "cuda" else "0")
        self._fp16 = self.device.type in ("cuda", "mps") and fp16_env == "1"
        self._tf = self._build_tf(timm, backbone)
        backbone.to(self.device)
        head.to(self.device)
        self._model, self._head = backbone, head
        self.dim = out_dim
        log.info("модель %s: веса %s, устройство %s, FP16 %s, кроп %s %d, размерность %d",
                 self.name, Path(self.weights_path).name, self.device, "да" if self._fp16 else "нет",
                 self.tf_mode, self.input_size, out_dim)
        # Release source tensors after copying; no second checkpoint retained.
        del ck, state, enc, hd


    def _create_backbone(self, timm, names, head_only):
        """Строит бэкбон, перебирая имена до первого сработавшего.

        Фолбэк с тега на голое имя архитектуры разрешён ТОЛЬКО когда
        pretrained=False: в этом случае тег ничего не решает — веса целиком
        приходят из чекпоинта. Для head-only (pretrained=True) голое имя
        `vit_base_patch16_siglip_224` тянет SigLIP 1 (timm/ViT-B-16-SigLIP),
        поэтому там фолбэк запрещён: лучше честно упасть, чем молча
        подставить чужие предобученные веса.
        """
        candidates = names if not head_only else names[:1]
        last = None
        for name in candidates:
            try:
                m = timm.create_model(name, pretrained=head_only, num_classes=0,
                                      img_size=self.input_size,
                                      dynamic_img_size=True)
                return m, name
            except Exception as e:  # тег не разрешился в этой версии timm
                last = e
        raise RuntimeError(
            f"не удалось построить бэкбон ни по одному из имён {candidates} "
            f"(timm {timm.__version__}): {last}") from last

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
        x = self._tf(image.convert("RGB")).unsqueeze(0).to(self.device)
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
                torch.from_numpy(np.asarray(target, dtype="float32")).unsqueeze(0)
                .to(self.device), p=2, dim=1)
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
        x = x.to(self.device, non_blocking=True)
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
        if self._fp16:
            with torch.autocast(self.device.type, dtype=torch.float16):
                z = self._model(x)
                if self._normalize_before_head:
                    z = F.normalize(z, p=2, dim=1)
                return self._head(z)[1].float()
        z = self._model(x)
        if self._normalize_before_head:
            z = F.normalize(z, p=2, dim=1)
        return self._head(z)[1]
