"""Реестр бэкбонов. Добавить новый = добавить строку сюда.

Каждый бэкбон умеет одно: превратить картинку в вектор фиксированной длины.
Всё остальное (голова, обучение, оценка) от бэкбона не зависит,
поэтому переключение делается параметром --backbone, а не правкой кода.
"""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    kind: str          # timm | fastreid
    dim: int
    input_size: int
    license_note: str
    model_name: str = ""
    config: str = ""   # только для fastreid
    weights: str = ""  # только для fastreid


REGISTRY = {
    "clip_vitb16": BackboneSpec(
        key="clip_vitb16", kind="timm", dim=768, input_size=224,
        model_name="vit_base_patch16_clip_224.openai",
        license_note="MIT (OpenAI CLIP); данные WIT не публикуются",
    ),
    "dinov2_vits14": BackboneSpec(
        key="dinov2_vits14", kind="timm", dim=384, input_size=224,
        model_name="vit_small_patch14_dinov2.lvd142m",
        license_note="Apache 2.0 (Meta DINOv2); данные LVD-142M внутренние",
    ),
    "siglip2_base": BackboneSpec(
        key="siglip2_base", kind="timm", dim=768, input_size=224,
        model_name="vit_base_patch16_siglip_224.v2_webli",
        license_note="Apache 2.0 (Google SigLIP 2); данные WebLI внутренние",
    ),
    "resnet50_in1k": BackboneSpec(
        key="resnet50_in1k", kind="timm", dim=2048, input_size=224,
        model_name="resnet50.a1_in1k",
        license_note="веса Apache 2.0 (timm); ImageNet — non-commercial research",
    ),
    # ВНИМАНИЕ: обучен на VERI-Wild, датасет non-commercial.
    # Только как исследовательский ориентир, в сдаваемое решение не идёт.
    # Подробности в LICENSES.md.
    "veriwild_r50ibn": BackboneSpec(
        key="veriwild_r50ibn", kind="fastreid", dim=2048, input_size=256,
        config="configs/VERIWild/bagtricks_R50-ibn.yml",
        weights="/opt/case17/models/zoo/veriwild_bot_R50-ibn.pth",
        license_note="ТОЛЬКО ИССЛЕДОВАНИЕ: датасет VERI-Wild non-commercial",
    ),
    "veri_r50ibn_sbs": BackboneSpec(
        key="veri_r50ibn_sbs", kind="fastreid", dim=2048, input_size=256,
        config="configs/VeRi/sbs_R50-ibn.yml",
        weights="/opt/case17/models/zoo/veri_sbs_R50-ibn.pth",
        license_note="ТОЛЬКО ИССЛЕДОВАНИЕ: датасет VeRi-776 non-commercial",
    ),
}

RESEARCH_ONLY = {"veriwild_r50ibn", "veri_r50ibn_sbs"}


def get(key: str) -> BackboneSpec:
    if key not in REGISTRY:
        raise SystemExit(f"неизвестный бэкбон {key}; доступны: {', '.join(REGISTRY)}")
    return REGISTRY[key]


def build_encoder(spec: BackboneSpec) -> Callable:
    """Возвращает (forward_fn, transform_fn) — детали спрятаны внутри."""
    import torch

    if spec.kind == "timm":
        import timm

        model = timm.create_model(spec.model_name, pretrained=True, num_classes=0)
        model.eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        cfg["input_size"] = (3, spec.input_size, spec.input_size)
        tf = timm.data.create_transform(**cfg, is_training=False)

        def forward(batch_tensor):
            return model(batch_tensor)

        return forward, tf, model

    if spec.kind == "fastreid":
        import sys
        from pathlib import Path

        import numpy as np
        from PIL import Image
        from torchvision.transforms import functional as TF

        sys.path.insert(0, "/opt/case17/third_party/fast-reid")
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model
        from fastreid.utils.checkpoint import Checkpointer

        cfg = get_cfg()
        cfg.merge_from_file(f"/opt/case17/third_party/fast-reid/{spec.config}")
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()
        model = build_model(cfg)
        model.eval()
        Checkpointer(model).load(spec.weights)
        h, w = cfg.INPUT.SIZE_TEST

        def tf(img: Image.Image):
            # fast-reid ждёт RGB 0..255, нормализацию делает сам внутри
            img = img.convert("RGB").resize((w, h), Image.BILINEAR)
            return TF.pil_to_tensor(img).float()

        def forward(batch_tensor):
            out = model({"images": batch_tensor})
            if isinstance(out, dict):
                out = out.get("features", next(iter(out.values())))
            return out

        return forward, tf, model

    raise SystemExit(f"неизвестный тип бэкбона {spec.kind}")
