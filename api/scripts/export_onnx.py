#!/usr/bin/env python3
"""Экспорт текущей модели в ONNX с обязательной проверкой корректности.

    docker compose exec api python scripts/export_onnx.py --out /models/reid.onnx

Рядом с .onnx кладётся .json с параметрами нормализации и размерностью,
чтобы в рантайме не требовался timm.

Если косинус между эталонным и ONNX-эмбеддингом ниже порога — экспорт
считается неудачным и скрипт возвращает ненулевой код. Молча подменить
модель на слегка другую нельзя.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/srv")

from app.config import settings                # noqa: E402
from app.models.registry import build_model    # noqa: E402


class Wrapper(torch.nn.Module):
    """Бэкбон и голова одним графом, с L2 на выходе."""

    def __init__(self, model):
        super().__init__()
        self.backbone = getattr(model, "_model")
        self.head = getattr(model, "_head", None)

    def forward(self, x):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)[1]
        return torch.nn.functional.normalize(f, p=2, dim=1)


def norm_params(model):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = getattr(model, "_tf", None)
    try:
        for t in tf.transforms:
            if hasattr(t, "mean"):
                mean, std = list(map(float, t.mean)), list(map(float, t.std))
    except Exception:
        pass
    return mean, std


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/models/reid.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tolerance", type=float, default=0.999)
    a = ap.parse_args()

    model = build_model(settings.embed_model, settings.input_size)
    model.load()
    size = model.input_size
    print(f"модель {model.name}, {model.dim}D, вход {size}")

    net = Wrapper(model).eval()
    dummy = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        ref = net(dummy).numpy()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net, dummy, str(out),
        input_names=["images"], output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=a.opset, do_constant_folding=True,
    )
    print(f"экспортировано: {out} ({out.stat().st_size / 1024**2:.1f} МБ)")

    mean, std = norm_params(model)
    out.with_suffix(".json").write_text(json.dumps(
        {"mean": mean, "std": std, "input_size": size, "dim": model.dim,
         "source_model": model.name}, ensure_ascii=False, indent=2))
    print(f"метаданные: mean={mean} std={std}")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": dummy.numpy()})[0]
    cos = float((ref * got).sum() / (np.linalg.norm(ref) * np.linalg.norm(got)))
    print(f"косинус эталон/ONNX, батч 1: {cos:.6f} (порог {a.tolerance})")

    b = torch.randn(4, 3, size, size)
    with torch.inference_mode():
        ref_b = net(b).numpy()
    got_b = sess.run(None, {"images": b.numpy()})[0]
    cos_b = float(np.mean([
        (ref_b[i] * got_b[i]).sum() / (np.linalg.norm(ref_b[i]) * np.linalg.norm(got_b[i]))
        for i in range(len(b))
    ]))
    print(f"косинус на батче 4: {cos_b:.6f}")

    ok = cos >= a.tolerance and cos_b >= a.tolerance
    print("ЭКСПОРТ КОРРЕКТЕН" if ok else "ЭКСПОРТ НЕКОРРЕКТЕН — выход разошёлся")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
