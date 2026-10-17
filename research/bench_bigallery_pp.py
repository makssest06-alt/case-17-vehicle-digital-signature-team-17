#!/usr/bin/env python3
"""Стоимость штрафа за ту же камеру и DBA: мс на Маке и оценка для RTX 4060.

Штраф на запрос: фоновый дескриптор полного кадра (resnet18 по кадру 448×256 с
замазанной машиной, как research/pseudo_camera.py) + косинус с фоном top-K
кадров галереи (их фон считается один раз при добавлении в галерею). Основной
путь — эмбеддинг SigLIP ViT-B/16 224 (a10). Мерится b1 на CPU этого Мака;
для 4060 время оценено по FLOPs: 8.27 мс SigLIP при b1 (FINAL_WORK_REPORT,
121 FPS) × (1 + FLOPs фона / FLOPs SigLIP) — это оценка, не замер на GPU.
DBA (research/dba.py): пересчёт всей галереи и добавление кадра через DBAGallery.add.

    python research/bench_bigallery_pp.py --zip dataset.zip --data-dir <данные>
"""
import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dba as dbam  # noqa: E402
import pseudo_camera as pc  # noqa: E402

GPU_SIGLIP_MS = 8.27        # RTX 4060, a10, FP16, b1 — FINAL_WORK_REPORT.md §8
TOPK = 100


def timed(fn, n=30, warm=5) -> float:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def flops(model, x) -> int:
    with torch.enable_grad(), FlopCounterMode(display=False) as fc:      # счётчику нужен граф, время не мерится
        model(x)
    return int(fc.get_total_flops())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "honest_eval_results" / "bench_bigallery_pp.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    rows = list(csv.DictReader((a.data_dir / "train.csv").open(encoding="utf-8")))[:40]
    zf = zipfile.ZipFile(a.zip)
    raw = [zf.read(f"images/{r['image_id']}.jpg") for r in rows]
    boxes = [tuple(int(r[k]) for k in ("x", "y", "w", "h")) for r in rows]

    bg = timm.create_model("resnet18", pretrained=False, num_classes=0).eval()
    vit = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=0).eval()
    xb, xv = torch.randn(1, 3, 256, 448), torch.randn(1, 3, 224, 224)
    i = [0]

    def bg_full():
        k = i[0] % len(raw)
        i[0] += 1
        im = Image.open(io.BytesIO(raw[k]))
        w0 = im.size[0]
        im.draft("RGB", (im.size[0] // pc.DRAFT, im.size[1] // pc.DRAFT))
        im = im.convert("RGB")
        m = pc.masked_frame(im, boxes[k], im.size[0] / w0)
        x = torch.from_numpy(np.asarray(m.resize((448, 256), Image.BILINEAR), dtype="float32") / 255.0)
        f = torch.nn.functional.normalize(bg(x.permute(2, 0, 1)[None]), dim=1).numpy()
        gal = np.random.default_rng(0).standard_normal((TOPK, f.shape[1])).astype("float32")
        return gal @ f[0]

    res = {"threads": torch.get_num_threads(),
           "mac_ms": {"siglip_b1_forward": timed(lambda: vit(xv)), "resnet18_448x256_forward": timed(lambda: bg(xb)),
                      "background_full_query": timed(bg_full)},
           "gflops": {"siglip_224": flops(vit, xv) / 1e9, "resnet18_448x256": flops(bg, xb) / 1e9}}
    r = res["gflops"]["resnet18_448x256"] / res["gflops"]["siglip_224"]
    res["gpu_4060_estimate"] = {"siglip_ms": GPU_SIGLIP_MS, "with_background_ms": GPU_SIGLIP_MS * (1 + r),
                                "fps_before": 1000 / GPU_SIGLIP_MS, "fps_after": 1000 / (GPU_SIGLIP_MS * (1 + r)),
                                "flops_ratio": r}
    # DBA (research/dba.py, k=1): пересчёт галереи 10.8k × 512 и добавление одного кадра
    rng = np.random.default_rng(0)
    g = dbam.l2(rng.standard_normal((10847, 512)).astype("float32"))
    t0 = time.perf_counter()
    dbam.dba(g, k=1)
    rebuild = time.perf_counter() - t0
    st, new = dbam.DBAGallery(g, k=1), dbam.l2(rng.standard_normal((60, 512)).astype("float32"))
    j = [0]

    def add_one():
        j[0] += 1
        return st.add(new[j[0] % len(new)][None])

    res["dba"] = {"rebuild_10847_s": rebuild, "add_one_frame_ms": timed(add_one, n=50)}
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
