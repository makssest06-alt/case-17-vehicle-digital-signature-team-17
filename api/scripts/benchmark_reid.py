#!/usr/bin/env python3
"""Benchmark the production ReID path without a running API or downloads.

Run from any directory with the API's Python dependencies installed. Only full
checkpoints (encoder.* plus head.*) are supported: head-only weights would make
the production loader request pretrained weights. Checkpoints must be trusted,
as the production loader uses torch.load(weights_only=False).

Each configuration runs --warmup batches, then --iterations measured batches,
cycling through the first --samples CSV rows to fill every requested batch.
Files are reread every time; OS filesystem caching is not disabled. Serialized
stage diagnostics are a separate pass, not components of the end-to-end result.
"""

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))


@dataclass(frozen=True)
class Sample:
    path: Path
    bbox: tuple


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def batch_sizes(value):
    try:
        return list(dict.fromkeys(positive(part) for part in value.split(",")))
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("expected positive sizes, e.g. 1,8,16,32") from exc


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="directory containing production-style image files")
    ap.add_argument("--csv", type=Path, required=True,
                    help="CSV with image_id,x,y,w,h; IDs are relative paths or extensionless names")
    ap.add_argument("--batches", type=batch_sizes, default=batch_sizes("1,8,16,32"))
    ap.add_argument("--tta", choices=("off", "on", "both"), default="both")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=positive, default=10)
    ap.add_argument("--samples", type=positive, default=64,
                    help="maximum CSV rows in the cycling sample pool (default: 64)")
    ap.add_argument("--json", type=Path, help="optional JSON report path")
    ap.add_argument("--require-cuda", action="store_true",
                    help="fail rather than run a CPU smoke benchmark")
    return ap


def read_samples(csv_path, images_dir, limit):
    root = images_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"--images-dir is not a directory: {root}")
    samples = []
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(("image_id", "x", "y", "w", "h")).issubset(reader.fieldnames or []):
            raise ValueError("CSV requires image_id,x,y,w,h columns")
        for line, row in enumerate(reader, 2):
            image_id = row["image_id"].strip()
            # Submission/gallery import always uses ``image_id.jpg``.
            # Keep this strict so the benchmark cannot silently measure a
            # path production would never process.
            if Path(image_id).suffix:
                raise ValueError(f"CSV line {line}: image_id must be extensionless")
            path = (root / f"{image_id}.jpg").resolve()
            if not image_id or not path.is_relative_to(root):
                raise ValueError(f"CSV line {line}: image_id must be inside --images-dir")
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"CSV line {line}: missing/invalid image {image_id!r}")
            try:
                bbox = tuple(int(row[key]) for key in ("x", "y", "w", "h"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CSV line {line}: bbox coordinates must be integers") from exc
            samples.append(Sample(path, bbox))
            if len(samples) == limit:
                break
    if not samples:
        raise ValueError("CSV has no samples")
    return samples


def decode_crop(samples):
    from app import config
    from app.imaging import crop_with_pad, load_image, validate_bbox

    crops = []
    try:
        for sample in samples:
            image = load_image(sample.path.read_bytes(), config.settings.max_upload_bytes)
            try:
                validate_bbox(image, *sample.bbox)
                crops.append(crop_with_pad(image, *sample.bbox, config.settings.bbox_pad))
            finally:
                image.close()
    except Exception:
        for crop in crops:
            crop.close()
        raise
    return crops


def load_model(weights):
    import torch
    from app import config
    from app.models.reid_checkpoint import ReidCheckpointModel

    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not any(key.startswith("encoder.") for key in state):
        raise ValueError("head-only checkpoint rejected: production loading could download weights")
    cfg = checkpoint.get("cfg") or checkpoint.get("args") or {}
    backbone = str(cfg.get("backbone") or config.settings.embed_model.removeprefix("reid:"))
    del checkpoint, state
    model = ReidCheckpointModel(backbone, str(weights), config.settings.input_size)
    model.load()
    return model


def synchronize(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def timed(device, operation):
    synchronize(device)
    start = time.perf_counter()
    result = operation()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000


def end_to_end(model, samples):
    # Keep all read/decode/validation/crop and the unchanged production method
    # inside the same wall-clock interval. No stage barriers in this path.
    crops = decode_crop(samples)
    try:
        return model.embed_batch(crops)
    finally:
        for crop in crops:
            crop.close()


def diagnostic(model, samples):
    """Mirror embed_batch's stages; never use this path for throughput claims."""
    import torch
    import torch.nn.functional as F
    from app import config

    device = model.device
    stages = {}
    with torch.inference_mode():
        crops, stages["decode_crop"] = timed(device, lambda: decode_crop(samples))
        try:
            x, stages["transform_stack"] = timed(
                device, lambda: torch.stack([model._tf(im.convert("RGB")) for im in crops]))
        finally:
            for crop in crops:
                crop.close()
        x, stages["h2d"] = timed(device, lambda: x.to(device, non_blocking=True))

        def encode():
            a = model._encode(x)
            b = model._encode(torch.flip(x, dims=[3])) if config.settings.tta_flip else None
            return a, b

        (a, b), stages["encode_tta"] = timed(device, encode)
        _, stages["normalize_d2h"] = timed(
            device, lambda: F.normalize(a if b is None else (a + b) / 2, p=2, dim=1)
            .cpu().numpy().astype("float32"))
    return stages


def latency_summary(values):
    import numpy as np
    return {"p50_batch_ms": float(np.percentile(values, 50)),
            "p95_batch_ms": float(np.percentile(values, 95)),
            "mean_batch_ms": float(np.mean(values)), "batch_ms": values}


def benchmark(model, samples, args):
    import torch
    from app import config

    original = config.settings
    results = []
    try:
        for tta in ([False, True] if args.tta == "both" else [args.tta == "on"]):
            config.settings = replace(original, tta_flip=tta)
            for size in args.batches:
                def batch(index):
                    return [samples[(index * size + offset) % len(samples)]
                            for offset in range(size)]

                for index in range(args.warmup):
                    end_to_end(model, batch(index))
                synchronize(model.device)
                cuda = model.device.type == "cuda"
                if cuda:
                    torch.cuda.reset_peak_memory_stats(model.device)
                times = []
                for index in range(args.iterations):
                    _, elapsed = timed(model.device, lambda: end_to_end(model, batch(index)))
                    times.append(elapsed)
                peak = torch.cuda.max_memory_allocated(model.device) if cuda else None
                reserved = torch.cuda.max_memory_reserved(model.device) if cuda else None
                # Read peaks BEFORE running the independently serialized diagnostics.
                stages = {}
                for index in range(args.iterations):
                    for name, elapsed in diagnostic(model, batch(index)).items():
                        stages.setdefault(name, []).append(elapsed)
                results.append({
                    "batch_size": size, "tta": "on" if tta else "off",
                    "end_to_end": {**latency_summary(times),
                                   "images_per_second": size * len(times) * 1000 / sum(times),
                                   "measured_images": size * len(times),
                                   "peak_gpu_allocated_bytes": peak,
                                   "peak_gpu_reserved_bytes": reserved},
                    "serialized_diagnostic": {
                        "note": "Separate pass; synchronization at every stage; not end-to-end latency. "
                                "H2D/D2H are CPU no-ops/copies when device=cpu.",
                        "stages": {name: latency_summary(values) for name, values in stages.items()}},
                })
    finally:
        config.settings = original
    return results


def metadata(model, args, samples):
    import torch
    from app import config

    digest = hashlib.sha256()
    with args.weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=API_DIR,
                                  capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "device": str(model.device),
        "device_name": torch.cuda.get_device_name(model.device)
        if model.device.type == "cuda" else platform.processor() or "CPU",
        "fp16": model._fp16, "model": model.name, "input_size": model.input_size,
        "tf_mode": model.tf_mode, "embedding_dim": model.dim,
        "bbox_pad": config.settings.bbox_pad, "torch_threads": torch.get_num_threads(),
        "checkpoint": str(args.weights.resolve()), "checkpoint_sha256": digest.hexdigest(),
        "git_revision": revision, "sample_pool_size": len(samples),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "protocol": "Full batches cycle through first N CSV rows; read bytes every batch; "
                    "OS cache uncontrolled; model loading/hash excluded; peaks include resident model.",
    }


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.warmup < 0:
        ap.error("--warmup must be nonnegative")
    try:
        import torch
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("--require-cuda requested but CUDA is unavailable")
        samples = read_samples(args.csv, args.images_dir, args.samples)
        model = load_model(args.weights)
        if args.require_cuda and model.device.type != "cuda":
            raise RuntimeError("--require-cuda requested but model is not on CUDA")
        report = {"metadata": metadata(model, args, samples),
                  "results": benchmark(model, samples, args)}
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Device: {model.device}; end-to-end read/decode/crop + production embed_batch")
        for result in report["results"]:
            stats = result["end_to_end"]
            print(f"batch={result['batch_size']:3} TTA={result['tta']:3} "
                  f"p50={stats['p50_batch_ms']:.2f}ms p95={stats['p95_batch_ms']:.2f}ms "
                  f"images/s={stats['images_per_second']:.2f} "
                  f"peak_GPU_bytes={stats['peak_gpu_allocated_bytes']}")
            print("  Serialized diagnostic stage p50 ms (separate pass): " + ", ".join(
                f"{name}={value['p50_batch_ms']:.2f}"
                for name, value in result["serialized_diagnostic"]["stages"].items()))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        ap.exit(1, f"benchmark_reid: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
