"""CPU smoke coverage using the real embed_batch with tiny mocked internals.

Run: python -m unittest discover -s api/tests -p test_benchmark_reid.py
Requires torch, numpy and Pillow; neither timm nor checkpoint weights are needed.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_reid.py"
SPEC = importlib.util.spec_from_file_location("benchmark_reid", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)

from app import config
from app.models.reid_checkpoint import ReidCheckpointModel


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.model = ReidCheckpointModel("mock", "unused", input_size=12)
        self.model.device = torch.device("cpu")
        self.model.dim = 3
        self.model.tf_mode = "mock"
        self.model._fp16 = False
        self.model._model = object()
        self.model._tf = lambda im: torch.from_numpy(
            np.array(im, dtype=np.float32).copy()).permute(2, 0, 1)
        self.calls = []

        def encode(x):
            self.calls.append(tuple(x.shape))
            return x.mean(dim=(2, 3))

        self.model._encode = encode

    def test_cpu_cli_real_embed_and_serialized_stages(self):
        original = config.settings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 20), (100, 50, 25)).save(root / "car.jpg")
            (root / "boxes.csv").write_text("image_id,x,y,w,h\ncar,2,2,10,10\n")
            (root / "weights.pt").write_bytes(b"mock-checkpoint")
            with patch.object(benchmark, "load_model", return_value=self.model):
                self.assertEqual(benchmark.main([
                    "--weights", str(root / "weights.pt"), "--images-dir", str(root),
                    "--csv", str(root / "boxes.csv"), "--batches", "1,2", "--tta", "both",
                    "--warmup", "1", "--iterations", "2", "--json", str(root / "out.json"),
                ]), 0)
            report = json.loads((root / "out.json").read_text())
            self.assertEqual(len(report["results"]), 4)
            # Each size: off=5 encodes, on=10 (warmup + E2E + diagnostics).
            self.assertEqual(len(self.calls), 30)
            self.assertTrue(all(shape[1:] == (3, 12, 12) for shape in self.calls))
            for result in report["results"]:
                stats = result["end_to_end"]
                self.assertGreater(stats["images_per_second"], 0)
                self.assertEqual(stats["measured_images"], 2 * result["batch_size"])
                self.assertIsNone(stats["peak_gpu_allocated_bytes"])
                self.assertEqual(set(result["serialized_diagnostic"]["stages"]), {
                    "decode_crop", "transform_stack", "h2d", "encode_tta", "normalize_d2h"})
            self.assertEqual(len(report["metadata"]["checkpoint_sha256"]), 64)
        self.assertIs(config.settings, original)

    def test_restores_settings_on_failure(self):
        args = benchmark.parser().parse_args([
            "--weights", "unused", "--images-dir", ".", "--csv", "unused"])
        original = config.settings
        with patch.object(benchmark, "end_to_end", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                benchmark.benchmark(self.model, [object()], args)
        self.assertIs(config.settings, original)

    def test_rejects_head_only_before_production_load(self):
        with patch.object(torch, "load", return_value={"head": {}}), \
                patch.object(ReidCheckpointModel, "load") as load:
            with self.assertRaisesRegex(ValueError, "head-only"):
                benchmark.load_model(Path("unused"))
            load.assert_not_called()

    def test_sample_paths_match_production_jpg_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (10, 10)).save(root / "car.jpg")
            csv_path = root / "boxes.csv"
            csv_path.write_text("image_id,x,y,w,h\ncar,1,1,5,5\n")
            samples = benchmark.read_samples(csv_path, root, 1)
            self.assertEqual(samples[0].path, (root / "car.jpg").resolve())

            csv_path.write_text("image_id,x,y,w,h\ncar.jpg,1,1,5,5\n")
            with self.assertRaisesRegex(ValueError, "extensionless"):
                benchmark.read_samples(csv_path, root, 1)

            csv_path.write_text("image_id,x,y,w,h\n../car,1,1,5,5\n")
            with self.assertRaises(ValueError):
                benchmark.read_samples(csv_path, root, 1)


if __name__ == "__main__":
    unittest.main()
