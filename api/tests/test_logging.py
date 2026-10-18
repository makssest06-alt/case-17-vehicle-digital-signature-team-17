"""Логирование и фиксация весов: подпись модели по чекпоинту, eval() и forward под inference_mode, image_id
в ошибке кадра, лог make_submission в файл, чистый JSON в stdout у bench_extract.

    python -m unittest tests.test_logging -v      # из каталога api/
"""
import contextlib
import importlib.util
import io
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.extractor import Extractor
from app.models import reid_checkpoint as rc
from tests.service_fakes import FakeModel

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def keep_root_logging(test: unittest.TestCase):
    """basicConfig(force=True) в скриптах меняет корневой логгер — вернуть как было."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level

    def restore():
        for h in root.handlers:
            if h not in handlers:
                h.close()
        root.handlers[:], root.level = handlers, level
    test.addCleanup(restore)


class TinyBackbone(nn.Module):
    num_features = 5

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 5)

    def forward(self, x):
        return self.fc(x)


def tiny_checkpoint(path: Path, backbone: str = "siglip2_base") -> None:
    enc = {f"encoder.{k}": v for k, v in TinyBackbone().state_dict().items()}
    head = {f"head.{k}": v for k, v in rc._ReidHead(5, 4, 3, 0.0).state_dict().items()}
    args = {"hidden": 4, "out_dim": 3, "dropout": 0.0, "tf_mode": "squash", "size": 12}
    if backbone:
        args["backbone"] = backbone
    torch.save({"args": args, "model": {**enc, **head}}, path)


class ModelNameAndFreezeTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "ck.pt"

    def test_name_from_checkpoint_without_building(self):
        tiny_checkpoint(self.path)
        self.assertEqual(rc.checkpoint_model_name("vit_base_patch16_siglip_224.webli", str(self.path)),
                         "reid:vit_base_patch16_siglip_224.v2_webli")

    def test_name_falls_back_to_embed_model(self):
        tiny_checkpoint(self.path, backbone="")
        self.assertEqual(rc.checkpoint_model_name("vit_x", str(self.path)), "reid:vit_x")

    def test_load_names_by_checkpoint_eval_inference_and_logs(self):
        tiny_checkpoint(self.path)
        m = rc.ReidCheckpointModel("vit_base_patch16_siglip_224.webli", str(self.path), 12)
        with patch.object(rc.ReidCheckpointModel, "_create_backbone", return_value=(TinyBackbone(), "tiny")), \
                patch.object(rc.ReidCheckpointModel, "_build_tf", return_value=lambda im: torch.ones(2)), \
                patch.dict("os.environ", {"REID_DEVICE": "cpu"}), \
                self.assertLogs("case17.model", "INFO") as cm:
            m.load()
        self.assertEqual(m.name, "reid:vit_base_patch16_siglip_224.v2_webli")
        self.assertFalse(m._model.training or m._head.training)
        self.assertTrue(any("устройство cpu" in line and "FP16 нет" in line for line in cm.output))
        self.assertIn("INFO:case17.model:бэкбон: tiny (веса SigLIP2 из чекпоинта)", cm.output)
        self.assertFalse([line for line in cm.output if "не разреш" in line or "EMBED_MODEL" in line])
        seen = []
        encode = m._encode
        with patch.object(m, "_encode", side_effect=lambda x: seen.append(torch.is_inference_mode_enabled())
                          or encode(x)):                    # TTA по умолчанию выкл. — один проход
            v = m.embed_batch([Image.new("RGB", (4, 4))])
        self.assertEqual(seen, [True])
        self.assertEqual(v.shape, (1, 3))


class SubmissionLogTest(unittest.TestCase):
    def setUp(self):
        keep_root_logging(self)
        self.ms = script("make_submission")
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        (self.dir / "images").mkdir()
        Image.new("RGB", (40, 30)).save(self.dir / "images" / "q1.jpg")

    def test_bad_bbox_and_missing_frame_name_image_id(self):
        frames = self.ms.Frames(self.dir)
        with self.assertRaises(self.ms.FrameError) as cm:
            frames.load({"image_id": "q1", "bbox": (30, 20, 20, 20)}, False, "query")
        self.assertIn("image_id q1", str(cm.exception))
        self.assertIn("выходит за границы кадра", str(cm.exception))
        with self.assertRaises(self.ms.FrameError) as cm:
            frames.load({"image_id": "nope", "bbox": (0, 0, 5, 5)}, False, "gallery")
        self.assertIn("gallery image_id nope", str(cm.exception))

    def test_log_goes_to_file_in_main_py_format(self):
        path = self.ms.setup_logging(self.dir)
        logging.getLogger("case17.submission").info("проверка %d", 1)
        for h in logging.getLogger().handlers:
            h.flush()
        line = path.read_text(encoding="utf-8").strip()
        self.assertRegex(line, r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} INFO проверка 1$")


class LoadableFake(FakeModel):
    def load(self):
        pass


class BenchStdoutTest(unittest.TestCase):
    def test_stdout_is_json_and_stages_go_to_stderr(self):
        keep_root_logging(self)
        bench = script("bench_extract")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "images").mkdir()
            rows = ["image_id,x,y,w,h"]
            for i in range(3):
                a = np.random.default_rng(i).integers(0, 256, (60, 80, 3), dtype=np.uint8)
                Image.fromarray(a).save(root / "images" / f"f{i}.jpg")
                rows.append(f"f{i},10,10,40,30")
            (root / "test_query.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with patch.object(bench, "Extractor", side_effect=lambda: Extractor(model=LoadableFake())), \
                    patch("sys.argv", ["bench_extract.py", "--data", d, "--smoke", "--batches", "1,2", "--scan"]), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(bench.main(), 0)
        res = json.loads(out.getvalue())
        self.assertEqual(res["frames"], 3)
        self.assertIn("latency_b1: медиана", err.getvalue())
        self.assertIn("extract(): модель fake загружена", err.getvalue())


if __name__ == "__main__":
    unittest.main()
