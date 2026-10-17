"""Сверка sha256 весов при старте (app/weights_check.py): только лог, ответы API не меняются.

    python -m unittest tests.test_weights_check -v      # из каталога api/
"""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import weights_check as wc


class WeightsCheckTest(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = Path(d.name) / "w.pt"
        self.path.write_bytes(b"weights" * 1000)
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_match_mismatch_and_unset(self):
        with patch.dict(os.environ, {"REID_WEIGHTS": str(self.path)}):
            with self.assertLogs("case17", "INFO") as cm:
                self.assertTrue(wc.check("reid:vit", self.sha.upper()))
            self.assertIn(self.sha, cm.output[0])
            with self.assertLogs("case17", "WARNING") as cm:
                self.assertFalse(wc.check("reid:vit", "0" * 64))
            self.assertIn("не те веса", cm.output[0])
            with self.assertLogs("case17", "INFO") as cm:
                self.assertIsNone(wc.check("reid:vit", ""))
            self.assertIn(self.sha, cm.output[0])

    def test_no_weights_file(self):
        self.assertIsNone(wc.check("resnet18", "abc"))
        with patch.dict(os.environ, {"REID_WEIGHTS": str(self.path) + ".missing"}):
            self.assertIsNone(wc.check("reid:vit", "abc"))
        self.assertEqual(wc.weights_path(f"onnx:{self.path}"), self.path)


if __name__ == "__main__":
    unittest.main()
