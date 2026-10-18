"""Сверка sha256 весов при старте (app/weights_check.py): лог и поле weights_sha256 в /health.

    python -m unittest tests.test_weights_check -v      # из каталога api/
"""
import hashlib
import json
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


class HealthShaTest(unittest.TestCase):
    """/health: sha из lifespan, без файла весов — прежнее тело байт в байт; sha не пересчитывается."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import service_fakes as sf

        self.main = sf.load_app()
        self.addCleanup(lambda: setattr(self.main.app.state, "weights_sha256", None))
        self.get = lambda path: sf.call(self.main.app, "GET", path)

    def test_without_sha_body_is_unchanged(self):
        self.main.app.state.weights_sha256 = None
        r = self.get("/health")
        m = self.main.app.state.model
        self.assertEqual((r["status"], r["body"]), (200, '{"status":"ok","model":"%s","dim":%d}' % (m.name, m.dim)))

    def test_sha_from_startup_without_rehash(self):
        self.main.app.state.weights_sha256 = "a" * 64
        with patch.object(wc, "sha256_file", side_effect=AssertionError("пересчёт sha на запросе")):
            r = self.get("/health")
        self.assertEqual(json.loads(r["body"])["weights_sha256"], "a" * 64)

    def test_inspect_returns_sha(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = Path(d.name) / "w.pt"
        path.write_bytes(b"x" * 10)
        with patch.dict(os.environ, {"REID_WEIGHTS": str(path)}), self.assertLogs("case17", "INFO"):
            self.assertEqual(wc.inspect("reid:vit", ""), (hashlib.sha256(b"x" * 10).hexdigest(), None))
        self.assertEqual(wc.inspect("resnet18", "abc"), (None, None))


if __name__ == "__main__":
    unittest.main()
