"""Плохая строка CSV не останавливает прогон: рамка за кадром обрезается, пустая — весь кадр, нечитаемый
кадр — серый (app/imaging.fit_bbox). make_submission.py и extractor.extract() ведут себя одинаково.

    python -m unittest tests.test_frame_fixes -v      # из каталога api/
"""
import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.config import settings
from app.extractor import Extractor
from app.imaging import crop_with_pad, fit_bbox, gray_frame
from tests.service_fakes import DIM, FakeModel
from tests.test_logging import keep_root_logging

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
W, H = 64, 48


def make_submission():
    spec = importlib.util.spec_from_file_location("make_submission", SCRIPTS / "make_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frame(seed: int) -> Image.Image:
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8))


def same(a: Image.Image, b: Image.Image) -> bool:
    return a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b))


class LoadableFake(FakeModel):
    def load(self):
        pass


class FitBboxTest(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(fit_bbox(W, H, 5, 5, 20, 20), ((5, 5, 20, 20), "ok"))
        self.assertEqual(fit_bbox(W, H, 50, 40, 30, 30), ((50, 40, 14, 8), "clipped"))
        self.assertEqual(fit_bbox(W, H, -4, -2, 10, 10), ((0, 0, 6, 8), "clipped"))
        self.assertEqual(fit_bbox(W, H, 100, 100, 10, 10), ((0, 0, W, H), "full"))      # целиком за кадром
        self.assertEqual(fit_bbox(W, H, 63, 10, 10, 10), ((0, 0, W, H), "full"))        # после обрезки 1 px
        self.assertEqual(fit_bbox(W, H, 5, 5, 0, 10), ((0, 0, W, H), "full"))          # пустая
        self.assertEqual(fit_bbox(W, H, 62, 10, 10, 10), ((62, 10, 2, 10), "clipped"))   # 2 px — ещё рамка


class SubmissionFramesTest(unittest.TestCase):
    def setUp(self):
        self.ms = make_submission()
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        (self.dir / "images").mkdir()
        self.img = frame(0)
        self.img.save(self.dir / "images" / "a.jpg", quality=95)
        (self.dir / "images" / "bad.jpg").write_bytes(b"not a jpeg")
        self.frames = self.ms.Frames(self.dir)
        self.decoded = self.ms.load_image((self.dir / "images" / "a.jpg").read_bytes(), settings.max_upload_bytes)

    def load(self, image_id, bbox, split="query"):
        with self.assertLogs("case17.submission", "WARNING") as cm:
            out = self.frames.load({"image_id": image_id, "bbox": bbox}, True, split)
        return out, cm.output

    def test_clipped_bbox(self):
        (_, crop, box), out = self.load("a", (50, 40, 30, 30))
        self.assertEqual(box, (50, 40, 14, 8))
        self.assertTrue(same(crop, crop_with_pad(self.decoded, 50, 40, 14, 8, settings.bbox_pad)))
        self.assertEqual(len(out), 1)
        self.assertRegex(out[0], r"^WARNING:.*query image_id a: рамка \(50, 40, 30, 30\) выходит за кадр 64×48 — "
                                 r"обрезана до \(50, 40, 14, 8\)$")

    def test_empty_bbox_takes_whole_frame(self):
        (_, crop, box), out = self.load("a", (100, 100, 10, 10), "gallery")
        self.assertEqual(box, (0, 0, W, H))
        self.assertTrue(same(crop, self.decoded))
        self.assertIn("gallery image_id a: рамка (100, 100, 10, 10)", out[0])
        self.assertIn("беру весь кадр (0, 0, 64, 48)", out[0])

    def test_unreadable_and_missing_frame_give_gray(self):
        for image_id in ("bad", "missing"):
            (full, crop, box), out = self.load(image_id, (1, 1, 10, 10))
            S = settings.input_size
            self.assertEqual(box, (0, 0, S, S))
            self.assertTrue(same(full, gray_frame(S)) and same(crop, gray_frame(S)))
            self.assertTrue(out[0].startswith("ERROR:") and f"query image_id {image_id}: кадр не прочитан" in out[0])
        self.assertEqual(self.frames.fixes_line(), "кадры: рамок обрезано 0, заменено на весь кадр 0, не прочитано 2")

    def test_gray_crop_matches_extractor(self):
        (_, crop, _), _ = self.load("bad", (1, 1, 10, 10))
        with self.assertLogs("case17.extractor", "ERROR"):
            ex_crop = Extractor(model=FakeModel()).crop(self.dir / "images" / "bad.jpg", (1, 1, 10, 10))
        self.assertTrue(same(crop, ex_crop))


class SubmissionRunTest(unittest.TestCase):
    """Сквозной прогон с подставной моделью: плохие строки не роняют сборку, строк в embeddings.npy — как в CSV."""

    def test_run_keeps_every_row(self):
        ms = make_submission()
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        keep_root_logging(self)
        (root / "images").mkdir()
        for i, name in enumerate(("q_clip", "q_full", "g_ok")):
            frame(i).save(root / "images" / f"{name}.jpg", quality=95)
        (root / "images" / "g_bad.jpg").write_bytes(b"\xff\xd8broken")
        (root / "test_query.csv").write_text("image_id,x,y,w,h\nq_clip,50,40,30,30\nq_full,90,90,5,5\n"
                                             "q_missing,1,1,10,10\n", encoding="utf-8")
        (root / "test_gallery.csv").write_text("image_id,x,y,w,h\ng_ok,5,5,30,30\ng_bad,5,5,30,30\n", encoding="utf-8")
        out = root / "out"
        argv = ["make_submission.py", "--data", str(root), "--out", str(out)]
        with patch.object(ms, "build_model", lambda name, size: LoadableFake()), patch("sys.argv", argv), \
                patch.dict("os.environ", {"REID_WEIGHTS": ""}):
            rc = ms.main()
        self.assertEqual(rc, 0)
        emb = np.load(out / "embeddings.npy")
        self.assertEqual(emb.shape, (5, DIM))
        self.assertTrue(np.isfinite(emb).all())
        idx = [r["image_id"] for r in csv.DictReader((out / "embeddings_index.csv").read_text(encoding="utf-8").splitlines())]
        self.assertEqual(idx, ["q_clip", "q_full", "q_missing", "g_ok", "g_bad"])
        sub = list(csv.DictReader((out / "submission.csv").read_text(encoding="utf-8").splitlines()))
        self.assertEqual([r["query_id"] for r in sub], ["q_clip", "q_full", "q_missing"])
        log_lines = (out / "make_submission.log").read_text(encoding="utf-8").splitlines()
        self.assertIn("WARNING кадры: рамок обрезано 1, заменено на весь кадр 1, не прочитано 2", log_lines[-2])
        self.assertTrue(log_lines[-1].endswith("код выхода 0, лог: " + str(out / "make_submission.log")))
        meta = json.loads((out / "postprocess.json").read_text(encoding="utf-8"))
        self.assertEqual({k: meta["frame_fixes"][k] for k in ("clipped", "full_frame", "unread_gray")},
                         {"clipped": 1, "full_frame": 1, "unread_gray": 2})


class ExtractorFixesTest(unittest.TestCase):
    def test_full_frame_and_unread_in_both_paths(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        frame(3).save(d / "f.jpg", quality=95)
        (d / "bad.jpg").write_bytes(b"\xff\xd8broken")
        img = Image.open(d / "f.jpg").convert("RGB")
        for draft in (False, True):
            ex = Extractor(model=FakeModel(), draft=draft)
            with self.assertLogs("case17.extractor", "WARNING") as cm:
                full = ex.crop(d / "f.jpg", (100, 100, 10, 10))
                gray = ex.crop(d / "bad.jpg", (1, 1, 10, 10))
                missing = ex.crop(d / "nope.jpg", (1, 1, 10, 10))
            self.assertTrue(same(full, img))
            self.assertTrue(same(gray, gray_frame(settings.input_size)) and same(missing, gray))
            self.assertEqual(ex.describe()["frame_fixes"], {"clipped": 0, "full": 1, "unread": 2})
            self.assertEqual([line.split(":", 1)[0] for line in cm.output], ["WARNING", "ERROR", "ERROR"])
            v = ex.extract_batch([(d / "bad.jpg", (1, 1, 10, 10)), (d / "f.jpg", (5, 5, 20, 20))])
            self.assertEqual(v.shape, (2, DIM))


if __name__ == "__main__":
    unittest.main()
