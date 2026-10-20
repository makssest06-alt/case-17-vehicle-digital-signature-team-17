"""Один кроп на всё решение: extract() (замер скорости, Q31), make_submission (три файла, Q41) и сервис (/v1/*)
режут кадр одним app.imaging.crop_frame — JPEG draft и всё остальное у них общее, векторы совпадают.

    python -m unittest tests.test_crop_parity -v      # из каталога api/
"""
import csv
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.config import settings
from app.extractor import Extractor
from app.imaging import crop_with_pad, load_image
from tests.service_fakes import FakeModel, load_app
from tests.test_frame_fixes import LoadableFake, make_submission
from tests.test_logging import keep_root_logging

FRAMES = {"even": (1920, 1080), "odd": (1919, 1079)}          # нечётная сторона — floor из b2fb29c
BOXES = {"big": (800, 240, 900, 560),                           # кроп ~1080×672: draft 1/2
         "small": (100, 100, 200, 150),                         # кроп ~240×180: без уменьшения
         "clipped": (1700, 900, 400, 400)}                      # за кадром — обрезка fit_bbox


def jpeg(size, seed: int) -> bytes:
    a = np.random.default_rng(seed).integers(0, 256, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).resize(size, Image.BILINEAR).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def same(a: Image.Image, b: Image.Image) -> bool:
    return a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b))


class CropParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        (cls.dir / "images").mkdir()
        cls.raw = {}
        for i, (name, size) in enumerate(FRAMES.items()):
            cls.raw[name] = jpeg(size, i)
            (cls.dir / "images" / f"{name}.jpg").write_bytes(cls.raw[name])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def test_make_submission_crop_is_extract_crop(self):
        frames = make_submission().Frames(self.dir)
        ex = Extractor(model=FakeModel(), threads=1)
        for f in FRAMES:
            for b, box in BOXES.items():
                with self.subTest(frame=f, box=b), self.assertNoLogs("case17.submission", "ERROR"):
                    _, crop, _ = frames.load({"image_id": f, "bbox": box}, False)
                    self.assertTrue(same(crop, ex.crop(self.raw[f], box)))

    def test_draft_really_applied(self):
        """Паритет не пустой: на большом кропе обе стороны декодированы вдвое меньше полного кадра."""
        frames = make_submission().Frames(self.dir)
        for f in FRAMES:
            with self.subTest(frame=f):
                _, crop, _ = frames.load({"image_id": f, "bbox": BOXES["big"]}, False)
                full = crop_with_pad(load_image(self.raw[f], settings.max_upload_bytes), *BOXES["big"],
                                     settings.bbox_pad)
                self.assertAlmostEqual(crop.width / full.width, 0.5, delta=0.01)
                self.assertGreaterEqual(min(crop.size), settings.input_size)

    def test_small_crop_equals_full_decode(self):
        """Где уменьшать нельзя (кроп меньше 2 × INPUT_SIZE), кроп — ровно crop_with_pad полного кадра."""
        for f in FRAMES:
            with self.subTest(frame=f):
                full = crop_with_pad(load_image(self.raw[f], settings.max_upload_bytes), *BOXES["small"],
                                     settings.bbox_pad)
                self.assertTrue(same(Extractor(model=FakeModel()).crop(self.raw[f], BOXES["small"]), full))

    def test_frame_for_background_is_full_decode(self):
        """Признак фона считается по полному кадру — по нему калибровался τ (app/pp/background.py)."""
        frames = make_submission().Frames(self.dir)
        full, _, _ = frames.load({"image_id": "even", "bbox": BOXES["big"]}, True)
        self.assertTrue(same(full, load_image(self.raw["even"], settings.max_upload_bytes)))

    def test_service_crop_is_extract_crop(self):
        main = load_app()
        ex = Extractor(model=FakeModel(), threads=1)
        for f in FRAMES:
            for b in ("big", "small"):                                  # сервис рамку за кадром отвергает
                with self.subTest(frame=f, box=b):
                    self.assertTrue(same(main._prepare(self.raw[f], *BOXES[b]), ex.crop(self.raw[f], BOXES[b])))
                    frame, crop = main._prepare_frame(self.raw[f], *BOXES[b])
                    self.assertTrue(same(crop, ex.crop(self.raw[f], BOXES[b])))
                    self.assertEqual(frame.size, FRAMES[f])

    def test_extract_reproduces_embeddings_npy(self):
        """Сквозной прогон make_submission с подставной моделью: embeddings.npy == extract_batch тех же строк."""
        ms = make_submission()
        root = self.dir
        keep_root_logging(self)
        for src, dst in (("even", "even_c"), ("odd", "odd_g")):                # image_id в файле не повторяются
            shutil.copy(root / "images" / f"{src}.jpg", root / "images" / f"{dst}.jpg")
        rows = [("even", BOXES["big"]), ("odd", BOXES["small"]), ("even_c", BOXES["clipped"])]
        q = "image_id,x,y,w,h\n" + "".join(f"{i},{x},{y},{w},{h}\n" for i, (x, y, w, h) in rows)
        (root / "test_query.csv").write_text(q, encoding="utf-8")
        (root / "test_gallery.csv").write_text("image_id,x,y,w,h\nodd_g,800,240,900,560\n", encoding="utf-8")
        out = root / "out"
        argv = ["make_submission.py", "--data", str(root), "--out", str(out)]
        with patch.object(ms, "build_model", lambda name, size: LoadableFake()), patch("sys.argv", argv), \
                patch.dict("os.environ", {"REID_WEIGHTS": ""}):
            self.assertEqual(ms.main(), 0)
        emb = np.load(out / "embeddings.npy")
        idx = list(csv.DictReader((out / "embeddings_index.csv").read_text(encoding="utf-8").splitlines()))
        items = [(root / "images" / f"{i}.jpg", b) for i, b in rows] + [(root / "images" / "odd_g.jpg", BOXES["big"])]
        want = Extractor(model=FakeModel(), threads=4).extract_batch(items)
        self.assertEqual([r["image_id"] for r in idx], ["even", "odd", "even_c", "odd_g"])
        self.assertTrue(np.allclose(emb, want, rtol=0, atol=1e-6), np.abs(emb - want).max())


if __name__ == "__main__":
    unittest.main()
