"""extract() — полный цикл на одно ТС (app/extractor.py, ответ Q31): кроп как у сервиса, флаги ускорения.

    python -m unittest tests.test_extractor -v      # из каталога api/
"""
import io
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.extractor import Extractor
from app.imaging import crop_with_pad, load_image
from tests.service_fakes import FakeModel

FRAME = (1920, 1080)
BOX = (800, 240, 900, 560)                     # кроп с запасом ~1080×672: draft 1/2 ещё ≥ 224 по обеим сторонам


def jpeg(seed: int) -> bytes:
    a = np.random.default_rng(seed).integers(0, 256, (FRAME[1] // 8, FRAME[0] // 8, 3), dtype=np.uint8)
    im = Image.fromarray(a).resize(FRAME, Image.BILINEAR)          # гладкий кадр, как у камеры
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class ExtractorTest(unittest.TestCase):
    def test_production_defaults(self):
        """Прод-дефолты пути замера: draft=2 и 16 потоков — случайный откат не должен пройти незамеченным."""
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXTRACT_JPEG_DRAFT", None)
            os.environ.pop("EXTRACT_THREADS", None)
            ex = Extractor(model=FakeModel())
        self.assertTrue(ex.draft)
        self.assertEqual(ex.draft_max, 2)
        self.assertEqual(ex.threads, 16)

    def test_crop_matches_service_without_draft(self):
        raw = jpeg(0)
        want = crop_with_pad(load_image(raw, settings.max_upload_bytes), *BOX, settings.bbox_pad)
        got = Extractor(model=FakeModel(), draft=False, threads=1).crop(raw, BOX)
        self.assertEqual(got.size, want.size)
        self.assertTrue(np.array_equal(np.asarray(got), np.asarray(want)))

    def test_draft_decodes_smaller_but_keeps_input_size(self):
        raw = jpeg(1)
        full = Extractor(model=FakeModel(), draft=False).crop(raw, BOX)
        small = Extractor(model=FakeModel(), draft=True).crop(raw, BOX)
        self.assertLess(small.width, full.width)
        self.assertGreaterEqual(min(small.size), settings.input_size)
        self.assertAlmostEqual(small.width / full.width, 0.5, delta=0.01)

    def test_draft_half_caps_at_half(self):
        raw = jpeg(3)
        big = (0, 0, 1820, 990)        # кроп с запасом ~1920×1080: draft=1 уменьшил бы до 1/4
        half = Extractor(model=FakeModel(), draft="2").crop(raw, big)
        quarter = Extractor(model=FakeModel(), draft=True).crop(raw, big)
        self.assertEqual(half.width, quarter.width * 2)

    def test_bbox_outside_frame_clipped_in_both_paths(self):
        raw = jpeg(2)
        for draft in (False, True):
            ex = Extractor(model=FakeModel(), draft=draft)
            with self.assertLogs("case17.extractor", "WARNING") as cm:
                got = ex.crop(raw, (1800, 900, 400, 400))
                ex.crop(raw, (1800, 900, 400, 400))                   # второй раз — без новой строки
            want = Extractor(model=FakeModel(), draft=draft).crop(raw, (1800, 900, 120, 180))
            self.assertTrue(np.array_equal(np.asarray(got), np.asarray(want)))
            self.assertEqual(len(cm.output), 1)
            self.assertIn("обрезана до (1800, 900, 120, 180)", cm.output[0])
            self.assertEqual(ex.describe()["frame_fixes"], {"clipped": 1, "full": 0, "unread": 0})

    def test_file_path_and_threads_do_not_change_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            items = []
            for i in range(6):
                p = Path(d) / f"{i}.jpg"
                p.write_bytes(jpeg(10 + i))
                items.append((p, BOX))
            one = Extractor(model=FakeModel(), threads=1).extract_batch(items)
            many = Extractor(model=FakeModel(), threads=4).extract_batch(items)
            single = np.stack([Extractor(model=FakeModel()).extract(*it) for it in items])
        self.assertTrue(np.array_equal(one, many))
        self.assertTrue(np.allclose(one, single))
        self.assertEqual(one.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
