"""Тесты дешёвого признака фона research/cheap_bg.py на синтетических кадрах.

Запуск из корня репозитория:
    python -m unittest discover -s research/tests -v
"""
import unittest

import numpy as np
from PIL import Image

from research import cheap_bg as cb

BOX = (700, 400, 500, 300)          # x, y, w, h в кадре 1920×1080


def frame(seed=0, car=None):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    if car is not None:
        x, y, w, h = BOX
        px, py = int(0.1 * w), int(0.1 * h)
        a[y - py:y + h + py, x - px:x + w + px] = car
    return Image.fromarray(a)


class CarPixelsTests(unittest.TestCase):
    """Признак описывает фон: машина (bbox + 10 %) на него не влияет."""

    def test_car_pixels_do_not_change_any_part(self):
        a = cb.describe(frame(car=0), BOX)
        b = cb.describe(frame(car=255), BOX)
        for part in cb.PARTS:
            np.testing.assert_allclose(a[part], b[part], atol=1e-6, err_msg=part)

    def test_background_change_changes_every_part(self):
        a = cb.describe(frame(seed=0, car=0), BOX)
        b = cb.describe(frame(seed=1, car=0), BOX)
        for part in cb.PARTS:
            self.assertLess(float(a[part] @ b[part]), 0.999, part)

    def test_car_pixels_do_not_leak_on_non_divisible_frames(self):
        """Кадр не кратен 128×72: ужатие не должно протаскивать машину за маску."""
        rng = np.random.default_rng(5)
        for (W, H), box in (((1366, 768), (834, 435, 532, 333)), ((800, 600), (700, 318, 100, 282)),
                            ((1000, 700), (13, 7, 333, 222)), ((1366, 768), (0, 0, 400, 300))):
            a = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
            x, y, w, h = box
            px, py = int(round(0.1 * w)) + 1, int(round(0.1 * h)) + 1
            b = a.copy()
            b[max(y - py, 0):y + h + py, max(x - px, 0):x + w + px] = 255 - b[max(y - py, 0):y + h + py,
                                                                            max(x - px, 0):x + w + px]
            da, db = cb.describe(Image.fromarray(a), box), cb.describe(Image.fromarray(b), box)
            for part in cb.PARTS:
                np.testing.assert_allclose(da[part], db[part], atol=1e-6, err_msg=f"{part} {W}x{H} {box}")

    def test_mask_covers_box_with_pad(self):
        _, bg = cb.small_frame(frame(), BOX)
        sx, sy = cb.SMALL[0] / 1920, cb.SMALL[1] / 1080
        x, y, w, h = BOX
        self.assertFalse(bg[int((y - 0.1 * h) * sy), int((x - 0.1 * w) * sx)])
        self.assertFalse(bg[int((y + 1.1 * h) * sy) - 1, int((x + 1.1 * w) * sx) - 1])
        self.assertTrue(bg[0, 0])
        self.assertTrue(bg[-1, -1])


class ShapeTests(unittest.TestCase):
    def test_parts_are_unit_norm_with_declared_size(self):
        d = cb.describe(frame(), BOX)
        for part, size in cb.PARTS.items():
            self.assertEqual(d[part].shape, (size,), part)
            self.assertAlmostEqual(float(np.linalg.norm(d[part])), 1.0, places=5)

    def test_non_divisible_frame_size(self):
        im = Image.fromarray(np.random.default_rng(0).integers(0, 256, (768, 1366, 3), dtype=np.uint8))
        d = cb.describe(im, (100, 100, 300, 200))
        for part, size in cb.PARTS.items():
            self.assertEqual(d[part].shape, (size,), part)

    def test_combine_weights_parts_equally(self):
        d = cb.describe(frame(), BOX)
        v = cb.combine([d["grad64"][None], d["tiny"][None]])[0]
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(v[:cb.PARTS["grad64"]])), 2 ** -0.5, places=5)

    def test_signature_changes_with_descriptor_settings(self):
        sig = cb.signature()
        old, cb.bgm.SMALL = cb.bgm.SMALL, (64, 36)
        try:
            self.assertNotEqual(sig, cb.signature())
        finally:
            cb.bgm.SMALL = old
        self.assertEqual(sig, cb.signature())

    def test_only_requested_parts(self):
        d = cb.describe(frame(), BOX, parts=("grad32",))
        self.assertEqual(set(d), {"grad32"})


if __name__ == "__main__":
    unittest.main()
