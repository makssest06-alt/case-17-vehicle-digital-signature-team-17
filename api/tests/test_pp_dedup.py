"""Правило Андрея (app/pp/dedup.py): геометрия рамок и условие почти-дубля.

    python -m unittest tests.test_pp_dedup -v      # из каталога api/
"""
import unittest

import numpy as np

from app.pp import dedup as dd


class BoxFeaturesTest(unittest.TestCase):
    def test_same_box(self):
        f = dd.box_features([100, 200, 50, 40], [[100, 200, 50, 40]])
        self.assertAlmostEqual(float(f["iou"][0]), 1.0)
        self.assertAlmostEqual(float(f["center"][0]), 0.0)
        self.assertAlmostEqual(float(f["size"][0]), 0.0)

    def test_disjoint_and_shifted(self):
        f = dd.box_features([0, 0, 10, 10], [[20, 0, 10, 10], [5, 0, 10, 10]])
        self.assertEqual(float(f["iou"][0]), 0.0)
        self.assertAlmostEqual(float(f["iou"][1]), 50 / 150)
        self.assertAlmostEqual(float(f["center"][1]), 0.5)                 # сдвиг 5 при размере 10

    def test_size_ratio(self):
        f = dd.box_features([0, 0, 10, 10], [[0, 0, 20, 10]])
        self.assertAlmostEqual(float(f["size"][0]), np.log(2))

    def test_zero_area_box_is_never_near_dup(self):
        flags = dd.near_dup_flags([0.99, 0.99], [0, 0, 0, 0], [[0, 0, 0, 0], [0, 0, 10, 10]], 0.5, 0.0, 10, 10)
        self.assertEqual(flags.tolist(), [False, False])

    def test_rule_needs_similarity_and_box(self):
        boxes = [[100, 100, 50, 50], [100, 100, 50, 50], [400, 100, 50, 50]]
        flags = dd.near_dup_flags([0.95, 0.80, 0.95], [100, 100, 50, 50], boxes, 0.9, 0.7, 0.2, 0.2)
        self.assertEqual(flags.tolist(), [True, False, False])           # высокое сходство И та же рамка

    def test_matrix_shapes(self):
        q = np.array([[0, 0, 10, 10], [5, 5, 10, 10]], dtype=float)[:, None, :]
        b = np.tile(np.array([[0, 0, 10, 10]], dtype=float), (2, 3, 1))
        f = dd.box_features(q, b)
        self.assertEqual(f["iou"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
