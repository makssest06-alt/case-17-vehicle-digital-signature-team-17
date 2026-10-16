"""Тесты протокола research/honest_eval.py на игрушечных данных.

Запуск из корня репозитория:
    python -m unittest discover -s research/tests -v

Каждый тест закрепляет одно расхождение между старой валидацией и эталоном
организаторов. Нужны numpy и pandas (pandas — зависимость эталона).
"""
import unittest

import numpy as np

from research import honest_eval as he


def rows(spec):
    """[(image_id, vehicle_id, camera_id), ...] -> строки как в train.csv"""
    return [{"image_id": i, "vehicle_id": v, "camera_id": c} for i, v, c in spec]


class JunkInTopTenTests(unittest.TestCase):
    """Эталон чистит junk из присланных 10 позиций: почти-дубль занимает место."""

    def setUp(self):
        self.query = he.frame(rows([("q", "A", "1")]))
        gal = [("junk", "A", "1")] + [(f"n{k}", f"N{k}", "9") for k in range(9)] + [("pos", "A", "2")]
        self.gallery = he.frame(rows(gal))
        # junk выше всех, затем 9 чужих машин, верный кадр с другой камеры — 11-й
        self.sim = np.array([[0.99] + [0.9 - 0.01 * k for k in range(9)] + [0.5]])

    def test_official_loses_positive_behind_junk(self):
        m = he.evaluate(self.sim, self.sim, self.query, self.gallery, "official", threshold=0.0)
        self.assertEqual(m["mAP@10"], 0.0)
        self.assertEqual(m["Rank-1"], 0.0)

    def test_nodup_frees_the_slot(self):
        m = he.evaluate(self.sim, self.sim, self.query, self.gallery, "nodup", threshold=0.0)
        self.assertAlmostEqual(m["mAP@10"], 1 / 10)


class SameCameraNegativeTests(unittest.TestCase):
    """Чужая машина с той же камеры — негатив в эталоне и пропадает в wholecam."""

    def test_wholecam_hides_same_camera_distractor(self):
        query = he.frame(rows([("q", "A", "1")]))
        gallery = he.frame(rows([("other", "B", "1"), ("pos", "A", "2")]))
        sim = np.array([[0.9, 0.6]])
        off = he.evaluate(sim, sim, query, gallery, "official", threshold=0.0)
        whole = he.evaluate(sim, sim, query, gallery, "wholecam", threshold=0.0)
        self.assertEqual(off["Rank-1"], 0.0)
        self.assertEqual(whole["Rank-1"], 1.0)


class RefusalNearDuplicateTests(unittest.TestCase):
    """Почти-дубль даёт высокий top-1 и «верный» ответ, без него — отказ."""

    def test_near_duplicate_inflates_recall(self):
        query = he.frame(rows([("q", "A", "1")]))
        gallery = he.frame(rows([("dup", "A", "1"), ("pos", "A", "2"), ("neg", "B", "3")]))
        sim = np.array([[0.97, 0.70, 0.40]])
        off = he.evaluate(sim, sim, query, gallery, "official", threshold=0.8)
        clean = he.evaluate(sim, sim, query, gallery, "nodup", threshold=0.8)
        self.assertEqual((off["TP"], off["FN"]), (1, 0))
        self.assertEqual((clean["TP"], clean["FN"]), (0, 1))


class FullRankingParityTests(unittest.TestCase):
    def test_full_metrics_match_official(self):
        rng = np.random.default_rng(0)
        q = rng.normal(size=(6, 8)).astype("float32")
        g = rng.normal(size=(9, 8)).astype("float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        query = he.frame(rows([(f"q{k}", str(k % 3), str(k % 2)) for k in range(6)]))
        gallery = he.frame(rows([(f"g{k}", str(k % 3), str(k % 4)) for k in range(9)]))
        he.self_check(q, g, query, gallery)          # бросает AssertionError при расхождении


class SplitRuleTests(unittest.TestCase):
    def test_first_frame_per_camera_goes_to_gallery(self):
        train = rows([("b", "1", "1"), ("a", "1", "1"), ("c", "1", "2"), ("z", "2", "1")])
        split = he.holdout_split(train, {"1"})
        roles = {r["image_id"]: r["role"] for r in split}
        self.assertEqual(roles, {"a": "gallery", "b": "query", "c": "gallery"})


if __name__ == "__main__":
    unittest.main()
