"""Быстрые smoke/regression-тесты исследовательских диагностик.

Запуск из корня репозитория:
    python -m unittest discover -s research/tests -v

Тесты не требуют PyTorch, Docker, датасета или GPU.
"""
import unittest

import numpy as np

from research.ablate_postprocess import ranking_metrics
from research.audit_generalization import audit
from research.calibrate_refusal import choose_threshold, metrics, top1_scores


class GeneralizationAuditTests(unittest.TestCase):
    def test_identity_disjoint_split_passes(self):
        source = [
            {"image_id": "a", "vehicle_id": "1", "camera_id": "1"},
            {"image_id": "b", "vehicle_id": "1", "camera_id": "2"},
            {"image_id": "c", "vehicle_id": "2", "camera_id": "1"},
            {"image_id": "d", "vehicle_id": "2", "camera_id": "2"},
        ]
        split = [
            {"image_id": "a", "vehicle_id": "1", "camera_id": "1", "role": "gallery"},
            {"image_id": "b", "vehicle_id": "1", "camera_id": "2", "role": "query"},
        ]

        report = audit(source, split, None, None)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["train"]["vehicle_id_overlap_with_holdout"], 0)
        self.assertEqual(report["holdout"]["usable_cross_camera_query"], 1)

    def test_identity_overlap_fails(self):
        source = [
            {"image_id": "a", "vehicle_id": "1", "camera_id": "1"},
            {"image_id": "b", "vehicle_id": "1", "camera_id": "2"},
        ]
        split = [
            {"image_id": "a", "vehicle_id": "1", "camera_id": "1", "role": "gallery"},
            {"image_id": "b", "vehicle_id": "1", "camera_id": "2", "role": "query"},
        ]

        report = audit(source, split, None, source)

        self.assertFalse(report["ok"])
        self.assertTrue(any("пересекаются" in item for item in report["failures"]))


class RankingMetricTests(unittest.TestCase):
    def test_map_uses_official_positive_normalization(self):
        scores = np.eye(3, dtype="float32")
        query = [
            {"image_id": "q1", "vehicle_id": "1", "camera_id": "1"},
            {"image_id": "q2", "vehicle_id": "2", "camera_id": "1"},
            {"image_id": "q3", "vehicle_id": "3", "camera_id": "1"},
        ]
        gallery = [
            {"image_id": "g1", "vehicle_id": "1", "camera_id": "2"},
            {"image_id": "g2", "vehicle_id": "2", "camera_id": "2"},
            {"image_id": "g3", "vehicle_id": "3", "camera_id": "2"},
        ]

        report = ranking_metrics(scores, query, gallery, top_k=10)

        self.assertEqual(report["mAP@10"], 1.0)
        self.assertEqual(report["Rank-1"], 1.0)
        self.assertEqual(report["Rank-5"], 1.0)


class RefusalCalibrationTests(unittest.TestCase):
    def test_calibration_honours_target_recall(self):
        known = np.array([0.95, 0.90, 0.85])
        unknown = np.array([0.30, 0.40, 0.50])

        selected = choose_threshold(known, unknown, target_recall=2 / 3)

        self.assertGreaterEqual(selected["recall"], 2 / 3)
        self.assertEqual(selected["TNR"], 1.0)

    def test_controlled_open_set_removes_unknown_vehicle(self):
        q = np.eye(4, dtype="float32")
        g = np.eye(4, dtype="float32")
        query = [{"vehicle_id": str(i), "camera_id": "1"} for i in range(4)]
        gallery = [{"vehicle_id": str(i), "camera_id": "2"} for i in range(4)]

        known, unknown = top1_scores(
            q, g, query, gallery, unknown_ids={"0", "1"}, cross_camera=False
        )

        self.assertEqual(len(known), 2)
        self.assertEqual(len(unknown), 2)
        self.assertTrue(np.all(known > unknown.max()))
        self.assertEqual(metrics(known, unknown, 1.0)["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
