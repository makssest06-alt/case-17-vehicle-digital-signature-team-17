"""research/a5000_compare_submission.py: два прогона make_submission — sha256, тот же top-1, принятые запросы,
confidence, векторы — на синтетических файлах.

Запуск из корня репозитория:
    python -m unittest discover -s research/tests -v
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research import a5000_compare_submission as cs


def write(d: Path, top1: dict, cand: list, emb: np.ndarray) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "submission.csv").write_text("query_id,gallery_id_1\n" + "".join(f"{q},{g}\n" for q, g in top1.items()),
                                      encoding="utf-8")
    (d / "candidates.csv").write_text("query_id,gallery_id,confidence\n" + "".join(f"{q},{g},{c}\n" for q, g, c in cand),
                                      encoding="utf-8")
    ids = list(top1) + ["g1", "g2"]
    (d / "embeddings_index.csv").write_text("row,split,image_id\n" + "".join(f"{i},x,{v}\n" for i, v in enumerate(ids)),
                                            encoding="utf-8")
    np.save(d / "embeddings.npy", emb.astype("float32"))


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        rng = np.random.default_rng(0)
        self.emb = rng.standard_normal((5, 8))
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)
        self.top1 = {"q1": "g1", "q2": "g2", "q3": "g1"}
        self.cand = [("q1", "g1", 0.9), ("q1", "g2", 0.5), ("q2", "g2", 0.8)]
        write(self.dir / "repo", self.top1, self.cand, self.emb)

    def test_identical_runs(self):
        shutil.copytree(self.dir / "repo", self.dir / "same")
        r = cs.compare(self.dir / "repo", self.dir / "same")
        self.assertEqual(r["same_files"], list(cs.FILES))
        self.assertEqual((r["top1"]["share"], r["accepted"]["same_set"], r["confidence"]["max_abs_diff"]), (1.0, True, 0.0))
        self.assertEqual(r["embeddings"]["max_abs_diff"], 0.0)

    def test_differences_are_counted(self):
        noisy = self.emb + 1e-3 * np.random.default_rng(1).standard_normal(self.emb.shape)
        top1 = {**self.top1, "q3": "g2"}                                  # другой top-1 у одного запроса
        cand = [("q1", "g1", 0.9003), ("q1", "g2", 0.5), ("q3", "g2", 0.77)]   # q2 стал отказом, q3 — принятым
        write(self.dir / "gpu", top1, cand, noisy)
        r = cs.compare(self.dir / "repo", self.dir / "gpu")
        self.assertEqual(r["same_files"], [])
        self.assertEqual((r["top1"]["same"], r["top1"]["of"]), (2, 3))
        a = r["accepted"]
        self.assertEqual((a["repo"], a["other"], a["refused_to_accepted"], a["accepted_to_refused"], a["same_set"]),
                         (2, 2, 1, 1, False))
        self.assertAlmostEqual(r["confidence"]["max_abs_diff"], 0.0003, places=6)
        self.assertEqual(r["confidence"]["pairs_common"], 2)
        self.assertLess(r["embeddings"]["min_cos"], 1.0)
        self.assertIn("отказ → принят 1, принят → отказ 1", r["summary"])

    def test_different_queries_rejected(self):
        write(self.dir / "bad", {"q1": "g1"}, self.cand, self.emb)
        with self.assertRaises(SystemExit):
            cs.compare(self.dir / "repo", self.dir / "bad")


if __name__ == "__main__":
    unittest.main()
