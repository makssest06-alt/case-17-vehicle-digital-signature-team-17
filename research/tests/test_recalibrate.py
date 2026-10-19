"""research/recalibrate_for_model.py: признак фона только для train.csv — те же векторы, что у service_bg,
кадры теста не читаются.

Запуск из корня репозитория:
    python -m unittest discover -s research/tests -v
"""
import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research import cheap_bg as cb
from research import recalibrate_for_model as rm

HEAD = "image_id,x,y,w,h,vehicle_id,camera_id\n"


class TrainBgTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        d = Path(self.dir.name)
        (d / "train.csv").write_text(HEAD + "t1,1,1,5,5,7,3\nt2,2,2,5,5,8,4\n", encoding="utf-8")
        (d / "test_query.csv").write_text("image_id,x,y,w,h\nq1,1,1,5,5\n", encoding="utf-8")
        (d / "test_gallery.csv").write_text("image_id,x,y,w,h\ng1,1,1,5,5\n", encoding="utf-8")
        self.a = argparse.Namespace(data_dir=d, cache_dir=d, zip=None)
        self.ids = ["t1", "t2", "q1", "g1"]                       # общий кэш — все строки CSV, как service_bg
        rng = np.random.default_rng(0)
        self.parts = {p: rng.standard_normal((4, 6)).astype("float32") for p in cb.PARTS}
        self.kind = "+".join(cb.bgm.SERVICE_PARTS)

    def test_subset_of_shared_cache_equals_service_bg(self):
        np.savez(Path(self.dir.name) / "cheap_bg_desc.npz", ids=np.array(self.ids), sig=np.array(cb.signature()),
                 **self.parts)
        got = rm.train_bg(self.a)
        full = cb.kind_bg(self.parts, self.ids, self.kind)            # то, что вернул бы tvg.service_bg
        self.assertEqual(sorted(got), ["t1", "t2"])
        for i in got:
            self.assertTrue(np.array_equal(got[i], full[i]))

    def test_without_cache_and_zip_asks_for_zip(self):
        with self.assertRaises(SystemExit) as cm:
            rm.train_bg(self.a)
        self.assertIn("--zip", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
