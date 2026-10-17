"""Постобработка включена: DBA в отдельной колонке, штраф за ту же камеру, z100.

Поддельная база (service_fakes.FakeDB) хранит те же колонки, что app/db.py:
embedding, embedding_dba, bg. Запуск из каталога api/:
    python -m unittest discover -s tests -v
"""
import dataclasses
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import service_fakes as sf  # noqa: E402
from app.pp import background as bgm  # noqa: E402
from app.pp import rerank as rr  # noqa: E402
from app.pp import state as pp_state  # noqa: E402
from app.pp.dba import dba  # noqa: E402

BOX = {"x": "40", "y": "30", "w": "120", "h": "90"}


def cfg(**kw):
    from app.config import Settings
    base = {"pp_dba_k": 0, "pp_camera_penalty": "off", "refusal_score": "raw", "search_threshold": 0.0}
    return dataclasses.replace(Settings(), **{**base, **kw})


def stored_dba(db):
    return np.stack([r["embedding_dba"] for r in db.rows])


def all_dba(db, k=1):
    return dba(np.stack([r["embedding"] for r in db.rows]), k=k)


class Harness(unittest.TestCase):
    def start(self, **kw):
        self.db = sf.FakeDB()
        self.settings = cfg(**kw)
        self.main = sf.load_app()
        self.pp = pp_state.Postprocess(self.settings, sf.DIM, self.db)
        self.main.app.state.pp = self.pp
        patches = [patch.object(self.main, "settings", self.settings), patch.object(self.main.db, "count", self.db.count),
                   patch.object(self.main.db, "search_pp", self.db.search_pp)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(setattr, self.main.app.state, "pp", None)

    def post(self, path, seed, fields=None):
        r = sf.call(self.main.app, "POST", path, fields={**BOX, **(fields or {})},
                    files={"image": (f"f{seed}.png", sf.png(seed))})
        return r["status"], json.loads(r["body"])


class DbaTest(Harness):
    def test_startup_migrates_and_fills_dba_column(self):
        self.start(pp_dba_k=1)
        self.assertEqual(self.db.schema, [("ensure_pp_schema", sf.DIM, bgm.BG_DIM)])
        np.testing.assert_allclose(stored_dba(self.db), all_dba(self.db), atol=1e-6)
        before = [r["embedding"].copy() for r in self.db.rows]
        self.db.calls.clear()
        self.pp.reload()                                           # повторный старт ничего не переписывает
        self.assertEqual(self.db.calls, [])
        for r, b in zip(self.db.rows, before):
            np.testing.assert_array_equal(r["embedding"], b)        # исходные векторы не тронуты

    def test_gallery_add_keeps_dba_column_exact(self):
        self.start(pp_dba_k=1)
        status, body = self.post("/v1/gallery", 5, {"vehicle_id": "555"})
        self.assertEqual(status, 201)
        self.assertEqual(body["id"], self.db.rows[-1]["id"])
        np.testing.assert_allclose(stored_dba(self.db), all_dba(self.db), atol=1e-6)

    def test_add_updates_neighbour_rows(self):
        self.start(pp_dba_k=1)
        near = self.db.rows[4]["embedding"] + np.float32(1e-3)       # почти копия кадра id=5: станет его соседом
        self.db.calls.clear()
        self.pp.add_many([(None, "near.jpg", [1, 2, 3, 4], near / np.linalg.norm(near), None)])
        upd = next(c for c in self.db.calls if c[0] == "update_dba")
        self.assertIn(5, upd[1])
        np.testing.assert_allclose(stored_dba(self.db), all_dba(self.db), atol=1e-6)

    def test_delete_updates_neighbours(self):
        self.start(pp_dba_k=1)
        r = sf.call(self.main.app, "DELETE", "/v1/gallery/3")
        self.assertEqual(r["status"], 200)
        self.assertNotIn(3, [x["id"] for x in self.db.rows])
        np.testing.assert_allclose(stored_dba(self.db), all_dba(self.db), atol=1e-6)
        self.assertEqual(sf.call(self.main.app, "DELETE", "/v1/gallery/999")["status"], 404)

    def test_search_uses_dba_column(self):
        self.start(pp_dba_k=1)
        status, body = self.post("/v1/search", 1)
        self.assertEqual(status, 200)
        call = next(c for c in self.db.calls if c[0] == "search_pp")
        self.assertEqual(call[1:], (10, "embedding_dba", 100))      # ef_search = max(100, PP_CAMERA_TOPK)
        q = sf.FakeModel().embed_batch([Image.open(io.BytesIO(sf.png(1))).crop((28, 21, 172, 129))])[0]
        g = all_dba(self.db)
        want = [self.db.rows[i]["id"] for i in np.argsort(-(g @ q))[:10]]
        self.assertEqual([c["id"] for c in body["candidates"]], want)

    def test_insert_failure_resyncs_from_db(self):
        self.start(pp_dba_k=1)
        n = len(self.db.rows)
        with patch.object(self.db, "insert_pp", side_effect=RuntimeError("база недоступна")):
            with self.assertRaises(RuntimeError):
                self.pp.add_many([(None, "x", [1, 2, 3, 4], np.ones(sf.DIM, "float32"), None)])
        self.assertEqual(len(self.pp.ids), n)
        self.assertEqual(len(self.pp.dba), n)


class CameraTest(Harness):
    def frame_bg(self, seed):
        im = Image.open(io.BytesIO(sf.png(seed))).convert("RGB")
        return bgm.service_descriptor(im, (40, 30, 120, 90))

    def test_same_camera_candidate_goes_last(self):
        self.start(pp_camera_penalty="on")
        _, plain = self.post("/v1/search", 1)
        top = plain["candidates"][0]["id"]
        row = next(r for r in self.db.rows if r["id"] == top)
        row["bg"] = self.frame_bg(1)                                # тот же фон, что у запроса
        self.pp.reload()
        _, body = self.post("/v1/search", 1)
        ids = [c["id"] for c in body["candidates"]]
        self.assertNotEqual(ids[0], top)
        self.assertEqual(ids[:9], [c["id"] for c in plain["candidates"][1:10]])
        call = next(c for c in self.db.calls if c[0] == "search_pp")
        self.assertEqual(call[1:], (100, "embedding", 100))         # штраф смотрит top-100

    def test_refusal_uses_best_unflagged(self):
        self.start(pp_camera_penalty="on")
        _, plain = self.post("/v1/search", 1)
        first, second = plain["candidates"][0], plain["candidates"][1]
        next(r for r in self.db.rows if r["id"] == first["id"])["bg"] = self.frame_bg(1)
        self.pp.reload()
        thr = (first["confidence"] + second["confidence"]) / 2
        with patch.object(self.main, "settings", dataclasses.replace(self.settings, search_threshold=thr)):
            _, body = self.post("/v1/search", 1)
        self.assertTrue(body["refusal"])                            # лучший без флага ниже порога
        self.assertIn(f"{second['confidence']:.3f}", body["message"])

    def test_gallery_add_stores_background(self):
        self.start(pp_camera_penalty="store")
        self.post("/v1/gallery", 7)
        np.testing.assert_allclose(self.db.rows[-1]["bg"], self.frame_bg(7), atol=1e-6)
        self.assertEqual(self.pp.bg, {})                            # store: хранить, но не штрафовать
        _, body = self.post("/v1/search", 7)
        self.assertEqual(next(c for c in self.db.calls if c[0] == "search_pp")[1:], (10, "embedding", 100))

    def test_bulk_import_crops_have_no_background(self):
        self.start(pp_camera_penalty="on")
        with sf.data_dir() as root, patch.object(self.main, "settings",
                                                 dataclasses.replace(self.settings, data_dir=str(root))):
            r = sf.call(self.main.app, "POST", "/v1/gallery/bulk-import",
                        json_body={"images_dir": str(root / "crops"), "csv_path": str(root / "g.csv")})
        self.assertEqual(json.loads(r["body"])["imported"], 3)
        self.assertTrue(all(x["bg"] is None for x in self.db.rows[-3:]))


class DedupTest(Harness):
    """PP_DEDUP=bbox — правило Андрея: сырой косинус ≥ s_hi И та же рамка."""

    def query_vec(self, seed=1):
        im = Image.open(io.BytesIO(sf.png(seed))).convert("RGB")
        return sf.FakeModel().embed_batch([im.crop((28, 21, 172, 129))])[0]

    def make_twin(self, box):
        row = self.db.rows[7]
        row["embedding"], row["bbox"] = self.query_vec().copy(), list(box)    # тот же кадр машины: cos = 1
        self.pp.reload()
        return row["id"]

    def test_near_duplicate_goes_last(self):
        self.start(pp_dedup="bbox")
        twin = self.make_twin((40, 30, 120, 90))
        _, body = self.post("/v1/search", 1)
        ids = [c["id"] for c in body["candidates"]]
        self.assertNotIn(twin, ids)                                          # 30 кадров, top_n 10: ушёл в конец
        self.assertEqual(next(c for c in self.db.calls if c[0] == "search_pp")[1:], (100, "embedding", 100))

    def test_far_box_is_not_near_duplicate(self):
        self.start(pp_dedup="bbox")
        twin = self.make_twin((300, 150, 20, 20))
        _, body = self.post("/v1/search", 1)
        self.assertEqual(body["candidates"][0]["id"], twin)

    def test_with_dba_rule_uses_raw_cosine(self):
        self.start(pp_dba_k=1, pp_dedup="bbox")
        twin = self.make_twin((40, 30, 120, 90))
        q = self.query_vec()
        raw = self.pp._raw_cos(q, [{"id": twin}], np.array([0.0]))
        self.assertAlmostEqual(float(raw[0]), 1.0, places=5)                 # не DBA-косинус
        _, body = self.post("/v1/search", 1)
        self.assertNotIn(twin, [c["id"] for c in body["candidates"]])

    def test_camera_penalty_on_means_dedup_bg(self):
        self.start(pp_camera_penalty="on")
        self.assertEqual((self.pp.dedup_rank, self.pp.dedup_refusal, self.pp.use_bg, self.pp.use_bbox),
                         ("bg", "bg", True, False))

    def test_refusal_only_dedup_keeps_ranking(self):
        """Правило рамок только для отказа: почти-дубль остаётся первым в ответе, но не в оценке."""
        self.start(pp_dedup_rank="off", pp_dedup_refusal="bbox", refusal_score="raw")
        twin = self.make_twin((40, 30, 120, 90))
        found, score = self.pp.search(self.query_vec(), 10, None, (40, 30, 120, 90))
        self.assertEqual(found[0]["id"], twin)                               # ранжирование не тронуто
        self.assertAlmostEqual(found[0]["confidence"], 1.0, places=4)
        self.assertAlmostEqual(score, found[1]["confidence"], places=4)      # оценка — по следующему

    def test_andrey_rule_similarity_alone(self):
        """cos ≥ s_any — почти-дубль и без рамки (правило Андрея: 0.94 или 0.85 + IoU 0.7)."""
        self.start(pp_dedup="bbox", pp_dedup_sim_any=0.94, pp_dedup_sim=0.85, pp_dedup_iou=0.7,
                   pp_dedup_center=float("inf"))
        twin = self.make_twin((300, 150, 20, 20))                            # рамка далеко, cos = 1
        _, body = self.post("/v1/search", 1)
        self.assertNotIn(twin, [c["id"] for c in body["candidates"]])


class RefusalScoreTest(unittest.TestCase):
    def test_raw_and_z100(self):
        cos = np.array([0.9, 0.8, 0.5, 0.4, 0.3])
        flags = np.array([True, False, False, False, False])
        self.assertEqual(rr.refusal_score(cos, flags, "raw"), 0.8)
        rest = np.array([0.5, 0.4, 0.3])
        self.assertAlmostEqual(rr.refusal_score(cos, flags, "z100"), (0.8 - rest.mean()) / rest.std())
        self.assertEqual(rr.refusal_score(cos, np.ones(5, bool), "raw"), float("-inf"))
        self.assertEqual(rr.refusal_score(cos[:1], flags[:1] * False, "z100"), float("-inf"))

    def test_z100_matches_research_definition(self):
        top = np.sort(np.random.default_rng(0).random(100))[::-1]
        bg = top[1:]
        want = (top[0] - bg.mean()) / max(bg.std(), 1e-6)            # research/refusal_score.z_of
        self.assertAlmostEqual(rr.refusal_score(top, np.zeros(100, bool), "z100"), want)

    def test_camera_flags_match_per_candidate_dot(self):
        rng = np.random.default_rng(1)
        q = bgm.l2(rng.standard_normal(bgm.BG_DIM).astype("float32"))
        cands = [None if i % 7 == 0 else bgm.l2((q + rng.standard_normal(bgm.BG_DIM) * s).astype("float32"))
                 .astype("float16") for i, s in enumerate(np.linspace(0.005, 0.05, 60))]
        want = [b is not None and float(np.dot(b.astype("float32"), q)) > 0.68127 for b in cands]
        got = rr.camera_flags(q, cands, 0.68127)
        self.assertEqual(got.tolist(), want)
        self.assertTrue(0 < got.sum() < 60 - 9)                    # есть и флаги, и не-флаги
        self.assertEqual(rr.camera_flags(None, cands, 0.5).tolist(), [False] * 60)
        self.assertEqual(rr.camera_flags(q, [None, None], 0.5).tolist(), [False, False])

    def test_rank_moves_flagged_last_keeping_order(self):
        order = rr.rank(np.array([0.9, 0.7, 0.8, 0.7]), np.array([True, False, False, False]))
        self.assertEqual(order.tolist(), [2, 1, 3, 0])

    def test_z100_search_without_postprocess(self):
        db = sf.FakeDB()
        main = sf.load_app()
        main.app.state.pp = None
        s = cfg(refusal_score="z100", search_threshold=1.0)
        with patch.object(main, "settings", s), patch.object(main.db, "count", db.count), \
                patch.object(main.db, "search_pp", db.search_pp):
            r = sf.call(main.app, "POST", "/v1/search", fields=BOX, files={"image": ("f1.png", sf.png(1))})
        body = json.loads(r["body"])
        self.assertEqual(len(body["candidates"] or []) <= 10, True)
        self.assertEqual(next(c for c in db.calls if c[0] == "search_pp")[1:], (100, "embedding", 100))


class ValidationTest(unittest.TestCase):
    def test_bad_flags_rejected(self):
        main = sf.load_app()
        for kw in ({"pp_camera_penalty": "yes"}, {"pp_dba_k": -1}, {"refusal_score": "softmax"},
                   {"refusal_score": "raw", "search_threshold": 3.0}, {"pp_dedup": "iou"},
                   {"pp_camera_penalty": "on", "pp_dedup": "bbox"}, {"pp_dedup_iou": 1.5},
                   {"pp_dedup_refusal": "rect"}, {"pp_dedup_rank": "all"}):
            with self.subTest(kw), patch.object(main, "settings", cfg(**{"search_threshold": 0.8, **kw})):
                with self.assertRaises(RuntimeError):
                    main._validate_runtime_config()

    def test_z100_allows_threshold_above_one(self):
        main = sf.load_app()
        with patch.object(main, "settings", cfg(refusal_score="z100", search_threshold=3.5,
                                               search_threshold_strict=4.0)):
            main._validate_runtime_config()


if __name__ == "__main__":
    unittest.main()
