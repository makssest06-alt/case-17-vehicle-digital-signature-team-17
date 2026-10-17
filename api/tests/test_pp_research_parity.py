"""Сервис и исследование на одних данных: top-K, флаги «та же камера», z100 и raw совпадают.

Сервис — Postprocess.search (DBA k=1 + штраф + z100) на поддельной базе и, если задан
PP_TEST_DSN, на настоящем pgvector через HNSW (hnsw.ef_search = max(100, PP_CAMERA_TOPK)).
Исследование — research/service_score.rank_queries: точный top-K по numpy, тем же
кодом app.pp. Признаки фона галереи выбраны далеко от τ, чтобы float16 у сервиса не
мог перевернуть флаг: тест проверяет логику, а не округление.

    python -m unittest tests.test_pp_research_parity -v     # из каталога api/, research/ рядом
"""
import dataclasses
import io
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1] / "research"
sys.path.insert(0, str(HERE))

import service_fakes as sf  # noqa: E402
from app.pp import background as bgm  # noqa: E402
from app.pp import state as pp_state  # noqa: E402

N, D, K = 300, 16, 100
BOX = (40, 30, 120, 90)


def settings():
    from app.config import Settings
    return dataclasses.replace(Settings(), pp_dba_k=1, pp_camera_penalty="on", refusal_score="z100")


def data():
    rng = np.random.default_rng(7)
    centers = rng.standard_normal((25, D))
    g = centers[rng.integers(0, 25, N)] + 0.35 * rng.standard_normal((N, D))
    g = (g / np.linalg.norm(g, axis=1, keepdims=True)).astype("float32")
    frames = [Image.open(io.BytesIO(sf.png(s))).convert("RGB") for s in (11, 12, 13)]
    qbg = np.stack([bgm.service_descriptor(f, BOX) for f in frames])
    gbg = bgm.l2(rng.standard_normal((N, bgm.BG_DIM)).astype("float32"))          # косинус с запросами ~0
    q = centers[[3, 8, 17]] + 0.2 * rng.standard_normal((3, D))
    q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype("float32")
    for i in range(3):                                                            # «та же камера» у части top
        near = np.argsort(-(g @ q[i]))[: 40: 4]
        gbg[near] = bgm.l2(qbg[i] + 0.01 * rng.standard_normal(bgm.BG_DIM).astype("float32"))
    return g, gbg, q, qbg, frames


def research_side(g, gbg, q, qbg, tau):
    sys.path.insert(0, str(RESEARCH))
    import service_score as ss

    s = q @ ss.dba_gallery(g, 1).T
    return ss.rank_queries(s, qbg, gbg, K, tau, True)


@unittest.skipUnless((RESEARCH / "service_score.py").exists(), "нет research/service_score.py")
class ParityTest(unittest.TestCase):
    def setUp(self):
        self.g, self.gbg, self.q, self.qbg, self.frames = data()
        self.tau = settings().pp_camera_tau
        cos = self.qbg @ self.gbg.T
        self.assertTrue((np.abs(cos - self.tau) > 0.05).all(), "признак фона слишком близко к τ")
        self.ref = research_side(self.g, self.gbg, self.q, self.qbg, self.tau)
        self.assertTrue(any(f.any() for f in self.ref["flags"]))                      # штраф реально срабатывает

    def check(self, pp, ids):
        for i in range(3):
            found, score = pp.search(self.q[i], K, self.frames[i], BOX)
            with self.subTest(query=i):
                self.assertEqual([c["id"] for c in found], [ids[j] for j in self.ref["order"][i]])
                self.assertAlmostEqual(score, self.ref["conf_z100"][i], places=5)

    def test_fake_db(self):
        db = sf.FakeDB(n=0)
        for i in range(N):
            db._insert(None, f"g{i}.jpg", [1, 2, 3, 4], self.g[i])
            db.rows[-1]["bg"] = self.gbg[i]
        pp = pp_state.Postprocess(settings(), D, db)
        self.check(pp, [r["id"] for r in db.rows])

    @unittest.skipUnless(os.getenv("PP_TEST_DSN"), "нет PP_TEST_DSN")
    def test_pgvector_hnsw(self):
        import psycopg

        from app import db
        schema = "pp_parity"
        old = db.settings
        with psycopg.connect(os.environ["PP_TEST_DSN"], autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.execute(f"CREATE SCHEMA {schema}")
        # enable_seqscan=off: поиск обязан идти через HNSW, как на большой галерее
        db.settings = dataclasses.replace(old, db_dsn=f"{os.environ['PP_TEST_DSN']}?options=-csearch_path%3D{schema}"
                                                      ",public%20-cenable_seqscan%3Doff")
        db.init_pool()
        try:
            db.ensure_schema(D)
            db.ensure_pp_schema(D, bgm.BG_DIM)
            ids = db.insert_pp([(None, f"g{i}.jpg", [1, 2, 3, 4], self.g[i], None, self.gbg[i]) for i in range(N)], [])
            pp = pp_state.Postprocess(settings(), D, db)
            self.check(pp, ids)
            with db.pool().connection() as conn, conn.cursor() as cur:            # SET LOCAL не утёк в пул
                cur.execute("SHOW hnsw.ef_search")
                self.assertEqual(cur.fetchone()[0], "40")
        finally:
            with db.pool().connection() as conn, conn.cursor() as cur:
                cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                conn.commit()
            db.close_pool()
            db.settings = old


if __name__ == "__main__":
    unittest.main()
