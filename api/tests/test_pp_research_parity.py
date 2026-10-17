"""Сервис и исследование на одних данных: top-K, флаги исключения, z100 и raw совпадают.

Сервис — Postprocess.search (DBA k=1 + исключение почти-дублей bg / bbox / both + z100)
на поддельной базе и, если задан
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


ANDREY = {"pp_dedup_sim_any": 0.94, "pp_dedup_sim": 0.85, "pp_dedup_iou": 0.7, "pp_dedup_center": float("inf")}
# (ранжирование, отказ, пороги правила рамок); последняя — финальная схема: ранжирование bg, отказ bg OR рамка
MODES = (("bg", "bg", {}), ("bbox", "bbox", {}), ("both", "both", {}), ("bg", "both", {}), ("bg", "both", ANDREY))


def settings(rank="bg", refusal="", rule=None):
    from app.config import Settings
    return dataclasses.replace(Settings(), pp_dba_k=1, pp_dedup_rank=rank, pp_dedup_refusal=refusal,
                               refusal_score="z100", **(rule or {}))


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
    gbox = np.column_stack([rng.integers(0, 1500, N), rng.integers(0, 800, N), rng.integers(80, 400, N),
                            rng.integers(60, 300, N)]).astype("float64")                 # рамки далеко от BOX
    gbox[np.abs(gbox[:, 0] - BOX[0]) < 300, 0] += 600
    for i in range(3):                                                            # «та же камера» у части top
        near = np.argsort(-(g @ q[i]))[: 40: 4]
        gbg[near] = bgm.l2(qbg[i] + 0.01 * rng.standard_normal(bgm.BG_DIM).astype("float32"))
        twins = np.argsort(-(g @ q[i]))[1: 30: 7]                                   # почти-дубли: сходство + рамка
        g[twins] = q[i] + 0.02 * rng.standard_normal((len(twins), D))
        g[twins] /= np.linalg.norm(g[twins], axis=1, keepdims=True)
        gbox[twins] = BOX
    return g, gbg, gbox, q, qbg, frames


def research_side(g, gbg, gbox, q, qbg, tau, rule, rank, refusal):
    sys.path.insert(0, str(RESEARCH))
    import service_score as ss
    from app.pp import dedup as dd

    s = q @ ss.dba_gallery(g, 1).T
    top, cos = ss.topk(s, K)
    fb = ss.pair_cos(qbg, gbg, top) > tau
    raw = np.take_along_axis(q @ g.T, top, 1)
    fx = dd.near_dup_flags(raw, np.array(BOX, float)[None, None, :], gbox[top], **rule)
    pick = {"bg": fb, "bbox": fx, "both": fb | fx}
    return ss.decide(top, cos, pick[rank], pick[refusal])


@unittest.skipUnless((RESEARCH / "service_score.py").exists(), "нет research/service_score.py")
class ParityTest(unittest.TestCase):
    def setUp(self):
        self.g, self.gbg, self.gbox, self.q, self.qbg, self.frames = data()
        self.tau = settings().pp_camera_tau
        cos = self.qbg @ self.gbg.T
        self.assertTrue((np.abs(cos - self.tau) > 0.05).all(), "признак фона слишком близко к τ")
        self.ref = {}
        for n, (rank, refusal, rule) in enumerate(MODES):
            st = settings(rank, refusal, rule)
            r = {"s_any": st.pp_dedup_sim_any, "s_hi": st.pp_dedup_sim, "t_iou": st.pp_dedup_iou,
                 "t_center": st.pp_dedup_center, "t_size": st.pp_dedup_size}
            self.ref[n] = research_side(self.g, self.gbg, self.gbox, self.q, self.qbg, self.tau, r, rank, refusal)
            self.assertTrue(any(f.any() for f in self.ref[n]["flags"]), MODES[n])   # исключение реально срабатывает
        self.assertNotEqual([list(o) for o in self.ref[0]["order"]], [list(o) for o in self.ref[1]["order"]])
        self.assertFalse(np.allclose(self.ref[0]["conf_z100"], self.ref[3]["conf_z100"]))   # отказ ≠ ранжирование

    def check(self, store, ids):
        for n, (rank, refusal, rule) in enumerate(MODES):
            pp = pp_state.Postprocess(settings(rank, refusal, rule), D, store)
            for i in range(3):
                found, score = pp.search(self.q[i], K, self.frames[i], BOX)
                with self.subTest(rank=rank, refusal=refusal, andrey=bool(rule), query=i):
                    self.assertEqual([c["id"] for c in found], [ids[j] for j in self.ref[n]["order"][i]])
                    self.assertAlmostEqual(score, self.ref[n]["conf_z100"][i], places=5)

    def test_fake_db(self):
        db = sf.FakeDB(n=0)
        for i in range(N):
            db._insert(None, f"g{i}.jpg", self.gbox[i], self.g[i])
            db.rows[-1]["bg"] = self.gbg[i]
        self.check(db, [r["id"] for r in db.rows])

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
            ids = db.insert_pp([(None, f"g{i}.jpg", [int(v) for v in self.gbox[i]], self.g[i], None, self.gbg[i])
                                for i in range(N)], [])
            self.check(db, ids)
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
