"""SQL постобработки на настоящем Postgres + pgvector. Без PP_TEST_DSN — пропускается.

    docker run -d --name pp-test-db -e POSTGRES_PASSWORD=t -p 55432:5432 pgvector/pgvector:pg16
    PP_TEST_DSN=postgresql://postgres:t@localhost:55432/postgres python -m unittest tests.test_pp_db -v

Всё — во временной схеме pp_test (search_path), которая удаляется в конце.
"""
import dataclasses
import os
import unittest

import numpy as np

DSN = os.getenv("PP_TEST_DSN")
SCHEMA = "pp_test"
DIM, BG = 8, 16


@unittest.skipUnless(DSN, "нет PP_TEST_DSN")
class PostprocessSqlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import config, db

        import psycopg

        cls.db, cls.config = db, config
        cls.old = db.settings
        with psycopg.connect(DSN, autocommit=True) as conn:       # пул регистрирует тип vector — расширение нужно до него
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            conn.execute(f"CREATE SCHEMA {SCHEMA}")
        # enable_seqscan=off: на 150 строках планировщик иначе идёт мимо HNSW, и проверка
        # «top-100 — это 100 строк» ничего не доказывает (HNSW по умолчанию отдаёт ≤ 40)
        db.settings = dataclasses.replace(
            db.settings, db_dsn=f"{DSN}?options=-csearch_path%3D{SCHEMA},public%20-cenable_seqscan%3Doff")
        db.init_pool()
        db.ensure_schema(DIM)
        rng = np.random.default_rng(0)
        v = rng.standard_normal((150, DIM)).astype("float32")
        cls.vecs = v / np.linalg.norm(v, axis=1, keepdims=True)
        db.insert_many([(str(i % 9), f"g{i}.jpg", [1, 2, 3, 4], cls.vecs[i]) for i in range(150)])   # старые строки

    @classmethod
    def tearDownClass(cls):
        with cls.db.pool().connection() as conn, conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            conn.commit()
        cls.db.close_pool()
        cls.db.settings = cls.old

    def test_1_migration_is_additive_and_idempotent(self):
        db = self.db
        before = db.search(self.vecs[0], 10)
        db.ensure_pp_schema(DIM, BG)
        db.ensure_pp_schema(DIM, BG)
        self.assertEqual(db.search(self.vecs[0], 10), before)              # старый поиск не изменился
        rows = db.fetch_pp(True)
        self.assertEqual(len(rows), 150)
        self.assertTrue(all(r["embedding_dba"] is None and r["bg"] is None for r in rows))

    def test_2_insert_update_search_delete(self):
        db = self.db
        db.ensure_pp_schema(DIM, BG)
        ids = [r["id"] for r in db.fetch_pp(False)]
        db.update_dba([(i, self.vecs[k]) for k, i in enumerate(ids)])
        bg = np.ones(BG, "float32") / 4
        new = db.insert_pp([("x", "n.jpg", [5, 6, 7, 8], self.vecs[1], self.vecs[1], bg)], [(ids[0], self.vecs[2])])
        rows = {r["id"]: r for r in db.fetch_pp(True)}
        np.testing.assert_allclose(rows[new[0]]["bg"], bg, atol=1e-7)
        np.testing.assert_allclose(rows[ids[0]]["embedding_dba"], self.vecs[2], atol=1e-6)
        got = db.search_pp(self.vecs[3], 100, "embedding_dba", 100)
        self.assertEqual(len(got), 100)                                     # HNSW по умолчанию отдал бы ≤ 40
        self.assertLess(len(db.search(self.vecs[3], 100)), 100)            # старый поиск упирается в ef_search (~40)
        cos = [g["confidence"] for g in got]
        self.assertEqual(cos, sorted(cos, reverse=True))
        self.assertTrue(db.delete_pp(new[0], [(ids[1], self.vecs[4])]))
        self.assertNotIn(new[0], {r["id"] for r in db.fetch_pp(False)})
        with self.assertRaises(ValueError):
            db.search_pp(self.vecs[0], 5, "bg; DROP TABLE gallery", 100)
        with self.assertRaises(ValueError):
            db.search_pp(self.vecs[0], 5, "embedding", 0)


if __name__ == "__main__":
    unittest.main()
