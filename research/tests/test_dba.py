"""Тесты research/dba.py: DBA галереи одним вызовом и с состоянием для /v1/gallery.

Запуск из корня репозитория:
    python -m unittest discover -s research/tests -v
"""
import unittest

import numpy as np

from research import dba


def unit(rows):
    x = np.asarray(rows, dtype="float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def random_gallery(n, d=16, seed=0):
    return unit(np.random.default_rng(seed).standard_normal((n, d)))


class DbaFormulaTests(unittest.TestCase):
    """g'_i = L2(g_i + Σ max(cos, 0)·g_j) по k ближайшим ДРУГИМ кадрам."""

    def test_k1_adds_nearest_other_frame_weighted_by_cosine(self):
        g = unit([[1, 0], [0.8, 0.6], [-1, 0]])
        out = dba.dba(g, k=1)
        cos01 = float(g[0] @ g[1])
        want0 = g[0] + cos01 * g[1]
        np.testing.assert_allclose(out[0], want0 / np.linalg.norm(want0), atol=1e-6)

    def test_negative_cosine_neighbour_gets_zero_weight(self):
        g = unit([[1, 0], [-1, 0.1]])
        out = dba.dba(g, k=1)
        np.testing.assert_allclose(out, g, atol=1e-6)

    def test_uniform_weights_ignore_cosine(self):
        g = unit([[1, 0], [0.8, 0.6], [-1, 0]])
        out = dba.dba(g, k=1, weights="uniform")
        want0 = g[0] + g[1]
        np.testing.assert_allclose(out[0], want0 / np.linalg.norm(want0), atol=1e-6)

    def test_output_is_unit_norm_and_new_array(self):
        g = random_gallery(50)
        out = dba.dba(g, k=2)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        self.assertIsNot(out, g)
        np.testing.assert_array_equal(g, random_gallery(50))       # вход не тронут

    def test_k0_and_single_frame_return_input(self):
        g = random_gallery(5)
        np.testing.assert_allclose(dba.dba(g, k=0), g, atol=1e-6)
        np.testing.assert_allclose(dba.dba(g[:1], k=3), g[:1], atol=1e-6)

    def test_k_larger_than_gallery_uses_all_other_frames(self):
        g = random_gallery(4)
        np.testing.assert_allclose(dba.dba(g, k=10), dba.dba(g, k=3), atol=1e-6)

    def test_unknown_weights_rejected(self):
        with self.assertRaises(ValueError):
            dba.dba(random_gallery(3), k=1, weights="softmax")

    def test_neighbours_exclude_self_and_are_sorted(self):
        g = random_gallery(30)
        idx, sim = dba.neighbours(g, 4)
        self.assertFalse((idx == np.arange(30)[:, None]).any())
        self.assertTrue((np.diff(sim, axis=1) <= 1e-7).all())
        full = g @ g.T
        np.fill_diagonal(full, -np.inf)
        np.testing.assert_array_equal(idx[:, 0], full.argmax(1))

    def test_neighbours_from_pool_only(self):
        g = random_gallery(30)
        pool = np.arange(0, 30, 2)
        idx, _ = dba.neighbours(g, 3, pool=pool)
        self.assertTrue(np.isin(idx, pool).all())


class DbaGalleryTests(unittest.TestCase):
    """Состояние для сервиса: добавление и удаление кадров дают то же, что dba() заново."""

    def test_fit_equals_dba(self):
        g = random_gallery(80)
        st = dba.DBAGallery(g, k=2)
        np.testing.assert_allclose(st.vectors, dba.dba(g, k=2), atol=1e-6)

    def test_add_equals_rebuild_and_reports_changed_rows(self):
        g = random_gallery(120, seed=1)
        for k in (1, 3):
            st = dba.DBAGallery(g[:100], k=k)
            before = st.vectors.copy()
            changed = st.add(g[100:])
            np.testing.assert_allclose(st.vectors, dba.dba(g, k=k), atol=1e-6)
            moved = np.flatnonzero(np.abs(st.vectors[:100] - before).max(1) > 1e-7)
            np.testing.assert_array_equal(np.intersect1d(changed, np.arange(100)), moved)
            np.testing.assert_array_equal(changed[changed >= 100], np.arange(100, 120))

    def test_add_one_by_one_equals_rebuild(self):
        g = random_gallery(40, seed=2)
        st = dba.DBAGallery(g[:1], k=2)
        for i in range(1, 40):
            st.add(g[i:i + 1])
        np.testing.assert_allclose(st.vectors, dba.dba(g, k=2), atol=1e-6)

    def test_remove_equals_rebuild(self):
        g = random_gallery(90, seed=3)
        st = dba.DBAGallery(g, k=2)
        gone = np.array([0, 5, 6, 40, 89])
        changed = st.remove(gone)
        rest = np.delete(g, gone, axis=0)
        np.testing.assert_allclose(st.vectors, dba.dba(rest, k=2), atol=1e-6)
        self.assertTrue(len(changed) < len(rest))
        self.assertEqual(len(st), len(rest))

    def test_remove_down_to_one_frame(self):
        g = random_gallery(3, seed=4)
        st = dba.DBAGallery(g, k=1)
        st.remove(np.array([0, 2]))
        np.testing.assert_allclose(st.vectors, g[1:2], atol=1e-6)

    def test_add_rejects_wrong_dimension(self):
        st = dba.DBAGallery(random_gallery(4, d=16), k=1)
        with self.assertRaises(ValueError):
            st.add(np.zeros((1, 32), "float32"))
        self.assertEqual(len(st), 4)

    def test_add_to_empty_gallery(self):
        g = random_gallery(5)
        st = dba.DBAGallery(np.zeros((0, 16), "float32"), k=2)
        st.add(g)
        np.testing.assert_allclose(st.vectors, dba.dba(g, k=2), atol=1e-6)

    def test_remove_rejects_masks_and_bad_indices(self):
        st = dba.DBAGallery(random_gallery(6), k=1)
        for rows in ([False, False, False, False, False, True], [-1], [6]):
            with self.assertRaises(ValueError):
                st.remove(rows)
        self.assertEqual(len(st), 6)

    def test_remove_accepts_duplicates(self):
        g = random_gallery(10)
        st = dba.DBAGallery(g, k=1)
        st.remove([3, 3])
        np.testing.assert_allclose(st.vectors, dba.dba(np.delete(g, [3], axis=0), k=1), atol=1e-6)

    def test_vectors_are_read_only(self):
        st = dba.DBAGallery(random_gallery(10), k=1)
        with self.assertRaises(ValueError):
            st.vectors[0, 0] = 1.0


if __name__ == "__main__":
    unittest.main()
