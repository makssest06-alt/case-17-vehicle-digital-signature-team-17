import unittest

import numpy as np

from online_gallery_rerank import gallery_graph, map10, reciprocal_scores, topk_rows


class OnlineGalleryRerankTests(unittest.TestCase):
    def test_chunked_graph_matches_dense_and_excludes_self(self):
        rng = np.random.default_rng(7)
        gallery = rng.normal(size=(11, 5)).astype(np.float32)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)

        chunked = gallery_graph(gallery, k=20, chunk=3)
        dense = gallery_graph(gallery, k=20, chunk=11)

        np.testing.assert_array_equal(chunked, dense)
        self.assertEqual(chunked.shape, (11, 10))
        for row, neighbours in enumerate(chunked):
            self.assertNotIn(row, neighbours)
            self.assertEqual(len(set(neighbours.tolist())), 10)

    def test_scores_are_independent_of_query_batch(self):
        rng = np.random.default_rng(8)
        gallery = rng.normal(size=(6, 4)).astype(np.float32)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        queries = rng.normal(size=(3, 4)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        graph = gallery_graph(gallery, k=3, chunk=2)

        batched = reciprocal_scores(queries, gallery, graph, k=3, mix=0.7)
        for i in range(len(queries)):
            np.testing.assert_allclose(
                batched[i], reciprocal_scores(queries[i:i + 1], gallery, graph, 3, 0.7)[0],
                rtol=1e-6,
            )

    def test_map10_does_not_recover_positive_beyond_junk_consuming_slot(self):
        query = [{"vehicle_id": "v", "camera_id": "q"}]
        gallery = [{"vehicle_id": "v", "camera_id": "q"}]
        gallery += [{"vehicle_id": str(i), "camera_id": "x"} for i in range(9)]
        gallery += [{"vehicle_id": "v", "camera_id": "positive"}]
        scores = np.array([[0.99] + [0.98] * 9 + [0.50]], dtype=np.float32)

        result = map10(scores, query, gallery, np.array([0]))
        self.assertEqual(result, {"mAP@10": 0.0, "Rank-1": 0.0, "Rank-5": 0.0, "n": 1})

    def test_invalid_and_empty_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            gallery_graph(np.empty((0, 3)), k=1)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            gallery_graph(np.ones((2, 3)), k=0)
        with self.assertRaisesRegex(ValueError, "2-D"):
            topk_rows(np.ones(3), k=1)

    def test_single_gallery_item_has_no_neighbours(self):
        graph = gallery_graph(np.ones((1, 3), dtype=np.float32), k=5)
        self.assertEqual(graph.shape, (1, 0))


if __name__ == "__main__":
    unittest.main()
