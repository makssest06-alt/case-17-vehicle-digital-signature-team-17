#!/usr/bin/env python3
"""CPU checks of streaming independence, edge cases, format and official top-10."""

import csv
import tempfile
from pathlib import Path

import numpy as np

from build_final_artifacts import compliant_scores, l2, write_artifacts


def main() -> None:
    rng = np.random.default_rng(17)
    q = l2(rng.normal(size=(7, 16)))
    g = l2(rng.normal(size=(12, 16)))
    raw, scores = compliant_scores(q, g)
    for index in range(len(q)):
        single_raw, single_scores = compliant_scores(q[index:index + 1], g)
        np.testing.assert_allclose(single_raw[0], raw[index], atol=1e-12)
        np.testing.assert_allclose(single_scores[0], scores[index], atol=1e-12)
    permuted = compliant_scores(q[::-1], g)[1]
    np.testing.assert_allclose(permuted[::-1], scores, atol=1e-12)
    for size in (2, 5, 6, 10):
        assert np.isfinite(compliant_scores(q, g[:size])[1]).all()
    for size in (0, 1):
        try:
            compliant_scores(q, g[:size])
        except ValueError:
            pass
        else:
            raise AssertionError("Small gallery must fail explicitly")
    rows_q = [{"image_id": f"q{i}"} for i in range(len(q))]
    rows_g = [{"image_id": f"g{i}"} for i in range(len(g))]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        counts = write_artifacts(out, rows_q, rows_g, q, g, -1.0)
        assert counts == {"accepted": len(q), "refused": 0}
        embeddings = np.load(out / "embeddings.npy")
        assert embeddings.dtype == np.float32 and embeddings.shape == (19, 16)
        with (out / "submission.csv").open() as stream:
            submission = list(csv.reader(stream))
        assert len(submission) == len(q) + 1
        assert all(len(row) == 11 and len(set(row[1:])) == 10 for row in submission[1:])
        with (out / "candidates.csv").open() as stream:
            candidates = list(csv.DictReader(stream))
        for i, qrow in enumerate(rows_q):
            chosen = [r for r in candidates if r["query_id"] == qrow["image_id"]]
            conf = [float(r["confidence"]) for r in chosen]
            assert conf == sorted(conf, reverse=True) and all(0 <= c <= 1 for c in conf)
            assert chosen[0]["gallery_id"] == submission[i + 1][1]
        write_artifacts(out, rows_q, rows_g, q, g, 1.0)
        with (out / "candidates.csv").open() as stream:
            assert not list(csv.DictReader(stream))
    print("PASS: independent queries, finite small-gallery CSLS, format, accept/refuse")


if __name__ == "__main__":
    main()
