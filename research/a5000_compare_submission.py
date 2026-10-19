#!/usr/bin/env python3
"""Сравнение двух прогонов make_submission (файлы в репозитории против, например, GPU/FP16 в контейнере жюри):
sha256 трёх файлов, тот же top-1, множество принятых запросов, разница confidence, векторы.

    python research/a5000_compare_submission.py artifacts <другой каталог> [--out compare.json]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

FILES = ("submission.csv", "embeddings.npy", "candidates.csv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def top1(path: Path) -> dict:
    return {r["query_id"]: r["gallery_id_1"] for r in rows(path)}


def candidates(path: Path) -> dict:
    return {(r["query_id"], r["gallery_id"]): float(r["confidence"]) for r in rows(path)}


def compare(repo: Path, other: Path) -> dict:
    res = {"repo": str(repo), "other": str(other),
           "sha256": {f: {"repo": sha256(repo / f), "other": sha256(other / f)} for f in FILES}}
    res["same_files"] = [f for f in FILES if res["sha256"][f]["repo"] == res["sha256"][f]["other"]]

    ta, tb = top1(repo / "submission.csv"), top1(other / "submission.csv")
    if set(ta) != set(tb):
        raise SystemExit(f"разные наборы query в submission.csv: {len(ta)} против {len(tb)}")
    same = sum(ta[q] == tb[q] for q in ta)
    res["top1"] = {"same": same, "of": len(ta), "share": same / max(len(ta), 1)}

    ca, cb = candidates(repo / "candidates.csv"), candidates(other / "candidates.csv")
    acc_a, acc_b = {q for q, _ in ca}, {q for q, _ in cb}
    res["accepted"] = {"repo": len(acc_a), "other": len(acc_b), "refused_to_accepted": len(acc_b - acc_a),
                       "accepted_to_refused": len(acc_a - acc_b), "same_set": acc_a == acc_b}
    common = ca.keys() & cb.keys()
    res["confidence"] = {"pairs_common": len(common), "pairs_only_repo": len(ca.keys() - cb.keys()),
                         "pairs_only_other": len(cb.keys() - ca.keys()),
                         "max_abs_diff": max((abs(ca[k] - cb[k]) for k in common), default=0.0)}

    ia, ib = [r["image_id"] for r in rows(repo / "embeddings_index.csv")], \
        [r["image_id"] for r in rows(other / "embeddings_index.csv")]
    ea, eb = np.load(repo / "embeddings.npy").astype("float64"), np.load(other / "embeddings.npy").astype("float64")
    if ia != ib or ea.shape != eb.shape:
        raise SystemExit(f"embeddings.npy: разный порядок строк или форма {ea.shape} против {eb.shape}")
    na = ea / np.maximum(np.linalg.norm(ea, axis=1, keepdims=True), 1e-12)
    nb = eb / np.maximum(np.linalg.norm(eb, axis=1, keepdims=True), 1e-12)
    res["embeddings"] = {"rows": int(ea.shape[0]), "dim": int(ea.shape[1]), "max_abs_diff": float(np.abs(ea - eb).max()),
                         "min_cos": float((na * nb).sum(1).min())}
    a, c, e = res["accepted"], res["confidence"], res["embeddings"]
    res["summary"] = (f"sha256 совпали у {len(res['same_files'])} из 3 ({', '.join(res['same_files']) or 'ни одного'}); "
                      f"тот же top-1 у {same} из {len(ta)} запросов ({100 * res['top1']['share']:.2f} %); принятых "
                      f"{a['repo']} → {a['other']} (отказ → принят {a['refused_to_accepted']}, принят → отказ "
                      f"{a['accepted_to_refused']}); max |Δ confidence| {c['max_abs_diff']:.4f} на {c['pairs_common']} "
                      f"общих парах; векторы: max |Δ| {e['max_abs_diff']:.2e}, мин. косинус {e['min_cos']:.7f}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", type=Path, help="каталог с файлами сдачи из репозитория (artifacts/)")
    ap.add_argument("other", type=Path, help="каталог другого прогона make_submission")
    ap.add_argument("--out", type=Path, default=None, help="куда записать JSON")
    a = ap.parse_args()
    res = compare(a.repo, a.other)
    text = json.dumps(res, ensure_ascii=False, indent=2)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    print(res["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
