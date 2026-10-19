#!/usr/bin/env python3
"""Чем на самом деле собраны артефакты сдачи в artifacts/.

Проверяет три утверждения об артефактах прежней сдачи (origin/main = 0924fb2, 16.10),
без модели и без датасета — только по файлам в репозитории:

  1. submission.csv воспроизводится из embeddings.npy конвейером
     research/reid_postprocess.rank с параметрами из postprocess.json
     (то есть его собирал reid_postprocess, а не api/scripts/make_submission.py);
  2. порог отказа в candidates.csv — 10-й процентиль top-1 ПОСЛЕ постобработки,
     посчитанный на самих тестовых запросах, а не 0.8025 из README;
  3. сколько запросов принял бы make_submission.py по умолчанию
     (сырой косинус >= SEARCH_THRESHOLD из api/app/config.py).

    python research/audit_main_artifacts.py [--json out.json]

Зависимости: только numpy (reid_postprocess.py тоже только на numpy).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import reid_postprocess as pp  # noqa: E402

# Значение по умолчанию SEARCH_THRESHOLD в api/app/config.py и в .env.example.
# Сервис не импортируется: config.py читает окружение при импорте.
SERVICE_THRESHOLD = 0.8025


def read_ids(art: Path):
    rows = list(csv.DictReader((art / "embeddings_index.csv").open(encoding="utf-8")))
    q = [r["image_id"] for r in rows if r["split"] == "query"]
    g = [r["image_id"] for r in rows if r["split"] == "gallery"]
    return q, g


def decimals(s: str) -> int:
    return len(s.split(".", 1)[1]) if "." in s else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    art = a.artifacts

    q_ids, g_ids = read_ids(art)
    emb = np.load(art / "embeddings.npy").astype("float64")
    q, g = emb[: len(q_ids)], emb[len(q_ids):]
    meta = json.loads((art / "postprocess.json").read_text(encoding="utf-8"))
    p = meta["params"]

    # 1. submission.csv против reid_postprocess.rank с параметрами артефакта
    sim, _, _ = pp.rank([q], [g], [1.0], qq_k=p["qq_k"], qq_alpha=p["qq_alpha"],
                        csls_k=p["csls_k"], rerank_lambda=p["rerank_lambda"],
                        rerank_k1=p["rerank_k1"], rerank_k2=p["rerank_k2"])
    order = np.argsort(-sim, axis=1)
    with (art / "submission.csv").open(encoding="utf-8") as f:
        sub = {r[0]: r[1:] for r in csv.reader(f)}
    sub.pop("query_id", None)
    same10 = sum(sub[qid] == [g_ids[j] for j in order[i, :10]] for i, qid in enumerate(q_ids))
    raw_top1 = np.argmax(q @ g.T, axis=1)
    top1_vs_raw = float(np.mean([sub[qid][0] != g_ids[raw_top1[i]]
                                 for i, qid in enumerate(q_ids)])) * 100

    # 2. candidates.csv: откуда порог и что лежит в confidence
    pp_top1 = sim.max(axis=1)
    thr_q10 = float(np.quantile(pp_top1, 0.10))
    rows = list(csv.DictReader((art / "candidates.csv").open(encoding="utf-8")))
    per_q = {}
    for r in rows:
        per_q.setdefault(r["query_id"], []).append(r)
    pos = {qid: i for i, qid in enumerate(q_ids)}
    conf = np.array([float(r["confidence"]) for r in rows])
    gid_is_pp_top1 = sum(r["gallery_id"] == g_ids[order[pos[r["query_id"]], 0]] for r in rows)
    max_dev = max(abs(float(r["confidence"]) - pp_top1[pos[r["query_id"]]]) for r in rows)

    # 3. что дал бы make_submission.py по умолчанию: сырой косинус >= порога сервиса
    raw_max = (q @ g.T).max(axis=1)

    out = {
        "submission": {
            "queries": len(q_ids),
            "top10_identical_to_reid_postprocess_rank": int(same10),
            "top1_differs_from_raw_cosine_pct": round(top1_vs_raw, 2),
            "postprocess_json_top1_mismatch_pct": meta.get("consistency", {}).get("top1_mismatch_pct"),
        },
        "candidates": {
            "rows": len(rows),
            "queries_with_candidates": len(per_q),
            "max_candidates_per_query": max(len(v) for v in per_q.values()),
            "confidence_min": float(conf.min()),
            "confidence_max": float(conf.max()),
            "confidence_decimals": sorted({decimals(r["confidence"]) for r in rows}),
            "gallery_id_equals_postprocess_top1": int(gid_is_pp_top1),
            "max_abs_confidence_minus_postprocess_top1": float(max_dev),
            "quantile_0_10_of_postprocess_top1_on_test": round(thr_q10, 6),
            "postprocess_json_threshold": meta.get("refusal", {}).get("threshold"),
            "refused_at_that_quantile": int((pp_top1 < thr_q10).sum()),
        },
        "make_submission_default": {
            "threshold_raw_cosine": SERVICE_THRESHOLD,
            "accepted_queries": int((raw_max >= SERVICE_THRESHOLD).sum()),
            "refused_queries": int((raw_max < SERVICE_THRESHOLD).sum()),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
