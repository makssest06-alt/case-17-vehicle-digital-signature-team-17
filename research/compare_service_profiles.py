#!/usr/bin/env python3
"""Сдаточный тест: «docker compose up без .env» против студента из .env.example.

Меток у теста нет, поэтому меряется не качество, а насколько расходятся
ответы двух конфигураций сервиса на одних и тех же кадрах:

  * совпадение top-1 и пересечение top-10 (сырой косинус, как в /v1/search);
  * согласие решений «принять / отказать» при пороге каждого профиля;
  * похожесть ранжирования: средняя по запросам корреляция Спирмена между
    строками сходства с галереей.

Векторы двух профилей напрямую не сравниваются: resnet18 и студент живут в
разных пространствах, косинус между ними смысла не имеет. Зато студент
сравнивается с artifacts/embeddings.npy построчно — это паритет «сервис
против сданных артефактов».

    python research/compare_service_profiles.py --data-dir /opt/case17/data \\
        --json research/honest_eval_results/service_profiles_test.json

--data-dir: test_query.csv, test_gallery.csv и crops/{query,gallery}/<id>.jpg
(кропы с запасом 10% — те же, из которых make_submission.py собирает артефакты).
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
import service_embeddings as se  # noqa: E402


def read_ids(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [r["image_id"] for r in csv.DictReader(f)]


def top1_stats(sim: np.ndarray, threshold: float) -> dict:
    t = sim.max(axis=1)
    return {"top1_min": round(float(t.min()), 4), "top1_p10": round(float(np.percentile(t, 10)), 4),
            "top1_median": round(float(np.median(t)), 4), "threshold": threshold,
            "accepted": int((t >= threshold).sum()), "refused": int((t < threshold).sum())}


def ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty_like(x, dtype="float64")
    r[np.arange(x.shape[0])[:, None], np.argsort(x, axis=1)] = np.arange(x.shape[1])
    return r


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def agreement(order_a: np.ndarray, order_b: np.ndarray) -> dict:
    top1 = float(np.mean(order_a[:, 0] == order_b[:, 0]))
    ov = np.mean([len(set(x[:10]) & set(y[:10])) / 10 for x, y in zip(order_a, order_b)])
    return {"top1_same_pct": round(top1 * 100, 2), "top10_overlap_mean": round(float(ov), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--limit-query", type=int, default=0, help="первые N запросов; 0 — все")
    ap.add_argument("--cache-dir", type=Path, default=se.DEFAULT_CACHE)
    ap.add_argument("--compose-ref", default="",
                    help="коммит, из которого взять docker-compose.yml для compose_default "
                         "(аудит main: 0924fb2); пусто — рабочее дерево")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    q_ids = read_ids(a.data_dir / "test_query.csv")
    g_ids = read_ids(a.data_dir / "test_gallery.csv")
    if a.limit_query:
        q_ids = q_ids[: a.limit_query]
    crops = a.data_dir / "crops"

    prof = {}
    for name in se.PROFILES:
        env = se.profile_env(name, a.compose_ref)
        q = se.embed(name, crops / "query", q_ids, a.cache_dir, compose_ref=a.compose_ref)
        g = se.embed(name, crops / "gallery", g_ids, a.cache_dir, compose_ref=a.compose_ref)
        prof[name] = {"env": env, "q": q, "g": g, "sim": q @ g.T}

    # сданные артефакты: тот же порядок строк, что в test_*.csv
    idx = {r["image_id"]: int(r["row"]) for r in csv.DictReader((a.artifacts / "embeddings_index.csv").open())}
    art = np.load(a.artifacts / "embeddings.npy").astype("float32")
    aq, ag = art[[idx[i] for i in q_ids]], art[[idx[i] for i in g_ids]]
    with (a.artifacts / "submission.csv").open(encoding="utf-8") as f:
        sub = {r[0]: r[1:] for r in csv.reader(f)}
    gpos = {gid: j for j, gid in enumerate(g_ids)}
    sub_order = np.array([[gpos[x] for x in sub[qid][:10]] for qid in q_ids])

    orders = {n: np.argsort(-p["sim"], axis=1) for n, p in prof.items()}
    orders["artifacts_raw_cosine"] = np.argsort(-(aq @ ag.T), axis=1)
    orders["artifacts_submission"] = sub_order

    st = prof["env_example"]
    cos_q = (st["q"] * aq).sum(1) / np.linalg.norm(aq, axis=1)
    cos_g = (st["g"] * ag).sum(1) / np.linalg.norm(ag, axis=1)
    cos_all = np.concatenate([cos_q, cos_g])

    cd, ee = prof["compose_default"], prof["env_example"]
    thr_cd = float(cd["env"].get("SEARCH_THRESHOLD", "nan"))
    thr_ee = float(ee["env"].get("SEARCH_THRESHOLD", "nan"))
    acc_cd = cd["sim"].max(1) >= thr_cd
    acc_ee = ee["sim"].max(1) >= thr_ee

    out = {
        "compose_ref": a.compose_ref or "working tree",
        "n_query": len(q_ids), "n_gallery": len(g_ids),
        "profiles": {n: {"env": {k: v for k, v in p["env"].items()},
                         "dim": int(p["q"].shape[1]),
                         **top1_stats(p["sim"], float(p["env"].get("SEARCH_THRESHOLD", "nan")))}
                     for n, p in prof.items()},
        "agreement": {
            "compose_default_vs_env_example": agreement(orders["compose_default"], orders["env_example"]),
            "compose_default_vs_submission": agreement(orders["compose_default"], orders["artifacts_submission"]),
            "env_example_vs_artifacts_raw_cosine": agreement(orders["env_example"], orders["artifacts_raw_cosine"]),
            "env_example_vs_submission": agreement(orders["env_example"], orders["artifacts_submission"]),
        },
        "ranking_spearman_compose_vs_student": {
            "mean": round(float(spearman_rows(cd["sim"], ee["sim"]).mean()), 3),
            "median": round(float(np.median(spearman_rows(cd["sim"], ee["sim"]))), 3),
        },
        "refusal_decision": {
            "same_decision_pct": round(float(np.mean(acc_cd == acc_ee)) * 100, 2),
            "compose_accepts_student_refuses": int((acc_cd & ~acc_ee).sum()),
            "compose_refuses_student_accepts": int((~acc_cd & acc_ee).sum()),
            "both_accept_but_different_top1": int((acc_cd & acc_ee & (orders["compose_default"][:, 0] != orders["env_example"][:, 0])).sum()),
        },
        "parity_student_service_vs_artifacts_embeddings": {
            "rows": int(len(cos_all)),
            "cos_min": round(float(cos_all.min()), 6), "cos_p01": round(float(np.percentile(cos_all, 1)), 6),
            "cos_median": round(float(np.median(cos_all)), 6),
            "share_ge_0_9999": round(float(np.mean(cos_all >= 0.9999)), 4),
            "share_ge_0_999": round(float(np.mean(cos_all >= 0.999)), 4),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
