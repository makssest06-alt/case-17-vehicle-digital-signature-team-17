#!/usr/bin/env python3
"""Score-контракт: сервис (сырой косинус) против артефактов (CSLS после постобработки).

В прежней сдаче (до 18.10) было два разных смысла confidence (вывод —
research/honest_eval_results/score_contract.md):

  service   — /v1/search: кандидат = top-1 по сырому косинусу, отказ при
              косинусе < SEARCH_THRESHOLD (0.8025);
  artifacts — candidates.csv: кандидат = top-1 после qq → k-reciprocal → CSLS
              (artifacts/postprocess.json), confidence = этот CSLS-скор, порог =
              10-й процентиль top-1 по ТЕМ ЖЕ запросам (research/reid_postprocess.py:208);
  mixed     — порог сервиса 0.8025, приложенный к CSLS-скору артефактов.

Скрипт меряет все три на одном протоколе research/honest_eval.py (big-галерея,
5 фолдов, 25 % машин без пары) метриками эталона (TP = принят и top-1 верный)
и печатает распределения скоров. Плюс та же сверка на сданных артефактах теста
(без меток — только доли принятых).

    python research/score_contract.py --data-dir /opt/case17/data
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
import calibrate_refusal_folds as crf  # noqa: E402
import honest_eval as he  # noqa: E402
import reid_postprocess as pp  # noqa: E402
import service_embeddings as se  # noqa: E402

SERVICE_THRESHOLD = 0.8025


def fold_contracts(D: he.Data, kind: str, mode: str, params: dict) -> list:
    extra = D.valid(D.distractors(kind))
    qry, gal = D.valid(D.qry), D.valid(D.gal)
    folds = he.folds_by_vehicle(sorted(D.hold), D.a.folds, D.a.seed)
    rng = np.random.default_rng(D.a.seed + 1)            # тот же open-set, что в honest_eval
    out = []
    for fvids in folds:
        fset = set(fvids)
        opened = set(rng.choice(sorted(fvids, key=int), size=int(round(len(fvids) * D.a.openset_frac)),
                                replace=False).tolist())
        Q = he.frame([r for r in qry if r["vehicle_id"] in fset])
        G = he.frame([r for r in gal if r["vehicle_id"] not in opened] + extra)
        q, g = D.vectors(list(Q.index)), D.vectors(list(G.index))
        same = ((G.vehicle_id.to_numpy()[None, :] == Q.vehicle_id.to_numpy()[:, None])
                & (G.camera_id.to_numpy()[None, :] == Q.camera_id.to_numpy()[:, None]))
        has_match = ((G.vehicle_id.to_numpy()[None, :] == Q.vehicle_id.to_numpy()[:, None]) & ~same).any(1)
        recs = {}
        for name, sim in (("raw", q @ g.T), ("csls", he.scores(q, g, "full", params))):
            if mode == "nodup":
                sim = np.where(same, -np.inf, sim)
            top = sim.argmax(1)
            recs[name] = {"conf": sim[np.arange(len(top)), top], "has_match": has_match,
                          "correct": G.vehicle_id.to_numpy()[top] == Q.vehicle_id.to_numpy()}
        out.append(recs)
    return out


def contract_metrics(folds: list) -> dict:
    res = {"service (raw, 0.8025)": [], "artifacts (CSLS, 10-й процентиль)": [], "mixed (CSLS, 0.8025)": []}
    for f in folds:
        res["service (raw, 0.8025)"].append(crf.metrics(f["raw"], SERVICE_THRESHOLD))
        q10 = float(np.quantile(f["csls"]["conf"][np.isfinite(f["csls"]["conf"])], 0.10))
        res["artifacts (CSLS, 10-й процентиль)"].append({**crf.metrics(f["csls"], q10), "threshold": q10})
        res["mixed (CSLS, 0.8025)"].append(crf.metrics(f["csls"], SERVICE_THRESHOLD))
    return {k: {m: crf.stats([p[m] for p in v]) for m in ("threshold", "recall", "TNR", "F1", "precision",
                                                         "accepted_share")}
            for k, v in res.items()}


def score_ranges(folds: list) -> dict:
    out = {}
    for name in ("raw", "csls"):
        c = np.concatenate([f[name]["conf"] for f in folds])
        c = c[np.isfinite(c)]
        out[name] = {p: round(float(np.percentile(c, p)), 4) for p in (1, 10, 50, 90, 99)}
    return out


def artifacts_on_test(art: Path) -> dict:
    """Сданные артефакты: доли принятых при каждом контракте (меток теста нет)."""
    idx = list(csv.DictReader((art / "embeddings_index.csv").open(encoding="utf-8")))
    e = np.load(art / "embeddings.npy").astype("float64")
    nq = sum(r["split"] == "query" for r in idx)
    q, g = e[:nq], e[nq:]
    p = json.loads((art / "postprocess.json").read_text(encoding="utf-8"))["params"]
    sim_pp, _, _ = pp.rank([q], [g], [1.0], qq_k=p["qq_k"], qq_alpha=p["qq_alpha"], csls_k=p["csls_k"],
                           rerank_lambda=p["rerank_lambda"], rerank_k1=p["rerank_k1"], rerank_k2=p["rerank_k2"])
    raw = (q / np.linalg.norm(q, axis=1, keepdims=True)) @ (g / np.linalg.norm(g, axis=1, keepdims=True)).T
    t_raw, t_pp = raw.max(1), sim_pp.max(1)
    q10 = float(np.quantile(t_pp, 0.10))
    return {"queries": int(nq),
            "service_raw_ge_0.8025": int((t_raw >= SERVICE_THRESHOLD).sum()),
            "artifacts_csls_ge_q10": int((t_pp >= q10).sum()), "q10": round(q10, 6),
            "mixed_csls_ge_0.8025": int((t_pp >= SERVICE_THRESHOLD).sum()),
            "csls_top1_max": round(float(t_pp.max()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--distractors", default="test_all", choices=tuple(he.DISTRACTORS))
    ap.add_argument("--out", type=Path, default=HERE / "honest_eval_results" / "score_contract.json")
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--pp-params", type=Path, default=ROOT / "artifacts" / "postprocess.json")
    ap.add_argument("--holdout", type=Path, default=HERE / "val_ids.txt")
    ap.add_argument("--exclude-frames", type=Path, default=HERE / "honest_eval_results" / "holdout_frame_dups.txt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--openset-frac", type=float, default=0.25)
    ap.add_argument("--profile", choices=se.PROFILES, default="env_example")
    ap.add_argument("--compose-ref", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--source", choices=("crops", "frames"), default="crops")
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--one-query-per-camera", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=se.DEFAULT_CACHE)
    a = ap.parse_args()

    D = he.Data(a)
    params = he.pp_params(a.pp_params)
    report = {"distractors": a.distractors, "params": params, "modes": {}}
    for mode in ("official", "nodup"):
        folds = fold_contracts(D, a.distractors, mode, params)
        report["modes"][mode] = {"contracts": contract_metrics(folds), "score_percentiles": score_ranges(folds)}
        print(f"\nрежим {mode}, дистракторы {a.distractors}:")
        for name, s in report["modes"][mode]["contracts"].items():
            print(f"  {name:36s} порог {s['threshold']['mean']:.4f}  принято {s['accepted_share']['mean']:.3f}  "
                  f"recall {s['recall']['mean']:.3f}  TNR {s['TNR']['mean']:.3f}  F1 {s['F1']['mean']:.3f}")
        print(f"  перцентили top-1: {report['modes'][mode]['score_percentiles']}")
    report["submitted_test_artifacts"] = artifacts_on_test(a.artifacts)
    print(f"\nсданные артефакты теста: {report['submitted_test_artifacts']}")
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
