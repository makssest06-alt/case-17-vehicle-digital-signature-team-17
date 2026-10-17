#!/usr/bin/env python3
"""Порог отказа на режиме «без почти-дублей», 5 фолдов, метрики эталона.

Дополняет research/REFUSAL_CALIBRATION.md (calibrate_refusal.py) тем, чего там нет:

  * распределение — режим nodup research/honest_eval.py: из галереи убраны все
    кадры той же машины с той же камеры, что у запроса, в том числе для отказа.
    Ближе всего к «чистой» выборке экспертов; верный кандидат — только с
    другой камеры;
  * метрики — семантика эталона research/official_evaluate.candidate_metrics:
    TP = принят И верхний кандидат верный. В calibrate_refusal.py recall —
    доля принятых запросов с парой, даже если top-1 неверный; на кросс-камерных
    top-1 это заметно завышает;
  * калибровка честная: порог выбирается на одном фолде (правило то же, что в
    calibrate_refusal.choose_threshold: максимум TNR при recall >= цели) и
    проверяется на остальных четырёх; повтор для каждого из 5 фолдов даёт
    разброс и самого порога, и метрик;
  * рабочие точки recall >= 0.6 и >= 0.7, для сравнения — текущие 0.8025 и 0.8465.

Галерея — big из honest_eval: все машины holdout + дистракторы (по умолчанию
весь тест, 2.2k кадров), 25 % машин фолда без пары. Скор — сырой косинус top-1,
как в /v1/search.

    python research/calibrate_refusal_folds.py --data-dir /opt/case17/data
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import honest_eval as he  # noqa: E402
import service_embeddings as se  # noqa: E402

TARGETS = (0.6, 0.7)
CURRENT = {"сбалансированный 0.8025": 0.8025, "строгий 0.8465": 0.8465}


def fold_records(D: he.Data, kind: str, mode: str) -> list:
    """По фолду: top-1 (сырой косинус), верен ли он и есть ли у запроса пара."""
    extra = D.valid(D.distractors(kind))
    qry, gal = D.valid(D.qry), D.valid(D.gal)
    folds = he.folds_by_vehicle(sorted(D.hold), D.a.folds, D.a.seed)
    rng = np.random.default_rng(D.a.seed + 1)           # тот же open-set, что в honest_eval
    out = []
    for fvids in folds:
        fset = set(fvids)
        n_open = int(round(len(fvids) * D.a.openset_frac))
        opened = set(rng.choice(sorted(fvids, key=int), size=n_open, replace=False).tolist())
        q_rows = [r for r in qry if r["vehicle_id"] in fset]
        g_rows = [r for r in gal if r["vehicle_id"] not in opened] + extra
        Q, G = he.frame(q_rows), he.frame(g_rows)
        sim = D.vectors(list(Q.index)) @ D.vectors(list(G.index)).T
        q_vid = Q.vehicle_id.to_numpy()[:, None]; q_cam = Q.camera_id.to_numpy()[:, None]
        g_vid = G.vehicle_id.to_numpy()[None, :]; g_cam = G.camera_id.to_numpy()[None, :]
        same = (g_vid == q_vid) & (g_cam == q_cam)
        if mode == "nodup":
            sim = np.where(same, -np.inf, sim)
        top = sim.argmax(1)
        conf = sim[np.arange(len(top)), top]
        has_match = ((g_vid == q_vid) & ~same).any(1)          # valid_positives > 0 у эталона
        correct = G.vehicle_id.to_numpy()[top] == Q.vehicle_id.to_numpy()
        out.append({"conf": conf, "has_match": has_match, "correct": correct,
                    "check": (Q, G, sim, mode)})
    return out


def metrics(rec: dict, t: float) -> dict:
    """candidate_metrics эталона на массивах (кандидат — один, top-1)."""
    acc = rec["conf"] >= t
    hm, ok = rec["has_match"], rec["correct"]
    tp = int((acc & hm & ok).sum())
    fp = int((acc & ~(hm & ok)).sum())
    fn = int((~acc & hm).sum())
    tn = int((~acc & ~hm).sum())
    fp_open = int((acc & ~hm).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(t), "precision": p, "recall": r,
            "F1": 2 * p * r / (p + r) if p + r else 0.0,
            "TNR": tn / (tn + fp_open) if tn + fp_open else float("nan"),
            "accepted_share": float(acc.mean()), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def merge(recs: list) -> dict:
    return {k: np.concatenate([r[k] for r in recs]) for k in ("conf", "has_match", "correct")}


def choose(rec: dict, target: float) -> dict:
    """Правило calibrate_refusal.choose_threshold, но с recall эталона."""
    values = np.unique(rec["conf"][np.isfinite(rec["conf"])])
    points = [metrics(rec, float(t)) for t in np.concatenate(([np.inf], values[::-1]))]
    feasible = [p for p in points if p["recall"] + 1e-12 >= target]
    if not feasible:
        return {**metrics(rec, -np.inf), "infeasible": True}
    return max(feasible, key=lambda p: (p["TNR"], p["precision"], p["threshold"]))


def self_check(recs: list, threshold: float) -> None:
    """Мои массивы против he.evaluate (эталон) на одном пороге."""
    Q, G, sim, mode = recs[0]["check"]
    raw = sim.copy()
    m = he.evaluate(np.where(np.isfinite(sim), sim, -np.inf), raw, Q, G, "official", threshold)
    mine = metrics(recs[0], threshold)
    for k in ("TP", "FP", "FN", "TN"):
        if m[k] != mine[k]:
            raise AssertionError(f"расхождение с эталоном: {k} {mine[k]} != {m[k]}")


def stats(vals: list) -> dict:
    """Среднее, выборочный разброс (ddof=1, как в honest_eval.summarize), min, max, n."""
    v = np.array(vals, dtype="float64")
    v = v[np.isfinite(v)]
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else float("nan"),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "n": int(len(v))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--distractors", default="test_all", choices=tuple(he.DISTRACTORS))
    ap.add_argument("--modes", default="nodup,official")
    ap.add_argument("--fixed", action="append", default=[],
                    help="ещё фиксированный порог, «имя=значение», например «отчёт cross-camera=0.6699»")
    ap.add_argument("--out", type=Path, default=HERE / "honest_eval_results" / "refusal_folds.json")
    # те же параметры данных и модели, что у honest_eval
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
    report = {"distractors": a.distractors, "folds": a.folds, "openset_frac": a.openset_frac,
              "env": {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~")
                      for k, v in D.env.items()}, "modes": {}}
    fixed = dict(CURRENT)
    for kv in a.fixed:
        name, _, val = kv.rpartition("=")
        fixed[name or f"порог {val}"] = float(val)
    for mode in a.modes.split(","):
        recs = fold_records(D, a.distractors, mode)
        self_check(recs, 0.8025)
        block = {"current": {}, "targets": {}}
        for name, t in fixed.items():
            per = [metrics(r, t) for r in recs]
            block["current"][name] = {k: stats([p[k] for p in per]) for k in ("recall", "TNR", "F1", "precision")}
        for target in TARGETS:
            thr, ev = [], []
            for f in range(len(recs)):
                chosen = choose(recs[f], target)                 # подбор на одном фолде
                rest = merge([r for i, r in enumerate(recs) if i != f])
                thr.append(chosen["threshold"])
                ev.append(metrics(rest, chosen["threshold"]))    # проверка на остальных
            block["targets"][f"recall>={target}"] = {
                "threshold": stats(thr),
                **{k: stats([e[k] for e in ev]) for k in ("recall", "TNR", "F1", "precision", "accepted_share")}}
        report["modes"][mode] = block
        print(f"\nрежим {mode}, дистракторы {a.distractors}:")
        for name, s in block["current"].items():
            print(f"  {name:24s} recall {s['recall']['mean']:.3f}  TNR {s['TNR']['mean']:.3f}  F1 {s['F1']['mean']:.3f}")
        for name, s in block["targets"].items():
            print(f"  цель {name:12s} порог {s['threshold']['mean']:.4f} ± {s['threshold']['std']:.4f}  "
                  f"на остальных фолдах: recall {s['recall']['mean']:.3f} ± {s['recall']['std']:.3f}  "
                  f"TNR {s['TNR']['mean']:.3f} ± {s['TNR']['std']:.3f}  F1 {s['F1']['mean']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
