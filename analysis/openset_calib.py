#!/usr/bin/env python3
"""Калибровка порога отказа на синтетическом open-set.

В нашем val-сплите все запросы имеют пару в галерее, поэтому порог отказа
по нему не настроить. Синтетический open-set: часть машин убирается из
галереи, их запросы становятся «без пары» (как в реальном open-set).

Для каждого порога считаются метрики ровно как в `evaluate.py`
(решение на уровне запроса):

    TP  есть пара, ответили и топ-1 верен
    FP  ответили, но топ-1 неверен (или пары нет — ложное срабатывание)
    FN  есть пара, но отказались
    TN  пары нет, и отказались
    F1  = 2TP / (2TP + FP + FN);   TNR = TN / (TN + FP_openset)

    python openset_calib.py --frac 0.25 --seeds 8

Веса смеси задаются флагами (по умолчанию — текущая лучшая конфигурация):

    python openset_calib.py --tags siglip2_ep28,sq30_ep26,ep29new_bs \
        --weights 0.45,0.25,0.30
"""
import argparse
from pathlib import Path

import numpy as np

from postproc_v2 import pipe
from rerank_test import l2, load_rows

HERE = Path(__file__).parent
DEFAULT_TAGS = "siglip2_ep28,sq30_ep26,ep29new_bs"
DEFAULT_WEIGHTS = "0.45,0.25,0.30"


def load(tag):
    return (l2(np.load(HERE / f"emb_{tag}_qry.npy").astype("float64")),
            l2(np.load(HERE / f"emb_{tag}_gal.npy").astype("float64")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.25,
                    help="доля машин, убранных из галереи (создаёт open-set)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--tags", default=DEFAULT_TAGS)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    a = ap.parse_args()

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    weights = [float(x) for x in a.weights.split(",")]
    if len(tags) != len(weights):
        raise SystemExit("число весов не совпадает с числом моделей")
    WEIGHTS = dict(zip(tags, weights))
    print("смесь: " + " + ".join(f"{w:.2f}*{t}" for t, w in WEIGHTS.items()), flush=True)

    qry, gal = load_rows()
    parts = [load(t) for t in WEIGHTS]

    g_vid = np.array([r["vehicle_id"] for r in gal])
    g_cam = np.array([r["camera_id"] for r in gal])
    vids = sorted({r["vehicle_id"] for r in qry})

    # уверенность и правильность считаем один раз для каждого разбиения:
    # состав галереи меняется, значит меняется и постобработка
    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        dropped = set(order[: int(len(order) * a.frac)])
        keep_g = np.array([r["vehicle_id"] not in dropped for r in gal])
        gg = [gal[j] for j in np.where(keep_g)[0]]

        q = l2(sum(w * p[0] for w, p in zip(WEIGHTS.values(), parts)))
        g = l2(sum(w * p[1][keep_g] for w, p in zip(WEIGHTS.values(), parts)))
        S = pipe(q, g)

        gv = g_vid[keep_g]
        gc = g_cam[keep_g]
        conf, correct, has_match = [], [], []
        for i, qr in enumerate(qry):
            n_pos = int(((gv == qr["vehicle_id"]) & (gc != qr["camera_id"])).sum())
            ordr = np.argsort(-S[i], kind="stable")
            clean = [j for j in ordr if not (gv[j] == qr["vehicle_id"]
                                              and gc[j] == qr["camera_id"])]
            top = clean[0]
            conf.append(float(S[i, top]))
            correct.append(bool(gv[top] == qr["vehicle_id"]))
            has_match.append(n_pos > 0)
        rows.append((np.array(conf), np.array(correct), np.array(has_match)))

    n_open = int((~rows[0][2]).sum())
    print(f"разбиений {a.seeds}, убрано {a.frac:.0%} машин из галереи")
    print(f"запросов без пары в галерее: {n_open} из {len(qry)}\n")

    print(f"{'порог':>8s} {'F1':>7s} {'TNR':>7s} {'Prec':>7s} {'Rec':>7s} "
          f"{'F1+TNR':>8s}")
    print("-" * 52)

    all_conf = np.concatenate([r[0] for r in rows])
    best = None
    for pct in (0, 2, 5, 8, 10, 12, 15, 20, 25, 30, 40):
        thr = np.quantile(all_conf, pct / 100) if pct > 0 else -np.inf
        f1s, tnrs, precs, recs = [], [], [], []
        for conf, correct, has_match in rows:
            keep = conf >= thr if pct > 0 else np.ones(len(conf), bool)
            tp = int((keep & correct).sum())
            fp = int((keep & ~correct).sum())
            fn = int((~keep & correct).sum())
            tn = int((~keep & ~has_match).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            openset = int((~has_match).sum())
            fp_open = int((keep & ~has_match).sum())
            tnr = tn / openset if openset else float("nan")
            f1s.append(f1)
            tnrs.append(tnr)
            precs.append(prec)
            recs.append(rec)
        f1, tnr = float(np.mean(f1s)), float(np.nanmean(tnrs))
        print(f"{pct:7d}% {f1:7.4f} {tnr:7.4f} {np.mean(precs):7.4f} "
              f"{np.mean(recs):7.4f} {f1 + tnr:8.4f}")
        if best is None or f1 + tnr > best[1]:
            best = (pct, f1 + tnr)

    print(f"\nлучший порог по F1+TNR: {best[0]}-й процентиль "
          f"(F1+TNR {best[1]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
