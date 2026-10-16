#!/usr/bin/env python3
"""Выбор порога отказа для candidates.csv.

Механика оценки (`evaluate.py`, candidate_metrics) важна для решения:

    F1 = 2*TP / (2*TP + FP + FN)

Запрос, у которого топ-1 неверен, при отказе превращается из FP в FN —
**F1 не меняется**. А отказ по запросу с верным топ-1 уменьшает TP и
увеличивает FN — **F1 падает**. Поэтому отказ вредит только тогда, когда
режет верные ответы.

Если в выборке нет open-set, отказ не может улучшить F1 и может только
испортить. Если open-set есть, отказ по запросу без пары даёт TN и
улучшает precision и TNR.

Скрипт показывает обе стороны: F1 как функцию порога и точность
отклонённых запросов (какая доля из них имела верный топ-1).

    python openset_tune.py
"""
from pathlib import Path

import numpy as np

from postproc_v2 import pipe
from rerank_test import l2, load_rows

HERE = Path(__file__).parent
WEIGHTS = {"sq30_ep26": 0.6, "ep29new_bs": 0.2, "repro_tta": 0.2}


def load(tag):
    return (l2(np.load(HERE / f"emb_{tag}_qry.npy").astype("float64")),
            l2(np.load(HERE / f"emb_{tag}_gal.npy").astype("float64")))


def main() -> int:
    qry, gal = load_rows()
    parts = [load(t) for t in WEIGHTS]
    q = l2(sum(w * p[0] for w, p in zip(WEIGHTS.values(), parts)))
    g = l2(sum(w * p[1] for w, p in zip(WEIGHTS.values(), parts)))
    S = pipe(q, g)

    g_vid = np.array([r["vehicle_id"] for r in gal])
    g_cam = np.array([r["camera_id"] for r in gal])

    conf, correct, has_match = [], [], []
    for i, qr in enumerate(qry):
        order = np.argsort(-S[i], kind="stable")
        # junk-пары (тот же vid И та же cam) исключаются, как в evaluate.py
        clean = [j for j in order if not (g_vid[j] == qr["vehicle_id"]
                                          and g_cam[j] == qr["camera_id"])]
        n_pos = int(((g_vid == qr["vehicle_id"]) & (g_cam != qr["camera_id"])).sum())
        top = clean[0]
        conf.append(float(S[i, top]))
        correct.append(bool(g_vid[top] == qr["vehicle_id"]))
        has_match.append(n_pos > 0)

    conf = np.array(conf)
    correct = np.array(correct)
    has_match = np.array(has_match)
    print(f"запросов {len(qry)}, с позитивом {has_match.sum()}, "
          f"без пары {int((~has_match).sum())}")
    print(f"точность топ-1 среди всех: {correct.mean() * 100:.2f}%\n")

    print("Если бы порог отсекал долю запросов с наименьшей уверенностью:")
    print(f"{'отсечено':>9s} {'F1':>7s} {'точность':>9s} {'верных':>8s} "
          f"{'точн. отсева':>13s}")
    print("-" * 52)
    for frac in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        thr = np.quantile(conf, frac) if frac > 0 else -np.inf
        keep = conf >= thr if frac > 0 else np.ones(len(conf), bool)
        tp = int((keep & correct).sum())
        fp = int((keep & ~correct).sum())
        fn = int((~keep & correct).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        cut = ~keep
        acc_cut = correct[cut].mean() * 100 if cut.any() else float("nan")
        n_cut_correct = int(correct[cut].sum())
        print(f"{frac * 100:8.0f}% {f1:7.4f} {prec * 100:8.2f}% "
              f"{n_cut_correct:8d} {acc_cut:12.1f}%")

    print("\nВывод: если отсев идёт по уверенности, а уверенность слабо связана")
    print("с правильностью, F1 падает. Смотрите колонку 'точн. отсева' —")
    print("если она близка к общей точности, порог не различает верные ответы.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
