#!/usr/bin/env python3
"""Добавление четвёртой модели к зафиксированной тройке.

База зафиксирована и перепроверена: sq30_ep26 0.6 + ep29new_bs 0.2 +
repro_tta 0.2. Вопрос здесь один: добавляет ли новая модель что-то сверх
неё, и с каким весом. Вес перебираем от малого к большому: если модель
слабее участников тройки, оптимум будет на малом весе, а если и там хуже
базы — она не нужна.

Считаем не только на полном наборе, но и на 20 разбиениях по машинам:
на 551 запросе шум ±1.5 mAP, и решение по одному сплиту ничего не значит.

    python mix4.py --tag siglip2_ep26
"""
import argparse
from pathlib import Path

import numpy as np

from official_protocol import official_metrics
from postproc_v2 import pipe
from rerank_test import l2, load_rows

HERE = Path(__file__).parent
BASE = {"sq30_ep26": 0.6, "ep29new_bs": 0.2, "repro_tta": 0.2}


def load(tag):
    return (l2(np.load(HERE / f"emb_{tag}_qry.npy").astype("float64")),
            l2(np.load(HERE / f"emb_{tag}_gal.npy").astype("float64")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--weights", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40")
    a = ap.parse_args()

    qry, gal = load_rows()
    tags = sorted(set(BASE) | {a.tag})
    S = {t: load(t) for t in tags}

    vids = sorted({r["vehicle_id"] for r in qry})
    folds = []
    for seed in range(a.splits):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        half = set(order[: len(order) // 2])
        idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
        folds.append((idx, [qry[i] for i in idx]))

    def run(name, w):
        q = l2(sum(x * S[t][0] for t, x in w.items()))
        g = l2(sum(x * S[t][1] for t, x in w.items()))
        sim = pipe(q, g)
        full = official_metrics(sim, qry, gal)
        vals = np.array([official_metrics(sim[i], rows, gal)["mAP@10"]
                         for i, rows in folds])
        print(f"{name:40s} {full['mAP@10']:7.2f} {full['Rank-1']:7.2f} "
              f"{vals.mean():8.2f} {vals.std():5.2f} {(vals > 0).sum():3d}/{len(folds)}",
              flush=True)
        return vals

    print(f"запросов {len(qry)}, разбиений {a.splits}, новая модель: {a.tag}")
    print(f"{'конфигурация':40s} {'mAP':>7s} {'R1':>7s} {'ср.':>8s} {'±':>5s} {'>0':>6s}")
    print("-" * 80)

    base_vals = run("тройка (база)", BASE)
    run(f"{a.tag} одна", {a.tag: 1.0})
    print("-" * 80)

    best_name, best_mean, best_vals = "тройка (база)", base_vals.mean(), base_vals
    for ws in a.weights.split(","):
        w = float(ws)
        mix = {k: v * (1 - w) for k, v in BASE.items()}
        mix[a.tag] = w
        vals = run(f"база x{1 - w:.2f} + {a.tag} {w:.2f}", mix)
        if vals.mean() > best_mean:
            best_name, best_mean, best_vals = f"+{a.tag} {w:.2f}", vals.mean(), vals

    delta = best_vals - base_vals
    print(f"\nлучшее по среднему: {best_name} -> ср. {best_mean:.2f} "
          f"(база {base_vals.mean():.2f})")
    print(f"парная дельта: {delta.mean():+.2f} ± {delta.std():.2f}, "
          f"лучше в {(delta > 0).sum()}/{len(delta)} разбиениях")
    print("вердикт: " + ("модель добавляет" if delta.mean() > 0.15 and
                         (delta > 0).sum() >= 0.75 * len(delta)
                         else "прирост в пределах шума — не берём"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
