#!/usr/bin/env python3
"""Поиск весов смеси из N моделей полным перебором по симплексу.

Отличие от mix4.py: тот масштабирует уже зафиксированную тройку целиком,
то есть проверяет одну ось (вес новичка при неизменных пропорциях тройки).
Здесь перебираются все веса независимо — при сильной новой модели
оптимум может лежать не на этой оси.

Максимум на 551 запросе сам по себе ничего не доказывает, поэтому скрипт
печатает не только лучшую точку, но и всё плато вокруг неё: если соседние
веса дают столько же, выбор точки не критичен. Отдельно считается
устойчивость на разбиениях по машинам.

    python mix4_grid.py --tags siglip2_ep28,sq30_ep26,ep29new_bs,repro_tta --step 0.05
"""
import argparse
import itertools
from pathlib import Path

import numpy as np

from official_protocol import official_metrics
from postproc_v2 import pipe
from rerank_test import l2, load_rows

HERE = Path(__file__).parent


def load(tag):
    return (l2(np.load(HERE / f"emb_{tag}_qry.npy").astype("float64")),
            l2(np.load(HERE / f"emb_{tag}_gal.npy").astype("float64")))


def compositions(n, total):
    """Все наборы из n неотрицательных целых с суммой total."""
    if n == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(n - 1, total - first):
            yield (first,) + rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    qry, gal = load_rows()
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

    total = int(round(1.0 / a.step))
    combos = list(compositions(len(tags), total))
    print(f"моделей {len(tags)}, шаг {a.step}, комбинаций {len(combos)}", flush=True)
    print("  " + "  ".join(f"{t[:14]:>14s}" for t in tags) + "   mAP     R1")

    scored = []
    for counts in combos:
        w = [c * a.step for c in counts]
        q = l2(sum(wi * S[t][0] for wi, t in zip(w, tags)))
        g = l2(sum(wi * S[t][1] for wi, t in zip(w, tags)))
        m = official_metrics(pipe(q, g), qry, gal)
        scored.append((m["mAP@10"], m["Rank-1"], counts))

    scored.sort(key=lambda x: -x[0])
    for m, r1, counts in scored[: a.top]:
        w = [c * a.step for c in counts]
        print("  " + "  ".join(f"{wi:14.2f}" for wi in w)
              + f" {m:6.2f} {r1:6.2f}", flush=True)

    # плато: насколько широко вокруг максимума держится результат
    best = scored[0][0]
    near = [s for s in scored if s[0] >= best - 0.5]
    print(f"\nв пределах 0.5 mAP от максимума: {len(near)} точек из {len(scored)}")

    print(f"\n{'топ-точки на разбиениях':52s} {'mAP':>7s} {'ср.':>8s} {'±':>5s}")
    print("-" * 76)
    for m, r1, counts in scored[:5]:
        w = [c * a.step for c in counts]
        q = l2(sum(wi * S[t][0] for wi, t in zip(w, tags)))
        g = l2(sum(wi * S[t][1] for wi, t in zip(w, tags)))
        sim = pipe(q, g)
        vals = np.array([official_metrics(sim[i], rows, gal)["mAP@10"]
                         for i, rows in folds])
        label = " ".join(f"{wi:.2f}" for wi in w)
        print(f"{label:52s} {m:7.2f} {vals.mean():8.2f} {vals.std():5.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
