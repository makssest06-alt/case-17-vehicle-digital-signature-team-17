#!/usr/bin/env python3
"""Независимая проверка весов 4-way смеси.

Веса подбираются по тем же 551 запросам, на которых потом рапортуются, —
это завышает результат. Здесь две проверки:

1. Парное сравнение с прежней тройкой на разбиениях с ДРУГИМИ зёрнами
   (подбор шёл на 0..19, проверка — на 100..139).
2. Честная схема tune/test: веса подбираются на одной половине машин и
   проверяются на другой, затем наоборот. Если выбор весов — подгонка под
   шум, на второй половине он не выиграет.

    python mix4_validate.py --tags siglip2_ep28,sq30_ep26,ep29new_bs,repro_tta \
        --weights 0.50,0.20,0.30,0.00
"""
import argparse
import itertools
from pathlib import Path

import numpy as np

from official_protocol import official_metrics
from postproc_v2 import pipe
from rerank_test import l2, load_rows

HERE = Path(__file__).parent
BASE3 = {"sq30_ep26": 0.6, "ep29new_bs": 0.2, "repro_tta": 0.2}


def load(tag):
    return (l2(np.load(HERE / f"emb_{tag}_qry.npy").astype("float64")),
            l2(np.load(HERE / f"emb_{tag}_gal.npy").astype("float64")))


def compositions(n, total):
    if n == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(n - 1, total - first):
            yield (first,) + rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--weights", required=True, help="веса через запятую, сумма 1")
    ap.add_argument("--step", type=float, default=0.1, help="шаг перебора при tune")
    a = ap.parse_args()

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    cand = [float(x) for x in a.weights.split(",")]
    if len(cand) != len(tags):
        raise SystemExit("число весов не совпадает с числом моделей")

    qry, gal = load_rows()
    S = {t: load(t) for t in tags}
    missing = [t for t in BASE3 if t not in S]
    if missing:
        raise SystemExit(f"для базовой тройки не хватает эмбеддингов: {missing}")

    def mix(w):
        q = l2(sum(wi * S[t][0] for wi, t in zip(w, tags)))
        g = l2(sum(wi * S[t][1] for wi, t in zip(w, tags)))
        return pipe(q, g)

    sim_cand = mix(cand)
    sim_base = mix([BASE3.get(t, 0.0) for t in tags])

    vids = sorted({r["vehicle_id"] for r in qry})

    def folds(seeds):
        out = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            order = vids[:]
            rng.shuffle(order)
            half = set(order[: len(order) // 2])
            idx = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
            out.append((idx, [qry[i] for i in idx]))
        return out

    # 1) парное сравнение на независимых зёрнах
    print("=== разбиения с зёрнами 100..139 (подбор весов шёл на 0..19) ===")
    fs = folds(range(100, 140))
    c = np.array([official_metrics(sim_cand[i], rows, gal)["mAP@10"] for i, rows in fs])
    b = np.array([official_metrics(sim_base[i], rows, gal)["mAP@10"] for i, rows in fs])
    d = c - b
    print(f"прежняя тройка : {b.mean():6.2f} ± {b.std():.2f}")
    print(f"4-way          : {c.mean():6.2f} ± {c.std():.2f}")
    print(f"парная дельта  : {d.mean():+.2f} ± {d.std():.2f}, "
          f"лучше в {(d > 0).sum()}/{len(d)}")

    # 2) tune на одной половине машин, test на другой
    print("\n=== tune на одной половине машин, test на другой ===")
    total = int(round(1.0 / a.step))
    combos = list(compositions(len(tags), total))
    for seed in (7, 21):
        rng = np.random.default_rng(seed)
        order = vids[:]
        rng.shuffle(order)
        half = set(order[: len(order) // 2])
        ia = [i for i, r in enumerate(qry) if r["vehicle_id"] in half]
        ib = [i for i, r in enumerate(qry) if r["vehicle_id"] not in half]
        ra, rb = [qry[i] for i in ia], [qry[i] for i in ib]

        best, best_w = -1.0, None
        for counts in combos:
            w = [x * a.step for x in counts]
            s = mix(w)
            v = official_metrics(s[ia], ra, gal)["mAP@10"]
            if v > best:
                best, best_w = v, w

        s_tuned = mix(best_w)
        base_a = official_metrics(sim_base[ia], ra, gal)["mAP@10"]
        base_b = official_metrics(sim_base[ib], rb, gal)["mAP@10"]
        print(f"\nзерно {seed}: машин A {len(half)} (запросов {len(ra)}), "
              f"B {len(vids) - len(half)} (запросов {len(rb)})")
        print(f"  лучшие веса на A: {[round(x, 2) for x in best_w]} -> A {best:.2f}")
        print(f"  тройка  : A {base_a:6.2f}   B {base_b:6.2f}")
        print(f"  тюненые : A {official_metrics(s_tuned[ia], ra, gal)['mAP@10']:6.2f}   "
              f"B {official_metrics(s_tuned[ib], rb, gal)['mAP@10']:6.2f}")
        print(f"  кандидат: A {official_metrics(sim_cand[ia], ra, gal)['mAP@10']:6.2f}   "
              f"B {official_metrics(sim_cand[ib], rb, gal)['mAP@10']:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
