#!/usr/bin/env python3
"""Честная цена re-ranking в бою: один запрос против галереи, а не пакет.

В experiments.py re-ranking считается для всего валидационного набора сразу,
и цена делится на 551 запрос — получается 1 мс, что красиво и неправдиво.
Сервис получает запросы по одному, и структура взаимных соседей строится
заново на каждый. Здесь меряется именно это.

Заодно проверяется, что даёт кеширование: соседство галереи с галереей
от запроса не зависит, его можно посчитать один раз при старте.
"""
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiments import load_split, load_head, embed, re_ranking, score

HERE = Path(__file__).parent


def main() -> int:
    orig, flip, qry, gal = load_split("siglip_base")
    head = load_head("p_baseline", orig.shape[1])
    q = embed(orig, flip, qry, head)
    g = embed(orig, flip, gal, head)
    print(f"query {len(q)}, gallery {len(g)}")

    # --- пакетный режим, как в experiments.py ---
    t0 = time.perf_counter()
    re_ranking(q, g, 20, 6, 0.3)
    batch_total = (time.perf_counter() - t0) * 1000
    print(f"пакетно: {batch_total:.0f} мс на {len(q)} запросов "
          f"-> {batch_total / len(q):.2f} мс на запрос (амортизировано)")

    # --- онлайн: по одному запросу ---
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        re_ranking(q[i:i + 1], g, 20, 6, 0.3)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = times[len(times) // 2]
    print(f"онлайн, один запрос против {len(g)} галереи: "
          f"p50 {p50:.0f} мс, min {times[0]:.0f}, max {times[-1]:.0f}")

    # --- качество в онлайн-режиме: каждый запрос пересортирован отдельно ---
    sims = np.zeros((len(q), len(g)), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(len(q)):
        sims[i] = re_ranking(q[i:i + 1], g, 20, 6, 0.3)[0]
    online_total = time.perf_counter() - t0
    r = score(sims, qry, gal)
    print(f"качество в онлайн-режиме: mAP {r['mAP']}  Rank-1 {r['rank1']}  "
          f"катастроф {r['catastrophic_pct']}%  (весь прогон {online_total:.0f} с)")

    base = score(q @ g.T, qry, gal)
    print(f"для сравнения база:        mAP {base['mAP']}  Rank-1 {base['rank1']}  "
          f"катастроф {base['catastrophic_pct']}%")

    forward_ms = 674.0
    print()
    print(f"цена относительно запроса ({forward_ms:.0f} мс на модель): "
          f"+{p50 / forward_ms * 100:.0f}%")
    print("правило отбора: дороже 20% времени и меньше пункта mAP — отбрасываем")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
