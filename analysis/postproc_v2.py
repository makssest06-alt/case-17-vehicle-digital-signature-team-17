#!/usr/bin/env python3
"""Трансдуктивные методы постобработки (используют query и gallery вместе).

Попарный re-ranking смотрит только на связи query-gallery. Но у нас есть и
связи query-query: ~3.7 запроса на машину. Это дополнительный сигнал.

Проверяем: CSLS, нормировка строк, Sinkhorn, расширение по query-соседям.
Всё — чистая постобработка кешированных эмбеддингов.
"""
from pathlib import Path

import numpy as np

from rerank_test import evaluate, l2, load_rows

HERE = Path(__file__).parent


def csls(S, k=10):
    """Cross-domain similarity local scaling: вычитаем 'типичность' объекта."""
    rq = np.sort(S, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    rg = np.sort(S, axis=0)[-k:, :].mean(axis=0, keepdims=True)
    return 2 * S - rq - rg


def row_norm(S, k=None):
    """Вычитаем среднее (или top-k среднее) сходство запроса — убирает 'хабы'."""
    if k is None:
        base = S.mean(axis=1, keepdims=True)
    else:
        base = np.sort(S, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    return S - base


def sinkhorn(S, tau=0.05, iters=20):
    """Симметричная нормировка матрицы сходства (оптимальный транспорт)."""
    Z = np.exp((S - S.max()) / tau)
    for _ in range(iters):
        Z /= Z.sum(axis=1, keepdims=True) + 1e-12
        Z /= Z.sum(axis=0, keepdims=True) + 1e-12
    return Z


def qq_expand(q, g, k=1, alpha=1.0):
    """Расширение запроса его же query-соседями (не gallery)."""
    sim_qq = q @ q.T
    np.fill_diagonal(sim_qq, -np.inf)
    idx = np.argsort(-sim_qq, axis=1)[:, :k]
    w = np.clip(sim_qq[np.arange(len(q))[:, None], idx], 0, None) ** alpha
    return l2(q + np.einsum("nk,nkd->nd", w, q[idx]))


# Параметры конвейера подобраны под смесь
# (siglip2-squash 0.45 + sq30-squash 0.25 + ep29-base 0.30).
# k1=30 и alpha=3.0 пришли от прежней тройки, CSLS_K=5 подобран заново.
# CSLS_K проверен парно на 40 независимых разбиениях: +0.64 ± 0.39 mAP,
# лучше в 39/40 (`postproc_validate.py`). Отключение query expansion (qk=0)
# тюнер предлагал, но на независимых разбиениях оно не подтвердилось
# (87.98 против 87.98), поэтому расширение оставлено.
RR_K1 = 30
RR_K2 = 6
RR_LAMBDA = 0.7
CSLS_K = 5
QQ_ALPHA = 3.0


def pipe(q, g):
    """Полный конвейер постобработки: qq-expand -> re-ranking -> CSLS."""
    from rerank_test import k_reciprocal_rerank

    qe = qq_expand(q, g, 1, QQ_ALPHA)
    return csls(-k_reciprocal_rerank(qe, g, RR_K1, RR_K2, RR_LAMBDA), CSLS_K)


def main():
    qry, gal = load_rows()
    q = l2(np.load(HERE / "emb_geom_qry.npy").astype("float64"))
    g = l2(np.load(HERE / "emb_geom_gal.npy").astype("float64"))
    S = q @ g.T

    print("baseline            :", evaluate(S, qry, gal))

    print("\n--- CSLS (нормировка 'типичности') ---")
    for k in (1, 3, 5, 10, 20):
        print(f"  k={k:<3} {evaluate(csls(S, k), qry, gal)}")

    print("\n--- вычитание среднего/топ-k сходства запроса ---")
    for k in (None, 5, 10, 20):
        print(f"  k={str(k):<4} {evaluate(row_norm(S, k), qry, gal)}")

    print("\n--- Sinkhorn ---")
    for tau in (0.02, 0.05, 0.1, 0.2):
        print(f"  tau={tau:<5} {evaluate(sinkhorn(S, tau), qry, gal)}")

    print("\n--- расширение запроса query-соседями ---")
    for k in (1, 2, 3):
        for a in (1.0, 3.0):
            qe = qq_expand(q, g, k, a)
            print(f"  k={k} alpha={a:<4} {evaluate(qe @ g.T, qry, gal)}")

    print("\n--- комбо: qq-expand + CSLS ---")
    qe = qq_expand(q, g, 1, 1.0)
    for k in (5, 10):
        print(f"  k={k:<3} {evaluate(csls(qe @ g.T, k), qry, gal)}")

    print("\n--- комбо: лучший + re-ranking ---")
    from rerank_test import k_reciprocal_rerank

    qe = qq_expand(q, g, 1, 1.0)
    for lam in (0.3, 0.5, 0.7):
        print(f"  lambda={lam} {evaluate(-k_reciprocal_rerank(qe, g, 20, 6, lam), qry, gal)}")


if __name__ == "__main__":
    main()
