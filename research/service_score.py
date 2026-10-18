"""Скор сервиса на матрицах: DBA, штраф за ту же камеру, top-K, оценка отказа.

Тот же код, что api/app/pp (dba.neighbours / dba_vectors, rerank.rank /
refusal_score); отличие от сервиса одно — top-K считается точно по numpy, а сервис
берёт его из pgvector (HNSW, hnsw.ef_search = max(100, PP_CAMERA_TOPK)). Совпадение
с сервисом на одних данных проверяет api/tests/test_pp_research_parity.py.

Только numpy и app.pp — без pandas, чтобы тест сервиса импортировал модуль в образе.
Используют: research/threshold_vs_gallery.py (records_service), research/bigallery_pp.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from app.pp import rerank as rr  # noqa: E402
from app.pp.dba import dba_vectors, neighbours  # noqa: E402


def pair_cos(A: np.ndarray, B: np.ndarray, top: np.ndarray, chunk: int = 256) -> np.ndarray:
    """A[q] · B[top[q, k]] кусками по запросам: (Q, K, D) целиком не держим в памяти."""
    out = np.empty(top.shape, np.result_type(A, B))
    for s in range(0, len(top), chunk):
        out[s:s + chunk] = np.einsum("qd,qkd->qk", A[s:s + chunk], B[top[s:s + chunk]])
    return out


def dba_gallery(gv: np.ndarray, k: int, pool: np.ndarray = None) -> np.ndarray:
    """g' = DBA(g, k); соседи — только из pool (индексы gv), по умолчанию все кадры."""
    idx, sim = neighbours(gv, k, pool=pool)
    return dba_vectors(gv, idx, sim, k)


def topk(s: np.ndarray, k: int) -> tuple:
    """top-k по строке в порядке убывания скора: (индексы (Q, k), скоры (Q, k))."""
    k = min(k, s.shape[1])
    top = np.argpartition(-s, k - 1, axis=1)[:, :k]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), axis=1, kind="stable"), 1)
    return top, np.take_along_axis(s, top, 1)


def decide(top: np.ndarray, cos: np.ndarray, flags: np.ndarray, flags_refusal: np.ndarray = None) -> dict:
    """Решение сервиса по готовым флагам: порядок ответа (кандидаты с флагом ранжирования — в
    конце), флаги в этом порядке и оценки отказа raw / z100 по кандидатам без флага отказа
    (flags_refusal; по умолчанию те же флаги) — как Postprocess.search."""
    flags_refusal = flags if flags_refusal is None else flags_refusal
    n = len(top)
    out = {"order": [], "flags": [], "cos": [], "conf_raw": np.full(n, -np.inf), "conf_z100": np.full(n, -np.inf)}
    for i in range(n):
        ok = np.isfinite(cos[i])
        c, f, fr, t = cos[i][ok], flags[i][ok], flags_refusal[i][ok], top[i][ok]
        o = rr.rank(c, f)
        out["order"].append(t[o])
        out["flags"].append(f[o])
        out["cos"].append(c[o])
        o2 = rr.rank(c, fr)
        out["conf_raw"][i] = rr.refusal_score(c[o2], fr[o2], "raw")
        out["conf_z100"][i] = rr.refusal_score(c[o2], fr[o2], "z100")
    return out


def rank_queries(s: np.ndarray, qbg: np.ndarray, gbg: np.ndarray, k: int, tau: float, camera: bool) -> dict:
    """s — скоры (Q, G), −inf — кадра для запроса нет. -> по каждому запросу индексы галереи
    в порядке ответа сервиса (order), флаги «та же камера» в этом порядке и оценки raw / z100."""
    top, cos = topk(s, k)
    flags = (pair_cos(qbg, gbg, top) > tau) & np.isfinite(cos) if camera else np.zeros(cos.shape, bool)
    return decide(top, cos, flags)
