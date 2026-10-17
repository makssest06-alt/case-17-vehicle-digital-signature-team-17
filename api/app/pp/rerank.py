"""Штраф за ту же камеру и оценка уверенности по списку кандидатов одного запроса.

Тот же код считает предварительный порог (research/threshold_vs_gallery.py,
--score service), поэтому порог и сервис не разъезжаются.

  camera_flags  кандидат «с той же камеры»: косинус признаков фона > τ;
  rank          сначала кандидаты без флага по убыванию косинуса, затем с флагом
                (= вычет 1e3 в research/bigallery_pp.py: «исключить», λ = inf);
  refusal_score raw — косинус лучшего кандидата без флага;
                z100 — (s1 − mean(s2..s100)) / std(s2..s100) по кандидатам без флага
                из top-100 (как research/refusal_score.z_of). Если кандидатов без
                флага нет или z не определён — −inf: это отказ.
"""
import numpy as np

FORMULAS = ("raw", "z100")
Z_TOPK = 100


def camera_flags(qbg, cand_bg: list, tau: float) -> np.ndarray:
    """cand_bg — признаки фона кандидатов, None — признака нет (флага нет)."""
    if qbg is None:
        return np.zeros(len(cand_bg), bool)
    q = np.asarray(qbg, dtype="float32")
    return np.array([b is not None and float(np.dot(np.asarray(b, dtype="float32"), q)) > tau for b in cand_bg], bool)


def rank(cos: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Порядок кандидатов; при равных косинусах сохраняется исходный (по базе)."""
    cos = np.asarray(cos, dtype="float64")
    return np.lexsort((-cos, np.asarray(flags, bool)))


def refusal_score(cos_sorted: np.ndarray, flags_sorted: np.ndarray, formula: str) -> float:
    if formula not in FORMULAS:
        raise ValueError(f"REFUSAL_SCORE: одно из {FORMULAS}, получено {formula!r}")
    keep = np.asarray(cos_sorted, dtype="float64")[~np.asarray(flags_sorted, bool)][:Z_TOPK]
    if keep.size == 0:
        return float("-inf")
    if formula == "raw":
        return float(keep[0])
    rest = keep[1:]
    if rest.size == 0:
        return float("-inf")
    return float((keep[0] - rest.mean()) / max(float(rest.std()), 1e-6))


def needed_k(top_n: int, camera_on: bool, camera_topk: int, formula: str) -> int:
    """Сколько кандидатов брать из базы, чтобы хватило и ответу, и штрафу, и z100."""
    return max(top_n, camera_topk if camera_on else 0, Z_TOPK if formula == "z100" else 0)
