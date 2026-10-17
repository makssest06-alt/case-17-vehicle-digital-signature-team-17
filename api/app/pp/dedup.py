"""Правило Андрея: почти-дубль = очень высокое сходство И та же рамка.

Кандидат — почти-дубль запроса (та же машина с той же камеры, соседний кадр), если
  сырой косинус эмбеддингов (без DBA) ≥ s_any (одного сходства хватает; inf — выкл)
  ИЛИ сырой косинус ≥ s_hi и рамки близки в координатах кадра: IoU ≥ t_iou, сдвиг
  центра ≤ t_center (в долях среднего размера рамки), размеры — не больше чем в
  exp(t_size) раз. Наша калибровка: s_hi 0.84, IoU 0.6, центр 0.3; Андрея: s_any 0.94
  ИЛИ s_hi 0.85 и IoU 0.7.
Почти-дубли исключаются из ранжирования и из оценки отказа (в метрике эталона они всё
равно junk). Пороги подобраны по camera_id обучающих машин вне фолда
(research/dedup_compare.py, honest_eval_results/dedup_compare_a10.md).

Рамки — xywh в пикселях кадра. Сравнивать пиксели честно: почти-дубль по определению
снят той же камерой, то есть в том же разрешении. Рамка без площади (w или h ≤ 0,
так bulk-import пишет строки без координат) — не почти-дубль.

Только numpy; работает и на одном запросе (qbox (4,), boxes (K, 4)), и на матрицах
(qbox (Q, 1, 4), boxes (Q, K, 4)).
"""
import numpy as np


def box_features(qbox, boxes) -> dict:
    """IoU, сдвиг центра в долях среднего размера рамки, наибольшее |log| отношения сторон."""
    q, b = np.asarray(qbox, dtype="float64"), np.asarray(boxes, dtype="float64")
    qx, qy, qw, qh = (q[..., i] for i in range(4))
    bx, by, bw, bh = (b[..., i] for i in range(4))
    iw = np.clip(np.minimum(qx + qw, bx + bw) - np.maximum(qx, bx), 0, None)
    ih = np.clip(np.minimum(qy + qh, by + bh) - np.maximum(qy, by), 0, None)
    inter = iw * ih
    union = qw * qh + bw * bh - inter
    ok = (qw > 0) & (qh > 0) & (bw > 0) & (bh > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(ok, inter / np.where(union > 0, union, 1), 0.0)
        scale = np.sqrt(((qw + bw) / 2) * ((qh + bh) / 2))
        center = np.where(ok, np.hypot((qx + qw / 2) - (bx + bw / 2), (qy + qh / 2) - (by + bh / 2))
                          / np.where(scale > 0, scale, 1), np.inf)
        size = np.where(ok, np.maximum(np.abs(np.log(np.where(ok, bw / np.where(qw > 0, qw, 1), 1))),
                                       np.abs(np.log(np.where(ok, bh / np.where(qh > 0, qh, 1), 1)))), np.inf)
    return {"iou": iou, "center": center, "size": size, "ok": ok}


def rule_from_features(raw_cos, f: dict, s_hi: float, t_iou: float, t_center: float, t_size: float,
                       s_any: float = float("inf")) -> np.ndarray:
    """Правило по готовым признакам box_features — одно место и для сервиса, и для калибровки:
    cos ≥ s_any ИЛИ (cos ≥ s_hi И рамки близки). s_any = inf — только вторая часть."""
    c = np.asarray(raw_cos)
    return (c >= s_any) | ((c >= s_hi) & f["ok"] & (f["iou"] >= t_iou) & (f["center"] <= t_center) &
                           (f["size"] <= t_size))


def near_dup_flags(raw_cos, qbox, boxes, s_hi: float, t_iou: float, t_center: float, t_size: float,
                   s_any: float = float("inf")) -> np.ndarray:
    """True — кандидат почти-дубль запроса."""
    return rule_from_features(raw_cos, box_features(qbox, boxes), s_hi, t_iou, t_center, t_size, s_any)
