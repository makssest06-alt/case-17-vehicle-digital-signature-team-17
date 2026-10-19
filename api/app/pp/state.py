"""Постобработка галереи в сервисе: DBA и штраф за ту же камеру. По умолчанию выключена.

Включается флагами (config.py, блок «постобработка»):
  PP_DBA_K           0 — выкл; k > 0 — кадры галереи ищутся по g' = DBA(g, k)
                     (research/HONEST_EVAL.md). g' лежат в колонке embedding_dba,
                     исходные embedding не трогаются;
  PP_DEDUP_RANK      что уходит в конец ранжирования: off; bbox — правило рамок:
                     почти-дубль по сырому косинусу и рамке (app/pp/dedup.py); bg —
                     «та же камера» по признаку фона (косинус > PP_CAMERA_TAU); both — OR;
  PP_DEDUP_REFUSAL   что не учитывается в оценке отказа (те же значения; пусто — как
                     ранжирование). По умолчанию (с 20.10) — ранжирование off, отказ off
                     (пресет «DBA»); запасной — отказ both (итог 18.10, docs/refusal.md);
  PP_DEDUP           сокращение для обоих;
  PP_CAMERA_PENALTY  off; store — при добавлении кадра считать и хранить признак
                     фона (колонка bg), в поиске не использовать; on — то же, что
                     PP_DEDUP=bg (оставлено для совместимости).
Схема дополняется только при включённых флагах и только новыми колонками с
NULL (ADD COLUMN IF NOT EXISTS): код без постобработки их не видит.

Состояние в памяти — один процесс (uvicorn --workers 1): исходные векторы и
соседи DBA (DBAGallery), признаки фона в float16. База — источник истины: при
старте состояние строится из неё, g' в базе сверяются и дописываются; любая
ошибка записи — состояние перечитывается из базы.
"""
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import background as bgm
from . import dedup as dd
from . import rerank as rr
from .dba import DBAGallery

log = logging.getLogger("case17.pp")
CAMERA_MODES = ("off", "store", "on")
DEDUP_MODES = ("off", "bbox", "bg", "both")


def dedup_rank(settings) -> str:
    """Исключение в ранжировании: PP_DEDUP_RANK, иначе PP_DEDUP, иначе bg при PP_CAMERA_PENALTY=on."""
    if settings.pp_dedup_rank:
        return settings.pp_dedup_rank
    if settings.pp_dedup != "off":
        return settings.pp_dedup
    return "bg" if settings.pp_camera_penalty == "on" else "off"


def dedup_refusal(settings) -> str:
    """Исключение в оценке отказа: PP_DEDUP_REFUSAL, иначе как в ранжировании."""
    return settings.pp_dedup_refusal or dedup_rank(settings)


def preset_key(settings) -> tuple:
    return (settings.pp_dba_k, dedup_rank(settings), dedup_refusal(settings), settings.refusal_score)


def enabled(settings) -> bool:
    return (settings.pp_dba_k > 0 or settings.pp_camera_penalty != "off" or settings.pp_dedup != "off"
            or bool(settings.pp_dedup_rank) or bool(settings.pp_dedup_refusal))


def validate(settings) -> List[str]:
    errors = []
    if settings.pp_dba_k < 0:
        errors.append(f"PP_DBA_K должен быть ≥ 0, получено {settings.pp_dba_k}")
    if settings.pp_camera_penalty not in CAMERA_MODES:
        errors.append(f"PP_CAMERA_PENALTY: одно из {CAMERA_MODES}, получено {settings.pp_camera_penalty!r}")
    if not 1 <= settings.pp_camera_topk <= 1000:
        errors.append(f"PP_CAMERA_TOPK должен быть от 1 до 1000, получено {settings.pp_camera_topk}")
    if not -1.0 <= settings.pp_camera_tau <= 1.0:
        errors.append(f"PP_CAMERA_TAU должен быть от -1 до 1, получено {settings.pp_camera_tau}")
    for name, v in (("PP_DEDUP", settings.pp_dedup), ("PP_DEDUP_RANK", settings.pp_dedup_rank or "off"),
                    ("PP_DEDUP_REFUSAL", settings.pp_dedup_refusal or "off")):
        if v not in DEDUP_MODES:
            errors.append(f"{name}: одно из {DEDUP_MODES}, получено {v!r}")
    if settings.pp_camera_penalty == "on" and settings.pp_dedup not in ("off", "bg", "both"):
        errors.append(f"PP_CAMERA_PENALTY=on (= PP_DEDUP=bg) противоречит PP_DEDUP={settings.pp_dedup}")
    if not -1.0 <= settings.pp_dedup_sim <= 1.0 or not 0.0 <= settings.pp_dedup_iou <= 1.0:
        errors.append("PP_DEDUP_SIM должен быть от -1 до 1, PP_DEDUP_IOU — от 0 до 1")
    if settings.pp_dedup_center < 0 or settings.pp_dedup_size < 0:
        errors.append("PP_DEDUP_CENTER и PP_DEDUP_SIZE должны быть ≥ 0")
    return errors


class Postprocess:
    """store — модуль app.db (или подделка в тестах) с функциями ensure_pp_schema,
    fetch_pp, update_dba, insert_pp, search_pp, delete_pp."""

    def __init__(self, settings, dim: int, store):
        self.k = settings.pp_dba_k
        self.dedup_rank, self.dedup_refusal = dedup_rank(settings), dedup_refusal(settings)
        modes = {self.dedup_rank, self.dedup_refusal}
        self.use_bg = bool(modes & {"bg", "both"})
        self.use_bbox = bool(modes & {"bbox", "both"})
        self.store_bg = settings.pp_camera_penalty in ("store", "on") or self.use_bg
        self.rule = {"s_any": settings.pp_dedup_sim_any, "s_hi": settings.pp_dedup_sim,
                     "t_iou": settings.pp_dedup_iou, "t_center": settings.pp_dedup_center,
                     "t_size": settings.pp_dedup_size}
        self.tau = settings.pp_camera_tau
        self.topk = settings.pp_camera_topk
        self.formula = settings.refusal_score
        self.dim = dim
        self.store = store
        self._lock = threading.RLock()
        store.ensure_pp_schema(dim, bgm.BG_DIM)
        self.reload()

    @property
    def column(self) -> str:
        return "embedding_dba" if self.k else "embedding"

    @property
    def wants_frame(self) -> bool:
        """Нужен ли полный кадр (признак фона) при добавлении и поиске."""
        return self.store_bg

    # ---------------------------------------------------------------- состояние
    def reload(self) -> None:
        with self._lock:
            rows = self.store.fetch_pp(self.use_bg)
            self.ids = [int(r["id"]) for r in rows]
            self.row = {i: n for n, i in enumerate(self.ids)}
            if self.k:
                g = (np.stack([np.asarray(r["embedding"], "float32") for r in rows]) if rows
                     else np.zeros((0, self.dim), "float32"))
                self.dba = DBAGallery(g, k=self.k)
                stale = [(r["id"], v) for r, v in zip(rows, self.dba.vectors)
                         if r["embedding_dba"] is None or np.abs(np.asarray(r["embedding_dba"]) - v).max() > 1e-5]
                if stale:
                    self.store.update_dba(stale)
                log.info("DBA k=%d: галерея %d кадров, g' дописано/исправлено %d", self.k, len(rows), len(stale))
            self.bg = ({int(r["id"]): np.asarray(r["bg"], "float16") for r in rows if r["bg"] is not None}
                       if self.use_bg else {})
            if self.use_bg:
                log.info("штраф за ту же камеру: признак фона есть у %d кадров из %d", len(self.bg), len(rows))

    def _resync(self, exc: Exception) -> None:
        log.error("постобработка: запись не удалась (%s), перечитываю состояние из базы", exc)
        self.reload()

    # ---------------------------------------------------------------- запись
    def add_many(self, items: Sequence[Tuple[Optional[str], str, Sequence[int], np.ndarray, Any]]) -> List[int]:
        """items: (vehicle_id, image_path, bbox, vector, полный кадр или None). -> новые id."""
        if not items:
            return []
        bgs = [bgm.service_descriptor(frame, bbox) if self.wants_frame and frame is not None else None
               for _, _, bbox, _, frame in items]
        with self._lock:
            try:
                n0, dba_new, updates = len(self.ids), [None] * len(items), []
                if self.k:
                    changed = self.dba.add(np.stack([np.asarray(it[3], "float32") for it in items]))
                    V = self.dba.vectors
                    updates = [(self.ids[i], V[i]) for i in changed if i < n0]
                    dba_new = [V[n0 + j] for j in range(len(items))]
                rows = [(vid, path, list(map(int, bbox)), vec, dba_new[j], bgs[j])
                        for j, (vid, path, bbox, vec, _) in enumerate(items)]
                new_ids = [int(i) for i in self.store.insert_pp(rows, updates)]
            except Exception as exc:
                self._resync(exc)
                raise
            self.ids.extend(new_ids)
            self.row.update({i: len(self.ids) - len(new_ids) + j for j, i in enumerate(new_ids)})
            if self.use_bg:
                self.bg.update({i: b.astype("float16") for i, b in zip(new_ids, bgs) if b is not None})
            return new_ids

    def delete(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self.ids:
                return self.store.delete_pp(item_id, [])
            row = self.ids.index(item_id)
            try:
                updates = []
                if self.k:
                    changed = self.dba.remove([row])
                    rest = self.ids[:row] + self.ids[row + 1:]
                    updates = [(rest[i], self.dba.vectors[i]) for i in changed]
                ok = self.store.delete_pp(item_id, updates)
            except Exception as exc:
                self._resync(exc)
                raise
            self.ids.pop(row)
            self.row = {i: n for n, i in enumerate(self.ids)}
            self.bg.pop(item_id, None)
            return ok

    # ---------------------------------------------------------------- поиск
    def search(self, vec: np.ndarray, top_n: int, frame=None, bbox=None) -> Tuple[List[Dict[str, Any]], float]:
        """-> (top_n кандидатов в порядке ответа, оценка уверенности для отказа)."""
        k = rr.needed_k(top_n, self.use_bg or self.use_bbox, self.topk, self.formula)
        rows = self.store.search_pp(vec, k, self.column, rr.ef_search(self.topk))
        cos = np.array([r["confidence"] for r in rows], dtype="float64")
        fb = fx = np.zeros(len(rows), bool)
        if self.use_bg and frame is not None and rows:
            fb = rr.camera_flags(bgm.service_descriptor(frame, bbox), [self.bg.get(int(r["id"])) for r in rows],
                                 self.tau)
        if self.use_bbox and bbox is not None and rows:
            fx = dd.near_dup_flags(self._raw_cos(vec, rows, cos), np.asarray(bbox, "float64"),
                                   np.array([r["bbox"] for r in rows], dtype="float64"), **self.rule)
        pick = {"off": np.zeros(len(rows), bool), "bg": fb, "bbox": fx, "both": fb | fx}
        order = rr.rank(cos, pick[self.dedup_rank])                     # порядок ответа
        f_ref = pick[self.dedup_refusal]
        o_ref = rr.rank(cos, f_ref)                                      # отказ — по своим исключениям
        score = rr.refusal_score(cos[o_ref], f_ref[o_ref], self.formula)
        found = [{**rows[i], "confidence": round(float(cos[i]), 4)} for i in order[:top_n]]
        return found, score

    def _raw_cos(self, vec, rows, cos) -> np.ndarray:
        """Сырой косинус запроса с кандидатами (без DBA) — для правила почти-дубля.
        Без DBA это и есть скор поиска; с DBA — по исходным векторам из памяти; кадр, которого
        ещё нет в памяти, почти-дублем не считается."""
        if not self.k:
            return cos
        q = np.asarray(vec, "float32")
        q = q / (np.linalg.norm(q) + 1e-12)
        g = self.dba.originals
        return np.array([float(g[self.row[int(r["id"])]] @ q) if int(r["id"]) in self.row else -np.inf
                         for r in rows])
