"""Постобработка галереи в сервисе: DBA и штраф за ту же камеру. По умолчанию выключена.

Включается флагами (config.py, блок «постобработка»):
  PP_DBA_K           0 — выкл; k > 0 — кадры галереи ищутся по g' = DBA(g, k)
                     (research/HONEST_EVAL.md). g' лежат в колонке embedding_dba,
                     исходные embedding не трогаются;
  PP_CAMERA_PENALTY  off — выкл; store — при добавлении кадра считать и хранить
                     признак фона (колонка bg), в поиске не использовать; on —
                     хранить и штрафовать: кандидат «с той же камеры» (косинус
                     фона > PP_CAMERA_TAU) уходит в конец списка.
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
from . import rerank as rr
from .dba import DBAGallery

log = logging.getLogger("case17.pp")
CAMERA_MODES = ("off", "store", "on")


def enabled(settings) -> bool:
    return settings.pp_dba_k > 0 or settings.pp_camera_penalty != "off"


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
    return errors


class Postprocess:
    """store — модуль app.db (или подделка в тестах) с функциями ensure_pp_schema,
    fetch_pp, update_dba, insert_pp, search_pp, delete_pp."""

    def __init__(self, settings, dim: int, store):
        self.k = settings.pp_dba_k
        self.camera = settings.pp_camera_penalty
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
        return self.camera != "off"

    # ---------------------------------------------------------------- состояние
    def reload(self) -> None:
        with self._lock:
            rows = self.store.fetch_pp(self.camera == "on")
            self.ids = [int(r["id"]) for r in rows]
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
                       if self.camera == "on" else {})
            if self.camera == "on":
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
            if self.camera == "on":
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
            self.bg.pop(item_id, None)
            return ok

    # ---------------------------------------------------------------- поиск
    def search(self, vec: np.ndarray, top_n: int, frame=None, bbox=None) -> Tuple[List[Dict[str, Any]], float]:
        """-> (top_n кандидатов в порядке ответа, оценка уверенности для отказа)."""
        k = rr.needed_k(top_n, self.camera == "on", self.topk, self.formula)
        rows = self.store.search_pp(vec, k, self.column)
        cos = np.array([r["confidence"] for r in rows], dtype="float64")
        flags = np.zeros(len(rows), bool)
        if self.camera == "on" and frame is not None and rows:
            flags = rr.camera_flags(bgm.service_descriptor(frame, bbox), [self.bg.get(int(r["id"])) for r in rows],
                                    self.tau)
        order = rr.rank(cos, flags)
        score = rr.refusal_score(cos[order], flags[order], self.formula)
        found = [{**rows[i], "confidence": round(float(cos[i]), 4)} for i in order[:top_n]]
        return found, score
