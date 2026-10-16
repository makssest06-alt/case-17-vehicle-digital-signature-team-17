"""Галерея в памяти: матрица векторов + точный поиск матричным умножением.

pgvector остаётся хранилищем — он переживает перезапуск и хранит метаданные.
Но на горячем пути его нет: при старте (и после каждой записи) векторы
поднимаются в один numpy-массив, и поиск становится одним matmul.

Для галереи в 750 записей это буквально 750x512 float32 = 1.5 МБ,
и умножение занимает десятки микросекунд против сетевого round-trip к базе.
"""
import threading
from typing import Any, Dict, List, Optional

import numpy as np


class GalleryIndex:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._vecs: Optional[np.ndarray] = None   # (N, D), L2-нормированные
        self._meta: List[Dict[str, Any]] = []
        self._dirty = True

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._meta)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        self._dirty = True

    def rebuild(self, rows: List[Dict[str, Any]]) -> int:
        """rows: [{id, vehicle_id, image_path, bbox, embedding(np.ndarray)}]"""
        with self._lock:
            if not rows:
                self._vecs, self._meta, self._dirty = None, [], False
                return 0
            vecs = np.stack([np.asarray(r["embedding"], dtype="float32") for r in rows])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._vecs = np.ascontiguousarray(vecs / norms)
            self._meta = [
                {k: r[k] for k in ("id", "vehicle_id", "image_path", "bbox")} for r in rows
            ]
            self._dirty = False
            return len(self._meta)

    def search(self, vec: np.ndarray, top_n: int) -> List[Dict[str, Any]]:
        with self._lock:
            if self._vecs is None or not len(self._meta):
                return []
            q = np.asarray(vec, dtype="float32")
            n = np.linalg.norm(q)
            if n:
                q = q / n
            sims = self._vecs @ q
            k = min(top_n, len(sims))
            # argpartition вместо полной сортировки: O(N) вместо O(N log N)
            part = np.argpartition(-sims, k - 1)[:k]
            order = part[np.argsort(-sims[part])]
            return [
                {**self._meta[i], "confidence": round(float(sims[i]), 4)} for i in order
            ]


index = GalleryIndex()
