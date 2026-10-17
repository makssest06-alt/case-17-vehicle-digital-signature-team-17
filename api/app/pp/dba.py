"""DBA галереи (database-side augmentation): функция для сервиса.

Каждый кадр галереи заменяется на L2(g_i + Σ_j w_ij·g_j) по k ближайшим (по
косинусу) ДРУГИМ кадрам галереи. Запрос не меняется, поиск — тот же q · g'.
Веса: weights="cos" — w = max(cos, 0) (так мерилось), "uniform" — w = 1.
Замер (research/bigallery_pp.py, k=1, weights="cos", кадры без почти-дублей
запроса): +0.059 / +0.049 / +0.060 mAP@10 на галереях 9.0k / 10.8k / 8996.

Соседи ищутся по ИСХОДНЫМ эмбеддингам, поэтому сервису нужно хранить и их, и g'.
Только numpy. Вход — эмбеддинги галереи (N, D); нормируются внутри.

    from research.dba import dba, DBAGallery
    g2 = dba(gallery, k=1)              # один пересчёт при старте
    st = DBAGallery(gallery, k=1)       # с состоянием — для /v1/gallery
    changed = st.add(new_vectors)       # строки st.vectors, которые поменялись
    changed = st.remove(rows)           # то же после удаления (новая нумерация)
"""
import numpy as np

WEIGHTS = ("cos", "uniform")


def l2(x: np.ndarray) -> np.ndarray:
    """Та же нормировка, что reid_postprocess.l2: замеры bigallery_pp не меняются."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def neighbours(g: np.ndarray, n: int, pool: np.ndarray = None, chunk: int = 2048) -> tuple:
    """n ближайших из pool (индексы g, по умолчанию все) для каждого кадра g, без
    самого кадра, по убыванию косинуса. -> (idx (N, n), sim (N, n))."""
    pool = np.arange(len(g)) if pool is None else np.asarray(pool)
    P = g[pool]
    idx = np.empty((len(g), n), int)
    sim = np.empty((len(g), n), "float32")
    if n == 0:
        return idx, sim
    for s in range(0, len(g), chunk):
        S = g[s:s + chunk] @ P.T
        self_hit = pool[None, :] == np.arange(s, s + len(S))[:, None]
        S[self_hit] = -np.inf
        part = np.argpartition(-S, n - 1, axis=1)[:, :n]
        ps = np.take_along_axis(S, part, 1)
        o = np.argsort(-ps, axis=1)
        idx[s:s + chunk] = pool[np.take_along_axis(part, o, 1)]
        sim[s:s + chunk] = np.take_along_axis(ps, o, 1)
    return idx, sim


def _weights(sim: np.ndarray, weights: str) -> np.ndarray:
    if weights == "cos":
        return np.maximum(sim, 0.0)
    if weights == "uniform":
        return np.ones_like(sim)
    raise ValueError(f"weights: одно из {WEIGHTS}, получено {weights!r}")


def dba_vectors(g: np.ndarray, idx: np.ndarray, sim: np.ndarray, k: int, weights: str = "cos") -> np.ndarray:
    """g' по готовым соседям (первые k столбцов idx, sim)."""
    w = _weights(sim[:, :k], weights)
    return l2(g + np.einsum("ik,ikd->id", w, g[idx[:, :k]]))


def dba(gallery: np.ndarray, k: int = 1, weights: str = "cos", chunk: int = 2048) -> np.ndarray:
    """Новые эмбеддинги галереи (N, D), L2-нормированные; вход не меняется."""
    _weights(np.zeros(1, "float32"), weights)
    g = l2(np.asarray(gallery, dtype="float32"))
    kk = min(k, len(g) - 1)
    if kk <= 0:
        return g
    idx, sim = neighbours(g, kk, chunk=chunk)
    return dba_vectors(g, idx, sim, kk, weights)


class DBAGallery:
    """DBA для меняющейся галереи (POST и DELETE /v1/gallery).

    Хранит исходные векторы, k соседей каждого кадра и g'. add и remove ищут
    соседей только для задетых строк (кадр в галерее из N — одно умножение N×D
    на D), g' пересчитываются для всех строк: 9 мс на кадр при N = 10.8k на CPU
    Мака. Итог совпадает с dba() на той же галерее (с точностью до ничьих по
    косинусу). Возвращаемые строки — надмножество поменявшихся (сосед с весом 0
    тоже попадает). Не потокобезопасен: вызывать под замком индекса. Большой
    импорт (тысячи кадров разом) — проще DBAGallery(все векторы) заново.
    """

    def __init__(self, gallery: np.ndarray, k: int = 1, weights: str = "cos"):
        _weights(np.zeros(1, "float32"), weights)
        g = np.asarray(gallery, dtype="float32")
        if g.ndim != 2:
            raise ValueError(f"галерея: нужен массив (N, D), получено {g.shape}")
        self.k, self.weights = k, weights
        self._set(l2(g))

    def __len__(self) -> int:
        return len(self._g)

    @property
    def vectors(self) -> np.ndarray:
        """Текущие g' (только чтение)."""
        return self._v

    @property
    def originals(self) -> np.ndarray:
        """Исходные L2-векторы галереи в том же порядке строк (только чтение)."""
        v = self._g.view()
        v.setflags(write=False)
        return v

    def _kk(self, n: int) -> int:
        return max(min(self.k, n - 1), 0)

    def _set(self, g: np.ndarray, idx: np.ndarray = None, sim: np.ndarray = None) -> None:
        kk = self._kk(len(g))
        if idx is None:
            idx, sim = neighbours(g, kk)
        v = dba_vectors(g, idx, sim, kk, self.weights) if kk else g.copy()
        v.setflags(write=False)
        self._g, self._idx, self._sim, self._v = g, idx, sim, v

    def add(self, new: np.ndarray) -> np.ndarray:
        """Добавить кадры (m, D) или (D,) в конец. -> индексы строк vectors, которые поменялись."""
        new = np.asarray(new, dtype="float32")
        new = new[None] if new.ndim == 1 else new
        if new.ndim != 2 or new.shape[1] != self._g.shape[1]:
            raise ValueError(f"add: нужен массив (m, {self._g.shape[1]}), получено {new.shape}")
        new = l2(new)
        n0, m = len(self._g), len(new)
        g = np.concatenate([self._g, new])
        kk = self._kk(len(g))
        if kk != self._idx.shape[1]:                         # галерея меньше k: пересчёт целиком
            self._set(g)
            return np.arange(len(g))
        S = self._g @ new.T                                  # старые кадры против новых
        cand_idx = np.concatenate([self._idx, np.broadcast_to(n0 + np.arange(m), (n0, m))], 1)
        cand_sim = np.concatenate([self._sim, S], 1)
        o = np.argsort(-cand_sim, axis=1, kind="stable")[:, :kk]
        idx_old, sim_old = np.take_along_axis(cand_idx, o, 1), np.take_along_axis(cand_sim, o, 1)
        moved = np.flatnonzero((idx_old != self._idx).any(1))
        idx_new, sim_new = self._rows_neighbours(g, n0 + np.arange(m), kk)
        self._set(g, np.concatenate([idx_old, idx_new]), np.concatenate([sim_old, sim_new]))
        return np.concatenate([moved, n0 + np.arange(m)])

    def remove(self, rows) -> np.ndarray:
        """Удалить строки rows (номера, не маска). -> индексы в НОВОЙ нумерации, у которых
        поменялся g'. Строки после удалённых сдвигаются на их число: хранилище с
        доступом по номеру строки надо перенумеровать целиком."""
        r = np.asarray(rows)
        if r.size == 0:
            return np.array([], int)
        if r.dtype == bool or not np.issubdtype(r.dtype, np.integer):
            raise ValueError("remove: нужны номера строк, а не маска")
        if r.min() < 0 or r.max() >= len(self._g):
            raise ValueError(f"remove: номер вне 0..{len(self._g) - 1}")
        gone = np.zeros(len(self._g), bool)
        gone[r] = True
        g = self._g[~gone]
        kk = self._kk(len(g))
        if kk != self._idx.shape[1]:
            self._set(g)
            return np.arange(len(g))
        renum = np.cumsum(~gone) - 1
        idx, sim = self._idx[~gone], self._sim[~gone]
        hit = np.flatnonzero(gone[idx].any(1))              # среди соседей был удалённый кадр
        idx = renum[idx]
        idx_hit, sim_hit = self._rows_neighbours(g, hit, kk)
        idx, sim = idx.copy(), sim.copy()
        idx[hit], sim[hit] = idx_hit, sim_hit
        self._set(g, idx, sim)
        return hit

    @staticmethod
    def _rows_neighbours(g: np.ndarray, rows: np.ndarray, n: int) -> tuple:
        S = g[rows] @ g.T
        S[np.arange(len(rows)), rows] = -np.inf
        o = np.argsort(-S, axis=1, kind="stable")[:, :n]
        return o, np.take_along_axis(S, o, 1)
