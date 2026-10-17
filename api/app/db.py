"""Работа с Postgres + pgvector."""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import settings

_pool: Optional[ConnectionPool] = None


def _configure(conn) -> None:
    register_vector(conn)


def init_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.db_dsn,
            min_size=1,
            max_size=4,
            configure=_configure,
            open=True,
            timeout=30,
        )
    return _pool


def pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("пул соединений не инициализирован")
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def ensure_schema(dim: int) -> None:
    """Идемпотентно создать расширение и таблицу под нужную размерность."""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS gallery (
                id          BIGSERIAL PRIMARY KEY,
                vehicle_id  TEXT,
                image_path  TEXT        NOT NULL,
                bbox        INTEGER[]   NOT NULL CHECK (array_length(bbox, 1) = 4),
                embedding   VECTOR({dim}) NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS gallery_vehicle_id_idx ON gallery (vehicle_id)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS gallery_embedding_idx "
            "ON gallery USING hnsw (embedding vector_cosine_ops)"
        )
        conn.commit()


def stored_dim() -> Optional[int]:
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = 'gallery'::regclass AND attname = 'embedding'
            """
        )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] and row[0] > 0 else None


def insert_one(vehicle_id: Optional[str], image_path: str, bbox: Sequence[int], vec: np.ndarray) -> int:
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO gallery (vehicle_id, image_path, bbox, embedding) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (vehicle_id, image_path, list(map(int, bbox)), vec),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return int(new_id)


def insert_many(rows: List[tuple]) -> int:
    """rows: (vehicle_id, image_path, bbox_list, vector)."""
    if not rows:
        return 0
    with pool().connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO gallery (vehicle_id, image_path, bbox, embedding) VALUES (%s, %s, %s, %s)",
            rows,
        )
        conn.commit()
        return len(rows)


def search(vec: np.ndarray, top_n: int) -> List[Dict[str, Any]]:
    """Ближайшие соседи по косинусу. confidence = 1 - косинусное расстояние."""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, vehicle_id, image_path, bbox,
                   1 - (embedding <=> %s) AS confidence
            FROM gallery
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (vec, vec, top_n),
        )
        return [
            {
                "id": r[0],
                "vehicle_id": r[1],
                "image_path": r[2],
                "bbox": list(r[3]),
                "confidence": round(float(r[4]), 4),
            }
            for r in cur.fetchall()
        ]


def list_items(limit: int, offset: int) -> List[Dict[str, Any]]:
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, vehicle_id, image_path, bbox, created_at FROM gallery "
            "ORDER BY id LIMIT %s OFFSET %s",
            (limit, offset),
        )
        return [
            {
                "id": r[0],
                "vehicle_id": r[1],
                "image_path": r[2],
                "bbox": list(r[3]),
                "created_at": r[4].isoformat(),
            }
            for r in cur.fetchall()
        ]


def fetch_vector(item_id: int):
    """Вектор одной записи галереи — нужен, чтобы объяснить конкретное совпадение."""
    import numpy as np

    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT embedding FROM gallery WHERE id = %s", (item_id,))
        row = cur.fetchone()
    return None if row is None else np.asarray(row[0], dtype="float32")


def fetch_all() -> List[Dict[str, Any]]:
    """Вся галерея разом — для построения индекса в памяти (gallery_index).

    Горячий путь поиска после этого идёт через numpy, а pgvector остаётся
    хранилищем: переживает перезапуск и хранит метаданные. Не заменяет
    fetch_vector — тот нужен /v1/explain для одной записи.
    """
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, vehicle_id, image_path, bbox, embedding FROM gallery ORDER BY id"
        )
        return [
            {
                "id": r[0],
                "vehicle_id": r[1],
                "image_path": r[2],
                "bbox": list(r[3]),
                "embedding": np.asarray(r[4], dtype="float32"),
            }
            for r in cur.fetchall()
        ]


def count() -> int:
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM gallery")
        return int(cur.fetchone()[0])


def delete_one(item_id: int) -> bool:
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM gallery WHERE id = %s", (item_id,))
        deleted = cur.rowcount
        conn.commit()
        return deleted > 0


# ---------------------------------------------------------------------------
# Постобработка (api/app/pp/state.py). Вызывается только при включённых флагах
# PP_*: без них схема и запросы — как выше.
# ---------------------------------------------------------------------------

PP_COLUMNS = ("embedding", "embedding_dba")


def ensure_pp_schema(dim: int, bg_dim: int) -> None:
    """Обратно совместимо: только новые колонки с NULL и индекс по ним."""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(f"ALTER TABLE gallery ADD COLUMN IF NOT EXISTS embedding_dba VECTOR({dim})")
        cur.execute(f"ALTER TABLE gallery ADD COLUMN IF NOT EXISTS bg VECTOR({bg_dim})")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS gallery_embedding_dba_idx "
            "ON gallery USING hnsw (embedding_dba vector_cosine_ops)"
        )
        conn.commit()


def fetch_pp(with_bg: bool) -> List[Dict[str, Any]]:
    cols = "id, embedding, embedding_dba" + (", bg" if with_bg else "")
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {cols} FROM gallery ORDER BY id")
        return [
            {
                "id": int(r[0]),
                "embedding": np.asarray(r[1], dtype="float32"),
                "embedding_dba": None if r[2] is None else np.asarray(r[2], dtype="float32"),
                "bg": None if not with_bg or r[3] is None else np.asarray(r[3], dtype="float32"),
            }
            for r in cur.fetchall()
        ]


def _update_dba(cur, pairs) -> None:
    if pairs:
        cur.executemany(
            "UPDATE gallery SET embedding_dba = %s WHERE id = %s",
            [(np.asarray(v, dtype="float32"), int(i)) for i, v in pairs],
        )


def update_dba(pairs) -> None:
    """pairs: [(id, g')]."""
    with pool().connection() as conn, conn.cursor() as cur:
        _update_dba(cur, pairs)
        conn.commit()


def insert_pp(rows, dba_updates) -> List[int]:
    """rows: (vehicle_id, image_path, bbox, embedding, embedding_dba | None, bg | None).
    Новые кадры и обновлённые g' соседей — одной транзакцией."""
    ids = []
    with pool().connection() as conn, conn.cursor() as cur:
        for vehicle_id, image_path, bbox, vec, dba_vec, bg in rows:
            cur.execute(
                "INSERT INTO gallery (vehicle_id, image_path, bbox, embedding, embedding_dba, bg) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (vehicle_id, image_path, list(map(int, bbox)), vec,
                 None if dba_vec is None else np.asarray(dba_vec, dtype="float32"),
                 None if bg is None else np.asarray(bg, dtype="float32")),
            )
            ids.append(int(cur.fetchone()[0]))
        _update_dba(cur, dba_updates)
        conn.commit()
    return ids


def search_pp(vec: np.ndarray, k: int, column: str, ef_search: int) -> List[Dict[str, Any]]:
    """k ближайших по колонке column; confidence — косинус без округления.
    SET LOCAL hnsw.ef_search — в транзакции этого поиска: по умолчанию (40) HNSW
    отдаёт не больше ~40 строк, а штрафу и z100 нужен top-100. Вызывающий передаёт
    max(100, PP_CAMERA_TOPK)."""
    if column not in PP_COLUMNS:
        raise ValueError(f"колонка поиска: одна из {PP_COLUMNS}")
    ef = int(ef_search)
    if not 1 <= ef <= 1000:
        raise ValueError(f"hnsw.ef_search должен быть от 1 до 1000, получено {ef}")
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")
        cur.execute(
            f"""
            SELECT id, vehicle_id, image_path, bbox, 1 - ({column} <=> %s) AS confidence
            FROM gallery
            WHERE {column} IS NOT NULL
            ORDER BY {column} <=> %s
            LIMIT %s
            """,
            (vec, vec, int(k)),
        )
        rows = cur.fetchall()
        conn.commit()
    return [
        {"id": r[0], "vehicle_id": r[1], "image_path": r[2], "bbox": list(r[3]), "confidence": float(r[4])}
        for r in rows
    ]


def delete_pp(item_id: int, dba_updates) -> bool:
    """Удаление кадра и обновлённые g' его бывших соседей — одной транзакцией."""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM gallery WHERE id = %s", (item_id,))
        deleted = cur.rowcount
        _update_dba(cur, dba_updates)
        conn.commit()
        return deleted > 0
