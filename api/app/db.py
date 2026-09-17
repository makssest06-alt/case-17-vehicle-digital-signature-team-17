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
