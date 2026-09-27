#!/bin/bash
set -e
DIM="${EMBED_DIM:-512}"
echo "[init] создаю схему, размерность вектора = $DIM"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS gallery (
    id          BIGSERIAL PRIMARY KEY,
    vehicle_id  TEXT,
    image_path  TEXT        NOT NULL,
    bbox        INTEGER[]   NOT NULL CHECK (array_length(bbox, 1) = 4),
    embedding   VECTOR($DIM) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS gallery_vehicle_id_idx ON gallery (vehicle_id);
CREATE INDEX IF NOT EXISTS gallery_created_at_idx ON gallery (created_at DESC);
CREATE INDEX IF NOT EXISTS gallery_embedding_idx
    ON gallery USING hnsw (embedding vector_cosine_ops);
SQL
echo "[init] схема готова"
