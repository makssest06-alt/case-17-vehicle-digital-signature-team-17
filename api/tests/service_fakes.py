"""Поддельные модель и база и вызов ASGI-приложения без сети и без httpx.

Нужны тестам сервиса (fastapi, pydantic, numpy, Pillow из api/requirements.txt,
torch не нужен): запрос уходит прямо в app(scope, receive, send), ответ
возвращается байтами — так сравнение «как было / как стало» точное.

Время (time.perf_counter) подменяется нулём на время запроса: took_ms и
X-Process-Time-Ms детерминированы и не зависят от числа вызовов таймера.
"""
import asyncio
import dataclasses
import io
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

DIM = 8
FRAME = (320, 180)


def png(seed: int, size=FRAME) -> bytes:
    a = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


class FakeModel:
    """Вектор — уменьшенный серый кроп, детерминированно и без torch."""
    name = "fake"
    dim = DIM

    def embed_batch(self, crops):
        out = []
        for im in crops:
            g = np.asarray(im.convert("L").resize((4, 2), Image.BILINEAR), dtype="float32").ravel()
            g = g - g.mean()
            out.append(g / (np.linalg.norm(g) + 1e-6))
        return np.stack(out).astype("float32")


class FakeDB:
    """Таблица gallery в памяти с теми же функциями, что app/db.py. Пишет журнал вызовов."""

    def __init__(self, n: int = 30, seed: int = 0):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, DIM)).astype("float32")
        self.rows = [{"id": i + 1, "vehicle_id": str(100 + i % 7), "image_path": f"g{i}.jpg", "bbox": [1, 2, 3, 4],
                      "embedding": v[i] / np.linalg.norm(v[i]), "embedding_dba": None, "bg": None}
                     for i in range(n)]
        self.next_id = n + 1
        self.calls = []
        self.schema = []

    # ---- функции app/db.py
    def count(self):
        self.calls.append(("count",))
        return len(self.rows)

    def search(self, vec, top_n):
        self.calls.append(("search", rnd(vec), int(top_n)))
        return [{k: r[k] for k in ("id", "vehicle_id", "image_path", "bbox", "confidence")}
                for r in self._ranked(vec, "embedding")[:top_n]]

    def insert_one(self, vehicle_id, image_path, bbox, vec):
        self.calls.append(("insert_one", vehicle_id, image_path, list(map(int, bbox)), rnd(vec)))
        return self._insert(vehicle_id, image_path, bbox, vec)

    def insert_many(self, rows):
        self.calls.append(("insert_many", [(r[0], norm_path(r[1]), list(r[2]), rnd(r[3])) for r in rows]))
        for r in rows:
            self._insert(r[0], norm_path(r[1]), r[2], r[3])       # путь временного DATA_DIR — в ответ не тащим
        return len(rows)

    def delete_one(self, item_id):
        self.calls.append(("delete_one", int(item_id)))
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["id"] != item_id]
        return len(self.rows) < before

    # ---- функции постобработки (app/db.py, только при включённых флагах)
    def ensure_pp_schema(self, dim, bg_dim):
        self.schema.append(("ensure_pp_schema", dim, bg_dim))

    def fetch_pp(self, with_bg):
        return [{"id": r["id"], "embedding": r["embedding"], "embedding_dba": r["embedding_dba"],
                 "bg": r["bg"] if with_bg else None} for r in self.rows]

    def update_dba(self, pairs):
        self.calls.append(("update_dba", sorted(int(i) for i, _ in pairs)))
        by = {r["id"]: r for r in self.rows}
        for i, v in pairs:
            by[i]["embedding_dba"] = np.asarray(v, "float32")

    def insert_pp(self, rows, dba_updates):
        ids = []
        for vehicle_id, image_path, bbox, vec, dba_vec, bg in rows:
            new_id = self._insert(vehicle_id, image_path, bbox, vec)
            self.rows[-1]["embedding_dba"] = None if dba_vec is None else np.asarray(dba_vec, "float32")
            self.rows[-1]["bg"] = None if bg is None else np.asarray(bg, "float32")
            ids.append(new_id)
        self.update_dba(dba_updates)
        self.calls.append(("insert_pp", ids))
        return ids

    def search_pp(self, vec, k, column):
        self.calls.append(("search_pp", int(k), column))
        return self._ranked(vec, column)[:k]

    def delete_pp(self, item_id, dba_updates):
        ok = self.delete_one(item_id)
        if ok:
            self.update_dba(dba_updates)
        return ok

    # ---- внутреннее
    def _insert(self, vehicle_id, image_path, bbox, vec):
        new_id = self.next_id
        self.next_id += 1
        self.rows.append({"id": new_id, "vehicle_id": vehicle_id, "image_path": image_path, "bbox": list(map(int, bbox)),
                          "embedding": np.asarray(vec, "float32"), "embedding_dba": None, "bg": None})
        return new_id

    def _ranked(self, vec, column):
        q = np.asarray(vec, "float32")
        rows = [r for r in self.rows if r[column] is not None]
        sims = [float(np.dot(q, r[column]) / (np.linalg.norm(q) * np.linalg.norm(r[column]) + 1e-12)) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: -sims[i])
        return [{"id": rows[i]["id"], "vehicle_id": rows[i]["vehicle_id"], "image_path": rows[i]["image_path"],
                 "bbox": list(rows[i]["bbox"]), "confidence": round(sims[i], 4)} for i in order]


def rnd(v):
    return [round(float(x), 6) for x in np.asarray(v).ravel()]


def norm_path(p):
    root = os.environ.get("DATA_DIR", "")
    return str(p).replace(root, "<data>") if root else str(p)


def multipart(fields: dict, files: dict) -> tuple:
    boundary = "fakeboundary7MA4YWxkTrZu0gW"
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (name, data) in files.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\n"
                  f"Content-Type: image/png\r\n\r\n".encode())
        out.write(data + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def call(app, method: str, path: str, fields=None, files=None, json_body=None) -> dict:
    """Один запрос в ASGI-приложение. -> {status, headers, body}."""
    if json_body is not None:
        body, ctype = json.dumps(json_body).encode(), "application/json"
    elif fields is not None or files is not None:
        body, ctype = multipart(fields or {}, files or {})
    else:
        body, ctype = b"", None
    headers = [(b"host", b"test"), (b"content-length", str(len(body)).encode())]
    if ctype:
        headers.append((b"content-type", ctype.encode()))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "", "headers": headers,
             "client": ("test", 1), "server": ("test", 80)}
    inbox = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        return inbox.pop(0) if inbox else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    with patch("time.perf_counter", return_value=0.0):
        asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    hdr = {k.decode(): v.decode() for k, v in start["headers"]}
    return {"status": start["status"], "headers": {k: hdr[k] for k in sorted(hdr) if k != "date"},
            "body": b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()}


@contextmanager
def data_dir():
    """DATA_DIR с тремя кропами и csv — для bulk-import."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "crops").mkdir()
        for i in range(3):
            (root / "crops" / f"c{i}.jpg").write_bytes(png(50 + i, (64, 48)))
        (root / "g.csv").write_text("image_id,vehicle_id\nc0,7\nc1,8\nc2,\nmissing,9\n", encoding="utf-8")
        with patch.dict(os.environ, {"DATA_DIR": str(root)}):      # norm_path прячет временный путь
            yield root


REQUESTS = (
    ("search_accept", "POST", "/v1/search", {"x": 40, "y": 30, "w": 120, "h": 90, "threshold": 0.0}, 1),
    ("search_default", "POST", "/v1/search", {"x": 40, "y": 30, "w": 120, "h": 90}, 2),
    ("search_top3_refuse", "POST", "/v1/search", {"x": 10, "y": 10, "w": 60, "h": 60, "top_n": 3, "threshold": 0.99}, 3),
    ("search_bad_bbox", "POST", "/v1/search", {"x": 300, "y": 30, "w": 120, "h": 90}, 4),
    ("gallery_add", "POST", "/v1/gallery", {"x": 40, "y": 30, "w": 120, "h": 90, "vehicle_id": "555"}, 5),
    ("search_after_add", "POST", "/v1/search", {"x": 40, "y": 30, "w": 120, "h": 90, "threshold": 0.0}, 5),
    ("delete", "DELETE", "/v1/gallery/3", None, None),
    ("delete_missing", "DELETE", "/v1/gallery/999", None, None),
    ("config", "GET", "/v1/config", None, None),
)


def run_requests(app, main_module, fake_db, root: Path, overrides: dict = None) -> list:
    """Прогон REQUESTS и bulk-import против app с подменённой базой; журнал — по каждому запросу.
    overrides — поля Settings поверх текущих (DATA_DIR — всегда root)."""
    out = []
    cfg = dataclasses.replace(main_module.settings, data_dir=str(root), **(overrides or {}))
    with patch.object(main_module, "settings", cfg), \
            patch.multiple(main_module.db, count=fake_db.count, search=fake_db.search, insert_one=fake_db.insert_one,
                           insert_many=fake_db.insert_many, delete_one=fake_db.delete_one):
        for name, method, path, fields, seed in REQUESTS:
            fake_db.calls.clear()
            files = {"image": (f"f{seed}.png", png(seed))} if seed is not None else None
            r = call(app, method, path, fields={k: str(v) for k, v in fields.items()} if fields else None, files=files)
            out.append({"name": name, **r, "db_calls": json.loads(json.dumps(fake_db.calls))})
        fake_db.calls.clear()
        r = call(app, "POST", "/v1/gallery/bulk-import",
                 json_body={"images_dir": str(root / "crops"), "csv_path": str(root / "g.csv")})
        out.append({"name": "bulk_import", **r, "db_calls": json.loads(json.dumps(fake_db.calls))})
        fake_db.calls.clear()
        r = call(app, "POST", "/v1/search", fields={"x": "40", "y": "30", "w": "120", "h": "90", "threshold": "0.0"},
                 files={"image": ("f9.png", png(9))})
        out.append({"name": "search_after_bulk", **r, "db_calls": json.loads(json.dumps(fake_db.calls))})
        empty = FakeDB(n=0)
        with patch.object(main_module.db, "count", empty.count):
            r = call(app, "POST", "/v1/search", fields={"x": "40", "y": "30", "w": "120", "h": "90"},
                     files={"image": ("f1.png", png(1))})
        out.append({"name": "search_empty_gallery", **r, "db_calls": []})
    return out


def load_app():
    """Импорт app.main с фейковой моделью в состоянии приложения (lifespan не запускается)."""
    from app import main as main_module
    main_module.app.state.model = FakeModel()
    main_module.app.state.model_loaded = True
    return main_module
