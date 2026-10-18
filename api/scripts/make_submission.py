#!/usr/bin/env python3
"""Офлайн-сборка трёх артефактов сдачи — тем же путём, что и сервис.

    docker compose exec api python scripts/make_submission.py        # DATA_DIR: test_*.csv + images/
    python api/scripts/make_submission.py --zip dataset.zip --out artifacts   # прямо из архива датасета

Кадр идёт как в /v1/search: полный кадр + bbox из test_*.csv -> validate_bbox -> кроп
с запасом BBOX_PAD -> модель (EMBED_MODEL, REID_WEIGHTS). Галерея добавляется как в
/v1/gallery/bulk: app.pp.state.Postprocess (DBA по PP_DBA_K соседям, признак фона кадра).
Поиск — Postprocess.search: ранжирование PP_DEDUP_RANK, оценка отказа REFUSAL_SCORE по
кандидатам без почти-дублей PP_DEDUP_REFUSAL, порог SEARCH_THRESHOLD. Вместо pgvector —
точный поиск в памяти (MemoryStore). Настройки — те же переменные окружения, что у сервиса
(api/app/config.py), поэтому по умолчанию это production: a10, DBA k=1, отказ z100
без почти-дублей «фон ИЛИ рамка», порог 4.6506 (docs/refusal.md).

Каждый query обрабатывается отдельно, другие query не используются (ответ организаторов
Q38): постобработка трогает только галерею — статичную базу.

На выходе в --out:
  submission.csv        query_id,gallery_id_1..gallery_id_10 — порядок ответа сервиса
  embeddings.npy        float32 (N_query + N_gallery, dim) — векторы модели без постобработки:
                        сначала query в порядке test_query.csv, затем gallery в порядке test_gallery.csv
  embeddings_index.csv  row,split,image_id — порядок строк embeddings.npy
  candidates.csv        query_id,gallery_id,confidence — принятые query: top-10 как в ответе
                        /v1/search (confidence — косинус кандидата); отказ — строк для query нет
  postprocess.json      модель, sha256 весов, постобработка, порог, принято / отказов
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # api/ — пакет app (в контейнере /srv)

from app.config import REFUSAL_PRESETS, settings      # noqa: E402
from app.imaging import crop_with_pad, load_image, validate_bbox  # noqa: E402
from app.models.registry import build_model             # noqa: E402
from app.pp import rerank as pp_rerank                  # noqa: E402
from app.pp import state as pp_state                    # noqa: E402

TOP_K = 10


class Frames:
    """Кадры и разметка датасета: каталог (test_*.csv, images/) или dataset.zip того же устройства."""

    def __init__(self, data_dir: Path, zip_path: Path = None, images: Path = None):
        self.zip = zipfile.ZipFile(zip_path) if zip_path else None
        self.data_dir = data_dir
        self.images = images or data_dir / "images"

    def rows(self, name: str, path: str = None) -> list:
        if path:
            text = Path(path).read_text(encoding="utf-8")
        elif self.zip is not None:
            text = self.zip.read(name).decode("utf-8")
        else:
            text = (self.data_dir / name).read_text(encoding="utf-8")
        return [{"image_id": r["image_id"], "bbox": tuple(int(float(r[c])) for c in "xywh")}
                for r in csv.DictReader(io.StringIO(text))]

    def raw(self, image_id: str) -> bytes:
        if self.zip is not None:
            return self.zip.read(f"images/{image_id}.jpg")
        return (self.images / f"{image_id}.jpg").read_bytes()

    def load(self, row: dict, need_frame: bool):
        """-> (полный кадр или None, кроп) — как _prepare / _prepare_frame в app/main.py."""
        img = load_image(self.raw(row["image_id"]), settings.max_upload_bytes)
        validate_bbox(img, *row["bbox"])
        return (img if need_frame else None), crop_with_pad(img, *row["bbox"], settings.bbox_pad)


class MemoryStore:
    """Хранилище для app.pp.state.Postprocess вместо app.db: те же функции, точный поиск."""

    def __init__(self):
        self.rows = []

    def ensure_pp_schema(self, dim, bg_dim):
        pass

    def fetch_pp(self, with_bg):
        return [{"id": r["id"], "embedding": r["embedding"], "embedding_dba": r["embedding_dba"],
                 "bg": r["bg"] if with_bg else None} for r in self.rows]

    def update_dba(self, pairs):
        for i, v in pairs:
            self.rows[i - 1]["embedding_dba"] = np.asarray(v, "float32")

    def insert_pp(self, rows, dba_updates):
        ids = []
        for vehicle_id, image_path, bbox, vec, dba_vec, bg in rows:
            ids.append(len(self.rows) + 1)
            self.rows.append({"id": ids[-1], "image_path": image_path, "bbox": list(bbox),
                              "embedding": np.asarray(vec, "float32"),
                              "embedding_dba": None if dba_vec is None else np.asarray(dba_vec, "float32"),
                              "bg": bg})
        self.update_dba(dba_updates)
        return ids

    def search_pp(self, vec, k, column, ef_search):
        """Как db.search_pp: k ближайших по колонке, confidence — косинус без округления."""
        rows = [r for r in self.rows if r[column] is not None]
        if not rows:
            return []
        m = np.stack([r[column] for r in rows]).astype("float64")
        q = np.asarray(vec, "float64")
        cos = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-12)
        top = np.argsort(-cos, kind="stable")[:k]
        return [{"id": rows[i]["id"], "vehicle_id": None, "image_path": rows[i]["image_path"],
                 "bbox": rows[i]["bbox"], "confidence": float(cos[i])} for i in top]

    def delete_pp(self, item_id, dba_updates):
        raise NotImplementedError("офлайн-сборка ничего не удаляет")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def embed_rows(model, frames: Frames, rows: list, batch: int, label: str, need_frame: bool):
    """Порциями по batch: (векторы порции, кадры порции или None, строки порции)."""
    t0 = time.perf_counter()
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        loaded = [frames.load(r, need_frame) for r in chunk]
        vecs = model.embed_batch([c for _, c in loaded]) if model is not None else None
        yield vecs, [f for f, _ in loaded], chunk
        done = start + len(chunk)
        if done % (batch * 10) == 0 or done == len(rows):
            print(f"  {label}: {done}/{len(rows)}  {done / (time.perf_counter() - t0):.1f} кадров/с", flush=True)


def check_settings() -> list:
    errors = pp_state.validate(settings) if pp_state.enabled(settings) else []
    if settings.refusal_score not in pp_rerank.FORMULAS:
        errors.append(f"REFUSAL_SCORE: одно из {pp_rerank.FORMULAS}, получено {settings.refusal_score!r}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=settings.data_dir,
                    help="каталог датасета: test_query.csv, test_gallery.csv, images/ (по умолчанию DATA_DIR)")
    ap.add_argument("--zip", type=Path, default=None, help="dataset.zip организаторов вместо --data")
    ap.add_argument("--images", type=Path, default=None, help="каталог кадров, если не <data>/images")
    ap.add_argument("--query-csv", default=None, help="test_query.csv, если не в --data / --zip")
    ap.add_argument("--gallery-csv", default=None, help="test_gallery.csv, если не в --data / --zip")
    ap.add_argument("--out", default=settings.artifacts_dir)
    ap.add_argument("--threshold", type=float, default=settings.search_threshold,
                    help=f"порог отказа по оценке {settings.refusal_score} (по умолчанию SEARCH_THRESHOLD)")
    ap.add_argument("--batch", type=int, default=settings.batch_size)
    ap.add_argument("--reuse-embeddings", action="store_true",
                    help="взять векторы из готового embeddings.npy (+ embeddings_index.csv), модель не грузить: "
                         "нужно, когда меняется только порог или постобработка")
    args = ap.parse_args()

    errors = check_settings()
    if settings.refusal_score == "raw" and not 0.0 <= args.threshold <= 1.0:
        errors.append(f"REFUSAL_SCORE=raw: порог — косинус от 0 до 1, получено {args.threshold}")
    if not math.isfinite(args.threshold):
        errors.append("порог должен быть конечным числом")
    if errors:
        print("ошибка настроек: " + "; ".join(errors), file=sys.stderr)
        return 2

    frames = Frames(Path(args.data), args.zip, args.images)
    q_rows = frames.rows("test_query.csv", args.query_csv)
    g_rows = frames.rows("test_gallery.csv", args.gallery_csv)
    q_ids = [r["image_id"] for r in q_rows]
    g_ids = [r["image_id"] for r in g_rows]
    print(f"query: {len(q_ids)}, gallery: {len(g_ids)}")
    if len(g_ids) < TOP_K:
        print(f"ВНИМАНИЕ: в галерее всего {len(g_ids)} объектов, топ-10 не набрать", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    emb_path, idx_path = out / "embeddings.npy", out / "embeddings_index.csv"
    want_ids = q_ids + g_ids

    saved, model = None, None
    if args.reuse_embeddings:
        if not emb_path.exists() or not idx_path.exists():
            print(f"нет {emb_path} или {idx_path} — пересчитай без --reuse-embeddings", file=sys.stderr)
            return 1
        saved_ids = [r["image_id"] for r in csv.DictReader(idx_path.open(encoding="utf-8"))]
        if saved_ids != want_ids:
            print("порядок или состав строк embeddings.npy не совпадает с test_*.csv — пересчитай заново",
                  file=sys.stderr)
            return 1
        saved = np.load(emb_path).astype("float32")
        dim, model_name = saved.shape[1], settings.embed_model
        print(f"векторы взяты из {emb_path}: {saved.shape}, модель не грузится")
    else:
        model = build_model(settings.embed_model, settings.input_size)
        t0 = time.perf_counter()
        model.load()
        dim, model_name = model.dim, model.name
        print(f"модель {model.name} загружена за {time.perf_counter() - t0:.1f} с, dim={dim}")

    weights = os.getenv("REID_WEIGHTS", "")
    weights_sha = sha256(weights) if weights and Path(weights).is_file() else None
    pp_on = pp_state.enabled(settings)
    store = MemoryStore()
    pp = pp_state.Postprocess(settings, dim, store) if pp_on else None
    need_frame = bool(pp is not None and pp.wants_frame)

    # ---- галерея: как /v1/gallery/bulk — порциями, DBA и признак фона считает Postprocess
    t_all = time.perf_counter()
    g_emb = np.zeros((len(g_ids), dim), "float32")
    gid_by_store = {}
    for n, (vecs, frs, chunk) in enumerate(embed_rows(model, frames, g_rows, args.batch, "gallery", need_frame)):
        s = n * args.batch
        vecs = vecs if vecs is not None else saved[len(q_ids) + s:len(q_ids) + s + len(chunk)]
        g_emb[s:s + len(chunk)] = vecs
        items = [(None, r["image_id"], r["bbox"], v, f) for r, v, f in zip(chunk, vecs, frs)]
        if pp is not None:
            new_ids = pp.add_many(items)
        else:
            new_ids = store.insert_pp([(None, r["image_id"], r["bbox"], v, None, None) for r, v in zip(chunk, vecs)],
                                      [])
        gid_by_store.update({i: r["image_id"] for i, r in zip(new_ids, chunk)})

    # ---- query: как /v1/search, каждый отдельно
    q_emb = np.zeros((len(q_ids), dim), "float32")
    answers, scores = [], []
    for n, (vecs, frs, chunk) in enumerate(embed_rows(model, frames, q_rows, args.batch, "query", need_frame)):
        s = n * args.batch
        vecs = vecs if vecs is not None else saved[s:s + len(chunk)]
        q_emb[s:s + len(chunk)] = vecs
        for r, v, f in zip(chunk, vecs, frs):
            if pp is not None:
                found, score = pp.search(v, TOP_K, f, r["bbox"])
            else:                                       # как _search_custom / db.search без постобработки
                found = store.search_pp(v, pp_rerank.needed_k(TOP_K, False, 0, settings.refusal_score),
                                        "embedding", 0)
                score = pp_rerank.refusal_score([c["confidence"] for c in found], [False] * len(found),
                                                settings.refusal_score)
                found = found[:TOP_K]
            answers.append([(gid_by_store[int(c["id"])], float(c["confidence"])) for c in found])
            scores.append(score if math.isfinite(score) else float("-inf"))
    took = time.perf_counter() - t_all
    print(f"галерея и поиск: {took:.1f} с")

    # ---- артефакты
    with (out / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, TOP_K + 1)])
        for q, ans in zip(q_ids, answers):
            w.writerow([q] + [g for g, _ in ans[:TOP_K]])

    if not args.reuse_embeddings:
        np.save(emb_path, np.vstack([q_emb, g_emb]).astype("float32"))
        with idx_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row", "split", "image_id"])
            for i, v in enumerate(want_ids):
                w.writerow([i, "query" if i < len(q_ids) else "gallery", v])

    accepted = 0
    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for q, ans, score in zip(q_ids, answers, scores):
            if score >= args.threshold:                 # как /v1/search: best < thr — отказ
                accepted += 1
                w.writerows([(q, g, round(c, 4)) for g, c in ans[:TOP_K]])
    refused = len(q_ids) - accepted

    # Векторы в embeddings.npy — без постобработки, а порядок submission.csv — по g' = DBA(g):
    # доля query, у которых top-1 разный, — честная мера этого расхождения.
    raw_top1 = np.argmax(q_emb @ g_emb.T, axis=1)
    mismatch = float(np.mean([g_ids[t] != ans[0][0] for t, ans in zip(raw_top1, answers)])) if answers else 0.0
    preset = REFUSAL_PRESETS.get(pp_state.preset_key(settings)) if pp_on else None
    meta = {
        "model": model_name,
        "weights": Path(weights).name if weights else None,
        "weights_sha256": weights_sha,
        "crop": f"bbox + {settings.bbox_pad:.0%} запаса (app/imaging.crop_with_pad), REID_TF_MODE="
                f"{os.getenv('REID_TF_MODE', '')}, TTA flip={settings.tta_flip}",
        "postprocess": {
            "path": "app.pp.state.Postprocess — тот же код, что в сервисе; другие query не используются",
            "PP_DBA_K": settings.pp_dba_k if pp_on else 0,
            "PP_DEDUP_RANK": pp_state.dedup_rank(settings) if pp_on else "off",
            "PP_DEDUP_REFUSAL": pp_state.dedup_refusal(settings) if pp_on else "off",
            "PP_CAMERA_TAU": settings.pp_camera_tau,
            "PP_DEDUP_SIM": settings.pp_dedup_sim, "PP_DEDUP_IOU": settings.pp_dedup_iou,
            "PP_DEDUP_CENTER": settings.pp_dedup_center,
        },
        "refusal": {
            "score": settings.refusal_score,
            "threshold": args.threshold,
            "preset": None if preset is None else {"balanced": preset["balanced"], "note": preset["note"]},
            "accepted": accepted,
            "refused": refused,
            "candidates_rule": "принятый query — top-10 как в ответе /v1/search, confidence — косинус кандидата",
        },
        "consistency": {
            "top1_mismatch_pct": round(100 * mismatch, 2),
            "note": "доля query, где top-1 submission.csv (по DBA-галерее) отличается от top-1 по косинусу "
                    "векторов embeddings.npy (без постобработки)",
        },
        "reproduce": "docker compose exec api python scripts/make_submission.py "
                     "(или python api/scripts/make_submission.py --zip dataset.zip --out artifacts)",
    }
    (out / "postprocess.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"submission.csv   {out / 'submission.csv'}  строк {len(q_ids)}")
    print(f"embeddings.npy   {emb_path}  форма {(len(want_ids), dim)}  модель {model_name}")
    print(f"candidates.csv   {out / 'candidates.csv'}  {settings.refusal_score} ≥ {args.threshold}")
    print(f"  принято query: {accepted}, отказов: {refused} ({refused / max(len(q_ids), 1) * 100:.1f}%)")
    print(f"  top-1 submission ≠ top-1 по embeddings.npy: {100 * mismatch:.2f}% query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
