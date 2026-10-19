#!/usr/bin/env python3
"""Офлайн-сборка трёх артефактов сдачи — тем же путём, что и сервис.

    docker compose exec api python scripts/make_submission.py        # DATA_DIR: test_*.csv + images/
    python api/scripts/make_submission.py --zip dataset.zip --out artifacts   # прямо из архива датасета

Кадр идёт как в /v1/search: полный кадр + bbox из test_*.csv -> кроп с запасом BBOX_PAD -> модель
(EMBED_MODEL, REID_WEIGHTS). Одна строка CSV не останавливает прогон (imaging.fit_bbox): рамка за кадром
обрезается по границам, пустая или уже 2 px — весь кадр, нечитаемый кадр — серый 224×224; каждый случай —
WARNING / ERROR с image_id, итог — строкой в конце лога. Строк в embeddings.npy ровно столько, сколько в CSV (Q27). Галерея добавляется как в
/v1/gallery/bulk: app.pp.state.Postprocess (DBA по PP_DBA_K соседям, признак фона кадра).
Поиск — Postprocess.search: ранжирование PP_DEDUP_RANK, оценка отказа REFUSAL_SCORE по
кандидатам без почти-дублей PP_DEDUP_REFUSAL, порог SEARCH_THRESHOLD. Вместо pgvector —
точный поиск в памяти (MemoryStore). Настройки — те же переменные окружения, что у сервиса
(api/app/config.py), поэтому по умолчанию это production: a10, DBA k=1, отказ raw
без почти-дублей «фон ИЛИ рамка», порог 0.7586 (docs/refusal.md).

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
  make_submission.log   лог прогона (он же в stderr): веса и sha256, устройство, порог, время по этапам
"""
import argparse
import collections
import contextlib
import csv
import io
import json
import logging
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # api/ — пакет app (в контейнере /srv)

from app import weights_check                           # noqa: E402
from app.config import REFUSAL_PRESETS, settings      # noqa: E402
from app.imaging import crop_with_pad, fit_bbox, gray_frame, load_image  # noqa: E402
from app.models.registry import build_model             # noqa: E402
from app.pp import rerank as pp_rerank                  # noqa: E402
from app.pp import state as pp_state                    # noqa: E402

TOP_K = 10
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"      # как в app/main.py
log = logging.getLogger("case17.submission")


class Frames:
    """Кадры и разметка датасета: каталог (test_*.csv, images/) или dataset.zip того же устройства."""

    def __init__(self, data_dir: Path, zip_path: Path = None, images: Path = None):
        self.zip = zipfile.ZipFile(zip_path) if zip_path else None
        self.data_dir = data_dir
        self.images = images or data_dir / "images"
        self.fixes = collections.Counter()               # clipped / full / unread — итог в конце лога

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

    def load(self, row: dict, need_frame: bool, split: str = ""):
        """-> (полный кадр или None, кроп, рамка) — как _prepare / _prepare_frame в app/main.py, но без остановки:
        рамка за кадром обрезается, вырожденная — весь кадр, нечитаемый кадр — серый (imaging.fit_bbox)."""
        iid, bbox = row["image_id"], row["bbox"]
        try:
            img = load_image(self.raw(iid), settings.max_upload_bytes)
        except Exception as exc:                          # битый файл, не тот формат, нет на диске или в zip
            img = gray_frame(settings.input_size)
            box = (0, 0, img.width, img.height)
            self.fixes["unread"] += 1
            log.error("%s image_id %s: кадр не прочитан (%s) — эмбеддинг по серому кадру %d×%d", split, iid, exc,
                      img.width, img.height)
        else:
            box, status = fit_bbox(img.width, img.height, *bbox)
            if status == "clipped":
                self.fixes["clipped"] += 1
                log.warning("%s image_id %s: рамка %s выходит за кадр %d×%d — обрезана до %s", split, iid, bbox,
                            img.width, img.height, box)
            elif status == "full":
                self.fixes["full"] += 1
                log.warning("%s image_id %s: рамка %s после обрезки по кадру %d×%d пустая или уже 2 px — беру весь "
                            "кадр %s", split, iid, bbox, img.width, img.height, box)
        return (img if need_frame else None), crop_with_pad(img, *box, settings.bbox_pad), box

    def fixes_line(self) -> str:
        f = self.fixes
        return (f"кадры: рамок обрезано {f['clipped']}, заменено на весь кадр {f['full']}, "
                f"не прочитано {f['unread']}")


class MemoryStore:
    """Хранилище для app.pp.state.Postprocess вместо app.db: те же функции, точный поиск."""

    def __init__(self):
        self.rows = []
        self.dba_updated = set()                       # id кадров, у которых g' переписан после вставки

    def ensure_pp_schema(self, dim, bg_dim):
        pass

    def fetch_pp(self, with_bg):
        return [{"id": r["id"], "embedding": r["embedding"], "embedding_dba": r["embedding_dba"],
                 "bg": r["bg"] if with_bg else None} for r in self.rows]

    def update_dba(self, pairs):
        for i, v in pairs:
            self.rows[i - 1]["embedding_dba"] = np.asarray(v, "float32")
            self.dba_updated.add(int(i))

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


def embed_rows(model, frames: Frames, rows: list, batch: int, label: str, need_frame: bool, took: dict):
    """Порциями по batch: (векторы порции, кадры порции или None, строки порции, рамки после fit_bbox).

    took[label] — секунды на чтение кадров и модель, без того, что вызывающий делает с порцией.
    """
    spent = 0.0
    for start in range(0, len(rows), batch):
        t0 = time.perf_counter()
        chunk = rows[start:start + batch]
        loaded = [frames.load(r, need_frame, label) for r in chunk]
        vecs = model.embed_batch([c for _, c, _ in loaded]) if model is not None else None
        spent += time.perf_counter() - t0
        took[label] = spent
        yield vecs, [f for f, _, _ in loaded], chunk, [b for _, _, b in loaded]
        done = start + len(chunk)
        if done % (batch * 10) == 0 or done == len(rows):
            log.info("%s: %d/%d, %.1f кадров/с", label, done, len(rows), done / max(spent, 1e-9))


def check_settings() -> list:
    errors = pp_state.validate(settings) if pp_state.enabled(settings) else []
    if settings.refusal_score not in pp_rerank.FORMULAS:
        errors.append(f"REFUSAL_SCORE: одно из {pp_rerank.FORMULAS}, получено {settings.refusal_score!r}")
    return errors


def parse_args() -> argparse.Namespace:
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
    return ap.parse_args()


def setup_logging(out: Path) -> Path:
    """Лог в stderr и в <out>/make_submission.log (перезаписывается), формат как в app/main.py."""
    path = out / "make_submission.log"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True,
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(path, "w", encoding="utf-8")])
    return path


@contextlib.contextmanager
def quiet(logger_name: str, level: int = logging.WARNING):
    """Приглушить логгер на время блока: reload() пустого MemoryStore пишет «галерея 0 кадров» до индекса."""
    lg = logging.getLogger(logger_name)
    old = lg.level
    lg.setLevel(level)
    try:
        yield
    finally:
        lg.setLevel(old)


def log_gallery_index(pp, store: "MemoryStore", n_gallery: int) -> bool:
    """Итог индекса галереи реальными числами; False — индекс видит не все кадры (сборку останавливаем)."""
    n_dba = sum(r["embedding_dba"] is not None for r in store.rows)
    parts = [f"постобработка по галерее: в индексе {len(pp.ids)} кадров из {n_gallery}"]
    if pp.k:
        parts.append(f"DBA k={pp.k}: g' у {n_dba}, пересчитан при добавлении соседей у {len(store.dba_updated)}")
    if pp.use_bg:
        parts.append(f"признак фона у {len(pp.bg)} из {len(pp.ids)} (τ {pp.tau:.5f})")
    log.info("; ".join(parts))
    ok = len(pp.ids) == n_gallery and (not pp.k or n_dba == n_gallery)
    if not ok:
        log.error("индекс галереи неполон: кадров %d, g' %d, ожидалось %d", len(pp.ids), n_dba, n_gallery)
    return ok


def model_label(weights: str) -> str:
    """Подпись модели без загрузки сети (--reuse-embeddings): бэкбон — по чекпоинту, как в load()."""
    name = settings.embed_model
    if name.startswith("reid:") and weights and Path(weights).is_file():
        from app.models.reid_checkpoint import checkpoint_model_name
        return checkpoint_model_name(name.split(":", 1)[1], weights)
    return name


def log_refusal_config(threshold: float, pp_on: bool):
    """Порог отказа и посчитанный пресет (config.REFUSAL_PRESETS) — как _warn_threshold_preset в app/main.py."""
    if pp_on:
        log.info("постобработка: DBA k=%d, почти-дубли — ранжирование %s, отказ %s (τ фона %.5f)", settings.pp_dba_k,
                 pp_state.dedup_rank(settings), pp_state.dedup_refusal(settings), settings.pp_camera_tau)
    else:
        log.info("постобработка выключена")
    key = pp_state.preset_key(settings) if pp_on else (0, "off", "off", settings.refusal_score)
    preset = REFUSAL_PRESETS.get(key)
    if preset is None:
        log.warning("отказ: оценка %s, порог %.4f — для %s нет посчитанного пресета (config.REFUSAL_PRESETS)",
                    settings.refusal_score, threshold, key)
        return preset
    log.info("отказ: оценка %s, порог %.4f; пресет %s: %.4f, строгий %.4f — %s", settings.refusal_score, threshold,
             key, preset["balanced"], preset["strict"], preset["note"])
    if abs(preset["balanced"] - threshold) > 1e-9:
        log.warning("порог %.4f не совпадает с посчитанным для пресета %.4f", threshold, preset["balanced"])
    return preset


def build(args: argparse.Namespace, out: Path) -> int:
    errors = check_settings()
    if settings.refusal_score == "raw" and not 0.0 <= args.threshold <= 1.0:
        errors.append(f"REFUSAL_SCORE=raw: порог — косинус от 0 до 1, получено {args.threshold}")
    if not math.isfinite(args.threshold):
        errors.append("порог должен быть конечным числом")
    if errors:
        log.error("ошибка настроек: %s", "; ".join(errors))
        return 2

    took = {}
    log.info("make_submission: данные %s, выход %s, EMBED_MODEL %s", args.zip or args.data, out, settings.embed_model)
    weights = os.getenv("REID_WEIGHTS", "")
    weights_sha, _ = weights_check.inspect(settings.embed_model, settings.reid_weights_sha256)  # пишет путь и sha256
    if weights_sha is None and settings.embed_model.startswith("reid:"):
        log.warning("файла весов нет: %s", weights_check.weights_path(settings.embed_model))
    pp_on = pp_state.enabled(settings)
    preset = log_refusal_config(args.threshold, pp_on)

    frames = Frames(Path(args.data), args.zip, args.images)
    q_rows = frames.rows("test_query.csv", args.query_csv)
    g_rows = frames.rows("test_gallery.csv", args.gallery_csv)
    q_ids = [r["image_id"] for r in q_rows]
    g_ids = [r["image_id"] for r in g_rows]
    log.info("query: %d, gallery: %d", len(q_ids), len(g_ids))
    if len(g_ids) < TOP_K:
        log.warning("в галерее всего %d объектов, топ-10 не набрать", len(g_ids))

    emb_path, idx_path = out / "embeddings.npy", out / "embeddings_index.csv"
    want_ids = q_ids + g_ids

    saved, model = None, None
    if args.reuse_embeddings:
        if not emb_path.exists() or not idx_path.exists():
            log.error("нет %s или %s — пересчитай без --reuse-embeddings", emb_path, idx_path)
            return 1
        saved_ids = [r["image_id"] for r in csv.DictReader(idx_path.open(encoding="utf-8"))]
        if saved_ids != want_ids:
            log.error("порядок или состав строк embeddings.npy не совпадает с test_*.csv — пересчитай заново")
            return 1
        saved = np.load(emb_path).astype("float32")
        dim, model_name = saved.shape[1], model_label(weights)
        log.info("векторы взяты из %s: %s, модель не грузится (%s)", emb_path, saved.shape, model_name)
    else:
        model = build_model(settings.embed_model, settings.input_size)
        t0 = time.perf_counter()
        model.load()
        took["модель"] = time.perf_counter() - t0
        dim, model_name = model.dim, model.name
        log.info("модель %s загружена за %.1f с: размерность %d, устройство %s, FP16 %s", model_name, took["модель"],
                 dim, getattr(model, "device", "cpu"), "да" if getattr(model, "_fp16", False) else "нет")

    store = MemoryStore()
    with quiet("case17.pp"):                    # нули пустого хранилища; итог — после индекса (log_gallery_index)
        pp = pp_state.Postprocess(settings, dim, store) if pp_on else None
    need_frame = bool(pp is not None and pp.wants_frame)

    # ---- галерея: как /v1/gallery/bulk — порциями, DBA и признак фона считает Postprocess
    g_emb = np.zeros((len(g_ids), dim), "float32")
    gid_by_store, took["галерея в индекс"] = {}, 0.0
    for n, (vecs, frs, chunk, boxes) in enumerate(embed_rows(model, frames, g_rows, args.batch, "gallery", need_frame,
                                                             took)):
        t0 = time.perf_counter()
        s = n * args.batch
        vecs = vecs if vecs is not None else saved[len(q_ids) + s:len(q_ids) + s + len(chunk)]
        g_emb[s:s + len(chunk)] = vecs
        items = [(None, r["image_id"], b, v, f) for r, b, v, f in zip(chunk, boxes, vecs, frs)]
        if pp is not None:
            new_ids = pp.add_many(items)
        else:
            new_ids = store.insert_pp([(None, r["image_id"], b, v, None, None) for r, b, v in zip(chunk, boxes, vecs)],
                                      [])
        gid_by_store.update({i: r["image_id"] for i, r in zip(new_ids, chunk)})
        took["галерея в индекс"] += time.perf_counter() - t0
    log.info("галерея: кадры и эмбеддинги %.1f с, в индекс (DBA, фон) %.1f с",
             took.get("gallery", 0.0), took["галерея в индекс"])
    if pp is not None and not log_gallery_index(pp, store, len(g_ids)):
        return 1

    # ---- query: как /v1/search, каждый отдельно
    q_emb = np.zeros((len(q_ids), dim), "float32")
    answers, scores, took["поиск"] = [], [], 0.0
    for n, (vecs, frs, chunk, boxes) in enumerate(embed_rows(model, frames, q_rows, args.batch, "query", need_frame,
                                                             took)):
        t0 = time.perf_counter()
        s = n * args.batch
        vecs = vecs if vecs is not None else saved[s:s + len(chunk)]
        q_emb[s:s + len(chunk)] = vecs
        for b, v, f in zip(boxes, vecs, frs):
            if pp is not None:
                found, score = pp.search(v, TOP_K, f, b)
            else:                                       # как _search_custom / db.search без постобработки
                found = store.search_pp(v, pp_rerank.needed_k(TOP_K, False, 0, settings.refusal_score),
                                        "embedding", 0)
                score = pp_rerank.refusal_score([c["confidence"] for c in found], [False] * len(found),
                                                settings.refusal_score)
                found = found[:TOP_K]
            answers.append([(gid_by_store[int(c["id"])], float(c["confidence"])) for c in found])
            scores.append(score if math.isfinite(score) else float("-inf"))
        took["поиск"] += time.perf_counter() - t0
    log.info("query: кадры и эмбеддинги %.1f с, поиск %.1f с", took.get("query", 0.0), took["поиск"])

    # ---- артефакты
    t0 = time.perf_counter()
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
            "preset": None if preset is None or not pp_on else {"balanced": preset["balanced"], "note": preset["note"]},
            "accepted": accepted,
            "refused": refused,
            "candidates_rule": "принятый query — top-10 как в ответе /v1/search, confidence — косинус кандидата",
        },
        "consistency": {
            "top1_mismatch_pct": round(100 * mismatch, 2),
            "note": "доля query, где top-1 submission.csv (по DBA-галерее) отличается от top-1 по косинусу "
                    "векторов embeddings.npy (без постобработки)",
        },
        "frame_fixes": {"clipped": frames.fixes["clipped"], "full_frame": frames.fixes["full"],
                        "unread_gray": frames.fixes["unread"],
                        "note": "рамка за кадром обрезана / пустая — весь кадр / кадр не прочитан — серый 224×224"},
        "reproduce": "docker compose exec api python scripts/make_submission.py "
                     "(или python api/scripts/make_submission.py --zip dataset.zip --out artifacts)",
    }
    (out / "postprocess.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    took["запись файлов"] = time.perf_counter() - t0
    log.info("запись файлов %.1f с", took["запись файлов"])

    log.info("submission.csv   %s  строк %d", out / "submission.csv", len(q_ids))
    log.info("embeddings.npy   %s  форма %s  модель %s%s", emb_path, (len(want_ids), dim), model_name,
             "  (не перезаписан: --reuse-embeddings)" if args.reuse_embeddings else "")
    log.info("candidates.csv   %s  %s ≥ %s", out / "candidates.csv", settings.refusal_score, args.threshold)
    log.info("принято query: %d, отказов: %d (%.1f%%)", accepted, refused, refused / max(len(q_ids), 1) * 100)
    log.info("top-1 submission ≠ top-1 по embeddings.npy: %.2f%% query", 100 * mismatch)
    log.log(logging.WARNING if sum(frames.fixes.values()) else logging.INFO, frames.fixes_line())
    return 0


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(out)
    t0 = time.perf_counter()
    try:
        rc = build(args, out)
    except Exception:
        log.exception("сборка упала — артефакты не перезаписаны или записаны не полностью")
        rc = 1
    log.info("всего %.1f с, код выхода %d, лог: %s", time.perf_counter() - t0, rc, log_path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
