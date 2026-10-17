#!/usr/bin/env python3
"""Честная валидация по протоколу экспертов: K фолдов по машинам, большая
галерея с дистракторами, запросы без пары, метрики эталонным evaluate.py.

Почему не research/evaluate.py. Тот скрипт выкидывает из выдачи ВСЮ камеру
запроса. Эталон организаторов (research/official_evaluate.py) выкидывает
только junk — кадры той же машины С ТОЙ ЖЕ камеры; чужие машины с той же
камеры остаются трудными негативами. И главное: эталон чистит junk из
присланных 10 позиций, поэтому почти-дубль с той же камеры занимает место в
top-10. Здесь top-10 строится так, как его строит сервис — без знания камер,
— а потом к нему применяется эталон.

Протокол (по умолчанию):
  * holdout = research/val_ids.txt (150 машин, не участвовавших в обучении
    студента — проверено research/audit_leaks.py по прототипам ArcFace);
  * gallery: по одному кадру (первому по image_id) с каждой камеры каждой
    машины, остальные кадры — query (правило research/make_split.py);
  * K=5 фолдов по машинам; в фолде query — кадры его машин;
  * open-set: --openset-frac машин фолда (25%) теряют все кадры галереи —
    их запросы оцениваются только режимом отказа (TNR);
  * две галереи: small — только машины фолда; big — все машины holdout
    (+ --extra-distractors: тестовая gallery как заведомо чужие машины);
  * три режима:
      official — правило эталона, top-10 без знания камер;
      wholecam — справочно: вся камера запроса выкинута (старый протокол команды);
      nodup    — справочно: кадры той же машины с той же камеры убраны из
                 галереи совсем, в том числе для отказа («чистая» выборка).

Метрики: mAP@10, Rank-1, Rank-5 (эталон по top-10), mAP по полному
ранжированию и mINP, F1/Precision/Recall/TNR/PR-AUC режима отказа (эталон по
candidates). Отказ решается по сырому косинусу top-1, как в /v1/search;
порог — --threshold (по умолчанию SEARCH_THRESHOLD профиля). Для вариантов
постобработки отказ не считается: развёрнутого контракта для них нет.

Оговорки, которые печатаются в отчёте: порог 0.8025 и параметры
постобработки подбирались на этих же 150 машинах (оценка «в выборке»);
постобработка трансдуктивна и /v1/search недостижима.

Вход — векторы: профиль сервиса (env_example = студент из .env.example,
compose_default = docker compose без .env, --compose-ref — коммит compose),
--env KEY=VAL для новой модели (например REID_WEIGHTS=...), либо готовый
файл --embeddings (.npz с ids/emb или .npy + --ids; L2-нормируются).
Источник кадров: crops (готовые кропы) или frames (полный кадр из --zip,
кроп кодом сервиса, как /v1/embed). Кадры, чей bbox сервис отверг,
выбрасываются и считаются. Векторы кэшируются (research/service_embeddings.py).

    # все галереи одной командой: small / 2.2k / 8996 / 10.8k → markdown
    python research/honest_eval.py --data-dir /opt/case17/data --suite --tag new \\
        --env REID_WEIGHTS=/models/new.pt --env REID_TF_MODE=squash
    # одна галерея состава research/GENERALIZATION_RESULTS.md (8996 / 550)
    python research/honest_eval.py --data-dir /opt/case17/data --team-clean-like --tag student
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import official_evaluate as oe  # noqa: E402
import reid_postprocess as pp  # noqa: E402
import service_embeddings as se  # noqa: E402

MODES = ("official", "wholecam", "nodup")
GALLERIES = ("small", "big")
PP_VARIANTS = ("raw", "qq", "csls", "rerank", "kr", "full")


# --------------------------------------------------------------------------
# данные и сплит
# --------------------------------------------------------------------------

def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def holdout_split(train: list, vids: set) -> list:
    """Правило research/make_split.py: первый кадр (по image_id) каждой
    камеры каждой машины — в gallery, остальные — в query."""
    by = defaultdict(list)
    for r in train:
        if r["vehicle_id"] in vids:
            by[(r["vehicle_id"], r["camera_id"])].append(r)
    out = []
    for key in sorted(by, key=lambda k: (int(k[0]), int(k[1]))):
        items = sorted(by[key], key=lambda r: r["image_id"])
        out.append({**items[0], "role": "gallery"})
        out.extend({**r, "role": "query"} for r in items[1:])
    return out


def folds_by_vehicle(vids: list, k: int, seed: int) -> list:
    order = sorted(vids, key=int)
    np.random.default_rng(seed).shuffle(order)
    return [order[i::k] for i in range(k)]


def frame(rows: list) -> pd.DataFrame:
    """DataFrame в формате эталона: index image_id, vehicle_id, camera_id."""
    return pd.DataFrame([{"image_id": r["image_id"], "vehicle_id": r["vehicle_id"],
                          "camera_id": r["camera_id"]} for r in rows]).set_index("image_id")


# --------------------------------------------------------------------------
# скоры
# --------------------------------------------------------------------------

def pp_params(path: Path) -> dict:
    p = {"qq_k": 3, "qq_alpha": 3.0, "csls_k": 5, "rerank_k1": 10, "rerank_k2": 6, "rerank_lambda": 0.7}
    if path and path.exists():
        p.update(json.loads(path.read_text(encoding="utf-8")).get("params", {}))
    return p


def scores(q: np.ndarray, g: np.ndarray, variant: str, p: dict) -> np.ndarray:
    """Те же шаги, что research/ablate_postprocess.score_variants, по одному.
    full = reid_postprocess.rank с параметрами artifacts/postprocess.json."""
    q, g = pp.l2(q), pp.l2(g)
    if variant == "raw":
        return q @ g.T
    if variant == "kr":
        # k-reciprocal БЕЗ расширения запроса. В ablate_postprocess.score_variants
        # (и в таблицах GENERALIZATION_RESULTS.md / CLEAN_RETRAIN_REPORT.md) строка
        # «k-reciprocal» считается поверх qq_expand, то есть тоже по другим запросам.
        # Здесь — только k-reciprocal; он всё равно трансдуктивен (граф соседей по
        # всем запросам и галерее), но без расширения запроса.
        return -pp.k_reciprocal_rerank(q, g, p["rerank_k1"], p["rerank_k2"], p["rerank_lambda"])
    qe = pp.qq_expand(q, p["qq_k"], p["qq_alpha"])
    if variant == "qq":
        return qe @ g.T
    if variant == "csls":
        return pp.csls(qe @ g.T, p["csls_k"])
    rr = -pp.k_reciprocal_rerank(qe, g, p["rerank_k1"], p["rerank_k2"], p["rerank_lambda"])
    if variant == "rerank":
        return rr
    if variant == "full":
        return pp.csls(rr, p["csls_k"])
    raise ValueError(variant)


# --------------------------------------------------------------------------
# метрики
# --------------------------------------------------------------------------

def full_metrics(sim: np.ndarray, query: pd.DataFrame, gallery: pd.DataFrame) -> dict:
    """mAP по полному ранжированию и mINP — логика oe.full_ranking_metrics,
    но по готовой матрице скоров (нужно для постобработки). Совпадение с
    эталоном на сыром косинусе проверяется в self_check()."""
    g_vid = gallery.vehicle_id.to_numpy()
    g_cam = gallery.camera_id.to_numpy()
    aps, inps = [], []
    for i, (_, row) in enumerate(query.iterrows()):
        keep = ~((g_vid == row.vehicle_id) & (g_cam == row.camera_id)) & np.isfinite(sim[i])
        if not keep.any():
            continue
        order = np.argsort(-sim[i][keep], kind="stable")
        rel = (g_vid[keep] == row.vehicle_id)[order]
        n_pos = int(rel.sum())
        if n_pos == 0:
            continue
        cum = np.cumsum(rel)
        aps.append(float((cum / (np.arange(len(rel)) + 1) * rel).sum() / n_pos))
        inps.append(n_pos / (int(np.max(np.nonzero(rel)[0])) + 1))
    return {"mAP_full": float(np.mean(aps)) if aps else 0.0,
            "mINP": float(np.mean(inps)) if inps else 0.0}


def self_check(q: np.ndarray, g: np.ndarray, query: pd.DataFrame, gallery: pd.DataFrame) -> None:
    ref = oe.full_ranking_metrics(q, g, list(query.index), list(gallery.index), query, gallery)
    mine = full_metrics(q @ g.T, query, gallery)
    for k in ("mAP_full", "mINP"):
        if abs(ref[k] - mine[k]) > 1e-9:
            raise AssertionError(f"full_metrics расходится с эталоном: {k} {mine[k]} != {ref[k]}")


def evaluate(sim: np.ndarray, raw: np.ndarray, query: pd.DataFrame, gallery: pd.DataFrame,
             mode: str, threshold: float, top_k: int = 10, refusal: bool = True) -> dict:
    """sim — скор ранжирования (raw или после постобработки), raw — косинус
    для решения об отказе. Возвращает метрики эталона.

    refusal=False — метрики отказа не считаются (NaN): для постобработки нет
    развёрнутого контракта «кандидат по одному скору, порог по другому».
    """
    q_vid = query.vehicle_id.to_numpy()[:, None]
    q_cam = query.camera_id.to_numpy()[:, None]
    g_vid = gallery.vehicle_id.to_numpy()[None, :]
    g_cam = gallery.camera_id.to_numpy()[None, :]
    if mode == "wholecam":
        mask = g_cam == q_cam
    elif mode == "nodup":
        mask = (g_vid == q_vid) & (g_cam == q_cam)
    else:
        mask = np.zeros(sim.shape, bool)
    s = np.where(mask, -np.inf, sim)
    r = np.where(mask, -np.inf, raw)
    g_ids = np.array(gallery.index)
    order = np.argsort(-s, axis=1, kind="stable")
    top_k_order = order[:, :top_k]
    # Как submission.csv: 10 позиций без знания камер; junk эталон вычистит сам.
    ranked = {qid: [g_ids[j] for j in top_k_order[i] if np.isfinite(s[i, j])]
              for i, qid in enumerate(query.index)}
    rm = oe.ranking_metrics(query, gallery, ranked, top_k=top_k)
    fm = full_metrics(s, query, gallery)
    base = {"mAP@10": rm["mAP@10"], "Rank-1": rm["Rank-1"], "Rank-5": rm["Rank-5"],
            "n_scored": rm["n_scored"], "n_openset": rm["n_openset_excluded"], **fm}
    if not refusal:
        nan = float("nan")
        return {**base, **{k: nan for k in ("F1", "Precision", "Recall", "TNR", "PR-AUC", "PR-AUC_all",
                                            "TP", "FP", "FN", "TN")}}
    # кандидат = top-1 по ранжированию, решение — по сырому косинусу этого top-1
    top = order[:, 0]
    conf = r[np.arange(len(top)), top]
    cands = {qid: [(g_ids[top[i]], float(conf[i]))] for i, qid in enumerate(query.index)
             if np.isfinite(conf[i]) and conf[i] >= threshold}
    cm = oe.candidate_metrics(query, gallery, cands)
    # Эталон считает PR-AUC по присланным confidence, отказ = -inf, поэтому его
    # PR-AUC зависит от порога. PR-AUC_all — та же мера по всем top-1 без порога.
    all_c = {qid: [(g_ids[top[i]], float(conf[i]))] for i, qid in enumerate(query.index) if np.isfinite(conf[i])}
    pr_all = oe.candidate_metrics(query, gallery, all_c)["PR-AUC"]
    return {**base,
            "F1": cm["F1"], "Precision": cm["Precision"], "Recall": cm["Recall"], "TNR": cm["TNR"],
            "PR-AUC": cm["PR-AUC"], "PR-AUC_all": pr_all,
            "TP": cm["TP"], "FP": cm["FP"], "FN": cm["FN"], "TN": cm["TN"]}


# --------------------------------------------------------------------------
# данные и векторы
# --------------------------------------------------------------------------

class Data:
    """Строки датасета, сплиты и источник векторов одного прогона."""

    def __init__(self, a):
        self.a = a
        d = a.data_dir
        self.train = read_csv(d / "train.csv")
        self.test = {p: read_csv(d / f"test_{p}.csv") for p in ("query", "gallery")}
        self.hold = set(a.holdout.read_text().split())
        # Старый протокол — ровно сплит make_split.py, без исключений (сверка с README).
        self.split_old = holdout_split(self.train, self.hold)
        # Честный — без кадров holdout, побайтно совпадающих с кадром обучающей
        # машины (research/audit_leaks.py --zip): эта машина частично видна в
        # чужом обучающем кропе. Исключаются до сплита — в gallery встаёт следующий кадр.
        ex = a.exclude_frames
        if ex and not ex.exists():
            print(f"ВНИМАНИЕ: нет файла исключений {ex} — кадры holdout с дублем полного кадра "
                  f"в обучении останутся в протоколе", file=sys.stderr, flush=True)
        self.excluded = set(ex.read_text().split()) if ex and ex.exists() else set()
        self.dropped = set()          # кадры, которые сервис не принял бы (NaN-вектор)
        self.split = holdout_split([r for r in self.train if r["image_id"] not in self.excluded], self.hold)
        self.gal = [r for r in self.split if r["role"] == "gallery"]
        self.qry = [r for r in self.split if r["role"] == "query"]
        if a.one_query_per_camera:
            self.qry = one_query_per_camera(self.qry)
        self.overrides = dict(kv.split("=", 1) for kv in a.env)
        self.env = se.profile_env(a.profile, a.compose_ref, self.overrides)
        self.boxes = {r["image_id"]: tuple(r[k] for k in ("x", "y", "w", "h"))
                      for rows in [self.train, *self.test.values()] for r in rows}
        self._vec = {}

    def vectors(self, ids: list) -> np.ndarray:
        """Векторы кадров. Кадры одной части (train / test query / test gallery)
        считаются и кэшируются целиком, чтобы разные галереи брали строки из
        одного кэша, а не пересчитывали модель."""
        need = [i for i in ids if i not in self._vec]
        if need:
            self._load(need)
        return np.stack([self._vec[i] for i in ids])

    def valid(self, rows: list) -> list:
        """Строки без кадров, чей bbox отверг сервис (вектор из NaN): сервис
        не принял бы такой запрос и не добавил бы такой кадр в галерею."""
        if not rows:                  # например, дистракторы none
            return []
        v = self.vectors([r["image_id"] for r in rows])
        ok = np.isfinite(v).all(axis=1)
        self.dropped |= {r["image_id"] for r, good in zip(rows, ok) if not good}
        return [r for r, good in zip(rows, ok) if good]

    def _load(self, need: list) -> None:
        a = self.a
        if a.embeddings:
            if a.embeddings.suffix == ".npz":
                z = np.load(a.embeddings)
                pos = {i: k for k, i in enumerate(z["ids"].tolist())}
                emb = z["emb"]
            else:
                emb = np.load(a.embeddings)
                id_list = Path(a.ids).read_text().split()
                if len(id_list) != len(emb):
                    raise SystemExit(f"{a.ids}: {len(id_list)} image_id против {len(emb)} строк в {a.embeddings}")
                pos = {i: k for k, i in enumerate(id_list)}
            missing = [i for i in need if i not in pos]
            if missing:
                raise SystemExit(f"в {a.embeddings} нет {len(missing)} кадров, например {missing[0]}")
            # L2 на нашей стороне, как у эталона (official_evaluate.load_embeddings):
            # порог отказа сравнивается с косинусом, а не со скалярным произведением.
            for i in need:
                v = emb[pos[i]].astype("float32")
                self._vec[i] = v / max(float(np.linalg.norm(v)), 1e-12)
            return
        parts = {"train": [r["image_id"] for r in self.train],
                 "query": [r["image_id"] for r in self.test["query"]],
                 "gallery": [r["image_id"] for r in self.test["gallery"]]}
        want = set(need)
        for part, universe in parts.items():
            if not want & set(universe):
                continue
            crops = a.data_dir / "crops" / part
            if a.source == "frames":
                emb = se.embed(a.profile, crops, universe, a.cache_dir, compose_ref=a.compose_ref,
                               overrides=self.overrides, frames_zip=a.zip, boxes=self.boxes)
            else:
                emb = se.embed(a.profile, crops, universe, a.cache_dir, compose_ref=a.compose_ref,
                               overrides=self.overrides)
            self._vec.update(zip(universe, emb))

    def distractors(self, kind: str) -> list:
        """Чужие машины для big-галереи.

        Тестовые кадры — машины, которых модель не видела (ТЗ, п.5.4: test и
        train не пересекаются). Меток у теста нет, поэтому каждый кадр — своя
        «машина» без камеры: в позитивы и в junk он не попадает. Тест здесь
        ничего не подбирает, только утяжеляет галерею.
        Кадры обучающих машин — как в research/GENERALIZATION_RESULTS.md:
        чужие для любого запроса holdout, но модель их видела.
        """
        out = []
        for part in DISTRACTORS[kind]:
            if part == "train":
                out += [r for r in self.train if r["vehicle_id"] not in self.hold]
            else:
                out += [{"image_id": r["image_id"], "vehicle_id": f"test:{r['image_id']}",
                         "camera_id": f"test:{r['image_id']}"} for r in self.test[part]]
        return out


DISTRACTORS = {"none": (), "test_gallery": ("gallery",), "test_all": ("gallery", "query"),
               "train_seen": ("train",), "test_all+train_seen": ("gallery", "query", "train")}


def summarize(rows: list) -> dict:
    """Среднее и выборочный разброс (ddof=1) по фолдам; n — сколько фолдов
    дали число (TNR у фолда без open-set — NaN и в среднее не входит)."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype="float64")
        v = v[np.isfinite(v)]
        out[k] = {"mean": round(float(v.mean()), 4) if len(v) else float("nan"),
                  "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else float("nan"),
                  "n": int(len(v))}
    return out


# --------------------------------------------------------------------------
# протоколы
# --------------------------------------------------------------------------

def run_single(D: Data, q_rows: list, g_rows: list, variants: list, params: dict,
               threshold: float, modes=("official",)) -> dict:
    """Одна галерея без фолдов: старый протокол и командная clean-like."""
    q_rows, g_rows = D.valid(q_rows), D.valid(g_rows)
    Q, G = frame(q_rows), frame(g_rows)
    qv, gv = D.vectors(list(Q.index)), D.vectors(list(G.index))
    raw = qv @ gv.T
    return {f"{v}/{m}": evaluate(scores(qv, gv, v, params), raw, Q, G, m, threshold, refusal=(v == "raw"))
            for v in variants for m in modes}


def run_old(D: Data, variants: list, params: dict, threshold: float) -> dict:
    q = D.valid([r for r in D.split_old if r["role"] == "query"])
    g = D.valid([r for r in D.split_old if r["role"] == "gallery"])
    Q, G = frame(q), frame(g)
    self_check(D.vectors(list(Q.index)), D.vectors(list(G.index)), Q, G)
    return {k.split("/")[0]: v for k, v in run_single(D, q, g, variants, params, threshold).items()}


def team8996(D: Data, audit_path: Path = HERE / "honest_eval_results" / "leak_audit.json") -> tuple:
    """Состав clean-like галереи research/GENERALIZATION_RESULTS.md (8996 / 550).

    Код сборки в репозитории не опубликован. Состав восстановлен по числам
    документа и сходится со всеми тремя: убраны ОБЕ стороны 5 пар побайтно
    одинаковых кадров holdout↔обучение (research/honest_eval_results/leak_audit.json):
      * query — 551 запрос сплита make_split.py без 1 кадра-дубля → 550;
      * gallery — 361 − 4 кадра-дубля = 357 кадров holdout плюс 8644 − 5 = 8639
        кадров обучающих машин → 8996;
      * после этого у 7 запросов (машины 904, 1176, 688) не остаётся пары с
        другой камеры — ровно «seven queries … are excluded» из документа.
    В тексте документа написано «361 + 8635» — сумма та же, но тогда у всех
    550 запросов осталась бы пара; 7 исключённых запросов подтверждают 357 + 8639.
    """
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    drop = set()
    for p in audit["exact_full_frame_duplicates"]["pairs"]:
        if p["kind"] == "holdout↔train1391":
            drop |= {p["a"], p["b"]}
    if not (drop <= {r["image_id"] for r in D.train}):
        raise ValueError("в парах дублей есть кадры не из train.csv")
    q = [r for r in D.split_old if r["role"] == "query" and r["image_id"] not in drop]
    g = [r for r in D.split_old if r["role"] == "gallery" and r["image_id"] not in drop]
    g += [r for r in D.train if r["vehicle_id"] not in D.hold and r["image_id"] not in drop]
    return q, g


def run_folds(D: Data, kind: str, variants: list, params: dict, threshold: float,
              galleries=GALLERIES, qry: list | None = None) -> tuple:
    """K фолдов по машинам holdout. qry — свой набор запросов (например один
    на пару машина-камера), по умолчанию D.qry. Фолды и open-set задаются
    seed и одинаковы для всех моделей и галерей."""
    extra = D.valid(D.distractors(kind))
    qry = D.valid(D.qry if qry is None else qry)
    gal = D.valid(D.gal)
    per_fold = []
    folds = folds_by_vehicle(sorted(D.hold), D.a.folds, D.a.seed)
    rng = np.random.default_rng(D.a.seed + 1)
    for f, fvids in enumerate(folds):
        fset = set(fvids)
        n_open = int(round(len(fvids) * D.a.openset_frac))
        opened = set(rng.choice(sorted(fvids, key=int), size=n_open, replace=False).tolist())
        q_rows = [r for r in qry if r["vehicle_id"] in fset]
        for gname in galleries:
            base = [r for r in gal if (gname == "big" or r["vehicle_id"] in fset)
                    and r["vehicle_id"] not in opened]
            g_rows = base + (extra if gname == "big" else [])
            Qf, Gf = frame(q_rows), frame(g_rows)
            qf, gf = D.vectors(list(Qf.index)), D.vectors(list(Gf.index))
            raw = qf @ gf.T
            for v in variants:
                s = scores(qf, gf, v, params)
                for mode in MODES:
                    m = evaluate(s, raw, Qf, Gf, mode, threshold, refusal=(v == "raw"))
                    per_fold.append({"fold": f, "gallery": gname, "postprocess": v, "mode": mode,
                                     "n_query": len(q_rows), "n_gallery": len(g_rows),
                                     "openset_vehicles": n_open, **m})
            print(f"  фолд {f}: {gname} галерея {len(g_rows)}, запросов {len(q_rows)}", flush=True)
    summary = {}
    for gname in galleries:
        for v in variants:
            for mode in MODES:
                rows = [r for r in per_fold if r["gallery"] == gname and r["postprocess"] == v and r["mode"] == mode]
                summary[f"{gname}/{v}/{mode}"] = summarize(
                    [{k: x[k] for k in x if k not in ("fold", "gallery", "postprocess", "mode")} for x in rows])
    return summary, per_fold, len(extra)


def one_query_per_camera(rows: list) -> list:
    first = {}
    for r in sorted(rows, key=lambda r: r["image_id"]):
        first.setdefault((r["vehicle_id"], r["camera_id"]), r)
    return list(first.values())


# --------------------------------------------------------------------------
# вывод
# --------------------------------------------------------------------------

def suite_markdown(tag: str, report: dict) -> str:
    """Одна таблица на модель: галереи small / 2.2k / 8996 / 10.8k."""
    rows = ["| Галерея | Кадров | Постобработка | mAP@10 | Rank-1 | Rank-5 | mAP | mINP | "
            "Recall / TNR / F1 (official) | Recall / TNR / F1 (без почти-дублей) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    f3 = lambda x: "—" if x != x else f"{x:.3f}"            # NaN -> прочерк
    pm = lambda d: "—" if d["mean"] != d["mean"] else (
        f"{d['mean']:.3f}" if d["std"] != d["std"] else f"{d['mean']:.3f} ± {d['std']:.3f}")
    for name, block in report["suite"].items():
        for v, s in block["metrics"].items():
            label = v if v == "raw" else f"{v} ¹"
            if block["kind"] == "folds":
                o, n = s["official"], s["nodup"]
                ref = lambda x: "—" if v != "raw" else f"{pm(x['Recall'])} / {pm(x['TNR'])} / {pm(x['F1'])}"
                rows.append(f"| {name} | {block['n_gallery']} | {label} | {pm(o['mAP@10'])} | "
                            f"{f3(o['Rank-1']['mean'])} | {f3(o['Rank-5']['mean'])} | {f3(o['mAP_full']['mean'])} | "
                            f"{f3(o['mINP']['mean'])} | {ref(o)} | {ref(n)} |")
            else:
                o = s["official"]
                ref = "—" if v != "raw" else (f"{f3(o['Recall'])} / {f3(o['TNR'])} / {f3(o['F1'])} "
                                              f"(без пары {o['n_openset']})")
                rows.append(f"| {name} | {block['n_gallery']} | {label} | {o['mAP@10']:.4f} | {f3(o['Rank-1'])} | "
                            f"{f3(o['Rank-5'])} | {f3(o['mAP_full'])} | {f3(o['mINP'])} | {ref} | — |")
    model = report["env"]
    head = [f"# honest_eval: {tag}", "",
            f"Модель: `{model.get('EMBED_MODEL')}`, веса `{model.get('REID_WEIGHTS', '')}`, "
            f"`REID_TF_MODE={model.get('REID_TF_MODE', '')}`, `EMBED_TTA_FLIP={model.get('EMBED_TTA_FLIP', '')}`, "
            f"источник кадров `{report['source']}`. Порог отказа {report['threshold']} (сырой косинус).", "",
            "Фолды: 5 по машинам holdout, open-set 25 %, среднее ± выборочный разброс (ddof=1). "
            "8996 — одна галерея состава research/GENERALIZATION_RESULTS.md, без фолдов; "
            "у 7 её запросов нет пары с другой камеры.", "",
            f"Кадров, чей bbox отверг сервис (выброшены): {report.get('dropped_frames', 0)}.", "",
            "Оговорки:",
            "- порог 0.8025 подобран на этих же 150 машинах holdout (`refusal()` в "
            "research/eval_service_path.py), поэтому recall/TNR/F1 при нём — оценка «в выборке»; "
            "калибровка вне фолда — research/calibrate_refusal_folds.py;",
            "- ¹ постобработка трансдуктивна (расширение запроса по ДРУГИМ запросам фолда, k-reciprocal по "
            "всей галерее), её параметры (artifacts/postprocess.json) подобраны на этих же 150 машинах, "
            "и /v1/search так не отвечает; отказ для неё не считается. Строка «1 запрос на камеру» — та же "
            "постобработка без соседних кадров-запросов одного проезда.", ""]
    return "\n".join(head + rows) + "\n"


def write(a, stem: str, report: dict, per_fold: list) -> None:
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if per_fold:
        with (a.out_dir / f"{stem}_folds.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_fold[0]))
            w.writeheader()
            w.writerows(per_fold)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="train.csv, test_*.csv, crops/")
    ap.add_argument("--holdout", type=Path, default=HERE / "val_ids.txt")
    ap.add_argument("--exclude-frames", type=Path,
                    default=HERE / "honest_eval_results" / "holdout_frame_dups.txt",
                    help="кадры holdout, которые не берём в честный протокол (по одному image_id в строке)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--openset-frac", type=float, default=0.25)
    ap.add_argument("--suite", action="store_true",
                    help="все галереи за один запуск: small, 2.2k (+весь тест), 8996 (как у команды), "
                         "10.8k (+тест и обучающие); таблица в honest_eval_<tag>_suite.md")
    ap.add_argument("--extra-distractors", default="test_all", choices=tuple(DISTRACTORS),
                    help="без --suite: чужие машины в big-галерее — тестовые кадры, которых модель не "
                         "видела (test_gallery +750, test_all +1860), и/или кадры обучающих машин "
                         "(train_seen +8644, как в GENERALIZATION_RESULTS.md)")
    ap.add_argument("--team-clean-like", action="store_true",
                    help="без --suite: только галерея 8996 состава GENERALIZATION_RESULTS.md")
    ap.add_argument("--one-query-per-camera", action="store_true",
                    help="по одному запросу на (машина, камера): убирает соседние кадры-запросы, "
                         "на которые опирается транздуктивное расширение запроса")
    ap.add_argument("--postprocess", default="raw",
                    help=f"через запятую из {PP_VARIANTS}; raw — как отвечает /v1/search")
    ap.add_argument("--pp-params", type=Path, default=ROOT / "artifacts" / "postprocess.json")
    ap.add_argument("--threshold", type=float, default=None,
                    help="порог отказа по сырому косинусу; по умолчанию SEARCH_THRESHOLD профиля")
    ap.add_argument("--profile", choices=se.PROFILES, default="env_example",
                    help="env_example — студент из .env.example; compose_default — docker compose без .env")
    ap.add_argument("--compose-ref", default="", help="коммит docker-compose.yml для compose_default (0924fb2 — main)")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL поверх профиля, например REID_WEIGHTS=...")
    ap.add_argument("--source", choices=("crops", "frames"), default="crops",
                    help="crops — готовые кропы crops/<часть>/ (как make_submission.py); frames — полный "
                         "кадр из --zip, обрезанный кодом сервиса (как /v1/embed)")
    ap.add_argument("--zip", type=Path, default=None, help="dataset.zip для --source frames")
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--ids", default=None, help="для --embeddings .npy: файл image_id по строке")
    ap.add_argument("--cache-dir", type=Path, default=se.DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=HERE / "honest_eval_results")
    ap.add_argument("--tag", default="student")
    a = ap.parse_args()
    if a.source == "frames" and not a.zip:
        ap.error("--source frames требует --zip")

    D = Data(a)
    threshold = a.threshold if a.threshold is not None else float(D.env.get("SEARCH_THRESHOLD", "nan"))
    variants = [v.strip() for v in a.postprocess.split(",") if v.strip()]
    params = pp_params(a.pp_params)
    report = {"tag": a.tag, "env": {k: v.replace(str(ROOT), "<repo>").replace(str(Path.home()), "~")
                                    for k, v in D.env.items()},
              "profile": a.profile, "compose_ref": a.compose_ref, "source": a.source,
              "threshold": threshold, "params": params,
              "holdout_vehicles": len(D.hold), "gallery_frames": len(D.gal), "query_frames": len(D.qry),
              "excluded_frames": len(D.excluded & {r["image_id"] for r in D.split_old}),
              "folds": a.folds, "openset_frac": a.openset_frac, "seed": a.seed}
    report["old_protocol"] = run_old(D, variants, params, threshold)

    if a.suite:
        report["suite"], per_fold = {}, []
        s1, pf1, n1 = run_folds(D, "test_all", variants, params, threshold)
        s2, pf2, n2 = run_folds(D, "test_all+train_seen", variants, params, threshold, galleries=("big",))
        tq, tg = team8996(D)
        tq, tg = D.valid(tq), D.valid(tg)
        t = run_single(D, tq, tg, variants, params, threshold)
        avg = lambda pf, g: int(round(np.mean([r["n_gallery"] for r in pf if r["gallery"] == g])))
        report["suite"]["small (машины фолда)"] = {"kind": "folds", "n_gallery": avg(pf1, "small"),
            "metrics": {v: {m: s1[f"small/{v}/{m}"] for m in MODES} for v in variants}}
        report["suite"]["2.2k (+ весь тест)"] = {"kind": "folds", "n_gallery": avg(pf1, "big"),
            "metrics": {v: {m: s1[f"big/{v}/{m}"] for m in MODES} for v in variants}}
        pf3 = []
        pp_only = [v for v in variants if v != "raw"]
        if pp_only:
            # Та же постобработка без соседних кадров-запросов одного проезда:
            # сколько она даёт, когда опереться на «братьев» по проезду нельзя.
            s3, pf3, _ = run_folds(D, "test_all", ["raw"] + pp_only, params, threshold, galleries=("big",),
                                   qry=one_query_per_camera(D.qry))
            report["suite"]["2.2k, 1 запрос на камеру"] = {"kind": "folds", "n_gallery": avg(pf3, "big"),
                "metrics": {v: {m: s3[f"big/{v}/{m}"] for m in MODES} for v in ["raw"] + pp_only}}
        report["suite"]["8996 (как у команды)"] = {"kind": "single", "n_gallery": len(tg), "n_query": len(tq),
            "metrics": {v: {"official": t[f"{v}/official"]} for v in variants}}
        report["suite"]["10.8k (+ тест и обучающие)"] = {"kind": "folds", "n_gallery": avg(pf2, "big"),
            "metrics": {v: {m: s2[f"big/{v}/{m}"] for m in MODES} for v in variants}}
        report["dropped_frames"] = len(D.dropped)
        stem = f"honest_eval_{a.tag}_suite"
        write(a, stem, report, [{"protocol": "2.2k", **r} for r in pf1] + [{"protocol": "10.8k", **r} for r in pf2]
              + [{"protocol": "2.2k_1q", **r} for r in pf3])
        md = suite_markdown(a.tag, report)
        (a.out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
        print("\n" + md)
        print(f"-> {a.out_dir / (stem + '.md')}")
        return 0

    if a.team_clean_like:
        tq, tg = team8996(D)
        tq, tg = D.valid(tq), D.valid(tg)
        report["team_clean_like"] = {"n_query": len(tq), "n_gallery": len(tg),
                                     **run_single(D, tq, tg, variants, params, threshold)}
        report["dropped_frames"] = len(D.dropped)
        write(a, f"honest_eval_{a.tag}", report, [])
        print(f"\nclean-like как у команды: query {len(tq)}, gallery {len(tg)}")
        for v in variants:
            m = report["team_clean_like"][f"{v}/official"]
            print(f"  {v:7s} mAP@10 {m['mAP@10']:.4f}  R1 {m['Rank-1']:.4f}  R5 {m['Rank-5']:.4f}  "
                  f"(в зачёте {m['n_scored']}, без пары {m['n_openset']})")
        return 0

    summary, per_fold, n_extra = run_folds(D, a.extra_distractors, variants, params, threshold)
    report["extra_distractors"] = n_extra
    report["summary"] = summary
    report["dropped_frames"] = len(D.dropped)
    stem = f"honest_eval_{a.tag}"
    write(a, stem, report, per_fold)
    print(f"\nстарый протокол (150 машин, gallery 361, query 551, эталон):")
    for v, m in report["old_protocol"].items():
        print(f"  {v:7s} mAP@10 {m['mAP@10']:.4f}  R1 {m['Rank-1']:.4f}  R5 {m['Rank-5']:.4f}  "
              f"mAP {m['mAP_full']:.4f}  mINP {m['mINP']:.4f}")
    print(f"\nчестный протокол ({a.folds} фолдов, open-set {a.openset_frac:.0%}, порог {threshold}):")
    for key, s in summary.items():
        print(f"  {key:24s} mAP@10 {s['mAP@10']['mean']:.4f}±{s['mAP@10']['std']:.4f}  "
              f"R1 {s['Rank-1']['mean']:.4f}  mAP {s['mAP_full']['mean']:.4f}  mINP {s['mINP']['mean']:.4f}  "
              f"F1 {s['F1']['mean']:.3f}  Rec {s['Recall']['mean']:.3f}  TNR {s['TNR']['mean']:.3f}  "
              f"PR-AUC {s['PR-AUC']['mean']:.3f}")
    print(f"\n-> {a.out_dir / (stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
