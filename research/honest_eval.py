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
порог — --threshold (по умолчанию SEARCH_THRESHOLD профиля).

Вход — векторы: профиль сервиса (по умолчанию env_example = студент),
--env KEY=VAL для новой модели (например REID_WEIGHTS=...), либо готовый
файл --embeddings (.npz с ids/emb или .npy + --ids). Векторы кэшируются.

    python research/honest_eval.py --data-dir /opt/case17/data --tag student
    python research/honest_eval.py --data-dir /opt/case17/data --tag new \\
        --env REID_WEIGHTS=/models/new.pt --env REID_TF_MODE=squash
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
PP_VARIANTS = ("raw", "qq", "csls", "rerank", "full")


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
             mode: str, threshold: float, top_k: int = 10) -> dict:
    """sim — скор ранжирования (raw или после постобработки), raw — косинус
    для решения об отказе. Возвращает метрики эталона."""
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
    return {"mAP@10": rm["mAP@10"], "Rank-1": rm["Rank-1"], "Rank-5": rm["Rank-5"],
            "n_scored": rm["n_scored"], "n_openset": rm["n_openset_excluded"], **fm,
            "F1": cm["F1"], "Precision": cm["Precision"], "Recall": cm["Recall"], "TNR": cm["TNR"],
            "PR-AUC": cm["PR-AUC"], "PR-AUC_all": pr_all,
            "TP": cm["TP"], "FP": cm["FP"], "FN": cm["FN"], "TN": cm["TN"]}


# --------------------------------------------------------------------------

def load_vectors(a, ids: list, images_dir: Path, universe: list | None = None) -> np.ndarray:
    """Векторы для ids. universe — полный список кадров папки (например весь
    train.csv): считаем и кэшируем его целиком, чтобы разные сплиты брали
    строки из одного кэша, а не пересчитывали модель."""
    if a.embeddings:
        if a.embeddings.suffix == ".npz":
            z = np.load(a.embeddings)
            pos = {i: k for k, i in enumerate(z["ids"].tolist())}
            emb = z["emb"]
        else:
            emb = np.load(a.embeddings)
            pos = {i: k for k, i in enumerate(Path(a.ids).read_text().split())}
        missing = [i for i in ids if i not in pos]
        if missing:
            raise SystemExit(f"в {a.embeddings} нет {len(missing)} кадров, например {missing[0]}")
        return emb[[pos[i] for i in ids]].astype("float32")
    overrides = dict(kv.split("=", 1) for kv in a.env)
    base = universe or ids
    emb = se.embed(a.profile, images_dir, base, a.cache_dir, overrides=overrides)
    pos = {i: k for k, i in enumerate(base)}
    return emb[[pos[i] for i in ids]]


def summarize(rows: list) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype="float64")
        out[k] = {"mean": round(float(np.nanmean(v)), 4), "std": round(float(np.nanstd(v)), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="train.csv, test_gallery.csv, crops/")
    ap.add_argument("--holdout", type=Path, default=HERE / "val_ids.txt")
    ap.add_argument("--exclude-frames", type=Path,
                    default=HERE / "honest_eval_results" / "holdout_frame_dups.txt",
                    help="кадры holdout, которые не берём в честный протокол (по одному image_id в строке)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--openset-frac", type=float, default=0.25)
    ap.add_argument("--extra-distractors", default="test_all",
                    choices=("none", "test_gallery", "test_all", "train_seen", "test_all+train_seen"),
                    help="чужие машины в big-галерее: тестовые кадры, которых модель не видела "
                         "(test_gallery +750, test_all +1860), и/или кадры обучающих машин "
                         "(train_seen +8644, как в GENERALIZATION_RESULTS.md)")
    ap.add_argument("--one-query-per-camera", action="store_true",
                    help="по одному запросу на (машина, камера): убирает соседние кадры-запросы, "
                         "на которые опирается транздуктивное расширение запроса")
    ap.add_argument("--postprocess", default="raw",
                    help=f"через запятую из {PP_VARIANTS}; raw — как отвечает /v1/search")
    ap.add_argument("--pp-params", type=Path, default=ROOT / "artifacts" / "postprocess.json")
    ap.add_argument("--threshold", type=float, default=None,
                    help="порог отказа по сырому косинусу; по умолчанию SEARCH_THRESHOLD профиля")
    ap.add_argument("--profile", choices=se.PROFILES, default="env_example")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL поверх профиля")
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--ids", default=None, help="для --embeddings .npy: файл image_id по строке")
    ap.add_argument("--cache-dir", type=Path, default=se.DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=HERE / "honest_eval_results")
    ap.add_argument("--tag", default="student")
    a = ap.parse_args()

    env = se.profile_env(a.profile, overrides=dict(kv.split("=", 1) for kv in a.env))
    threshold = a.threshold if a.threshold is not None else float(env.get("SEARCH_THRESHOLD", "nan"))
    variants = [v.strip() for v in a.postprocess.split(",") if v.strip()]
    params = pp_params(a.pp_params)

    train = read_csv(a.data_dir / "train.csv")
    hold = set(a.holdout.read_text().split())
    # Старый протокол — ровно сплит make_split.py, без исключений (сверка с README).
    split_old = holdout_split(train, hold)
    # Честный — без кадров holdout, побайтно совпадающих с кадром обучающей
    # машины (research/audit_leaks.py --zip): эта машина частично видна в чужом
    # обучающем кропе. Исключаются до сплита — в gallery встаёт следующий кадр.
    excluded = set(a.exclude_frames.read_text().split()) if a.exclude_frames and a.exclude_frames.exists() else set()
    split = holdout_split([r for r in train if r["image_id"] not in excluded], hold)
    ids = sorted({r["image_id"] for r in split_old + split})
    vec = dict(zip(ids, load_vectors(a, ids, a.data_dir / "crops" / "train",
                                     universe=[r["image_id"] for r in train])))
    gal_all = [r for r in split if r["role"] == "gallery"]
    qry_all = [r for r in split if r["role"] == "query"]
    if a.one_query_per_camera:
        first = {}
        for r in sorted(qry_all, key=lambda r: r["image_id"]):
            first.setdefault((r["vehicle_id"], r["camera_id"]), r)
        qry_all = list(first.values())

    # Дистракторы — заведомо чужие машины, которых модель не видела: тестовые
    # кадры (по ТЗ, п.5.4, машины test и train не пересекаются). Метки
    # машин и камер у теста нет, поэтому каждый кадр — своя «машина» без
    # камеры: в позитивы и в junk он не попадает. Тест здесь ничего не
    # подбирает — только утяжеляет галерею.
    extra, extra_vec = [], {}
    parts = {"none": (), "test_gallery": ("gallery",), "test_all": ("gallery", "query"),
             "train_seen": ("train",), "test_all+train_seen": ("gallery", "query", "train")}[a.extra_distractors]
    for part in parts:
        if part == "train":
            # Как в research/GENERALIZATION_RESULTS.md: кадры машин, на которых
            # модель училась. Чужие для любого запроса holdout, но модель их видела.
            seen = [r for r in train if r["vehicle_id"] not in hold]
            extra += [{"image_id": r["image_id"], "vehicle_id": r["vehicle_id"], "camera_id": r["camera_id"]}
                      for r in seen]
            sid = [r["image_id"] for r in seen]
            extra_vec.update(zip(sid, load_vectors(a, sid, a.data_dir / "crops" / "train",
                                                   universe=[r["image_id"] for r in train])))
            continue
        tid = [r["image_id"] for r in read_csv(a.data_dir / f"test_{part}.csv")]
        extra += [{"image_id": i, "vehicle_id": f"test:{i}", "camera_id": f"test:{i}"} for i in tid]
        extra_vec.update(zip(tid, load_vectors(a, tid, a.data_dir / "crops" / part)))

    report = {"tag": a.tag, "env": {k: v.replace(str(ROOT), "<repo>") for k, v in env.items()},
              "threshold": threshold, "params": params,
              "holdout_vehicles": len(hold), "gallery_frames": len(gal_all), "query_frames": len(qry_all),
              "excluded_frames": len(excluded & {r["image_id"] for r in split_old}),
              "extra_distractors": len(extra), "folds": a.folds, "openset_frac": a.openset_frac,
              "seed": a.seed}

    # Старый протокол — одна галерея из всех 150 машин, без фолдов и без
    # open-set; эталон. Сверка с числами README (0.8581 raw, 0.8961 full).
    Q = frame([r for r in split_old if r["role"] == "query"])
    G = frame([r for r in split_old if r["role"] == "gallery"])
    qv = np.stack([vec[i] for i in Q.index]); gv = np.stack([vec[i] for i in G.index])
    self_check(qv, gv, Q, G)
    raw_old = qv @ gv.T
    report["old_protocol"] = {v: evaluate(scores(qv, gv, v, params), raw_old, Q, G, "official", threshold)
                              for v in variants}

    per_fold = []
    folds = folds_by_vehicle(sorted(hold), a.folds, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    for f, fvids in enumerate(folds):
        fset = set(fvids)
        n_open = int(round(len(fvids) * a.openset_frac))
        opened = set(rng.choice(sorted(fvids, key=int), size=n_open, replace=False).tolist())
        q_rows = [r for r in qry_all if r["vehicle_id"] in fset]
        for gname in GALLERIES:
            base = [r for r in gal_all if (gname == "big" or r["vehicle_id"] in fset)
                    and r["vehicle_id"] not in opened]
            g_rows = base + (extra if gname == "big" else [])
            Qf, Gf = frame(q_rows), frame(g_rows)
            qf = np.stack([vec[i] for i in Qf.index])
            gf = np.stack([vec.get(i, extra_vec.get(i)) for i in Gf.index])
            raw = qf @ gf.T
            for v in variants:
                s = scores(qf, gf, v, params)
                for mode in MODES:
                    m = evaluate(s, raw, Qf, Gf, mode, threshold)
                    per_fold.append({"fold": f, "gallery": gname, "postprocess": v, "mode": mode,
                                     "n_query": len(q_rows), "n_gallery": len(g_rows),
                                     "openset_vehicles": n_open, **m})
            print(f"  фолд {f}: {gname} галерея {len(g_rows)}, запросов {len(q_rows)}", flush=True)

    summary = {}
    for gname in GALLERIES:
        for v in variants:
            for mode in MODES:
                rows = [r for r in per_fold if r["gallery"] == gname and r["postprocess"] == v and r["mode"] == mode]
                summary[f"{gname}/{v}/{mode}"] = summarize(
                    [{k: x[k] for k in x if k not in ("fold", "gallery", "postprocess", "mode")} for x in rows])
    report["summary"] = summary

    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"honest_eval_{a.tag}"
    (a.out_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (a.out_dir / f"{stem}_folds.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_fold[0]))
        w.writeheader()
        w.writerows(per_fold)

    print(f"\nстарый протокол (150 машин, gallery {len(G)}, query {len(Q)}, эталон):")
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
