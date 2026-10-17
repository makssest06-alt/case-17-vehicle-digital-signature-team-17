#!/usr/bin/env python3
"""Постобработка по одному запросу на больших галереях: штраф за ту же камеру и DBA.

Всё считается по запросу и заранее известной галерее, другие запросы не
участвуют (кроме справочной строки kr — reid_postprocess.k_reciprocal_rerank
строит соседства по запросам и галерее вместе).

Варианты скора (кандидат — top-1, отказ — top-1 против порога):
  raw          сырой косинус;
  campen λ     штраф за ту же камеру: среди top-K кандидатов кадр «с той же
               камеры», если косинус фоновых дескрипторов (кадр с замазанной
               машиной, research/pseudo_camera.py) с запросом > τ; у него из скора
               вычитается λ, λ = inf — исключить. τ — по настоящим camera_id
               обучающих машин, проверка по группам камер (вне фолда);
  dba k        каждый кадр галереи заменяется на L2(g + Σ max(cos, 0)·g_j) по его
               k соседям в галерее (один раз при старте); запрос не меняется;
               реализация для сервиса — research/dba.py;
  kr           k-reciprocal (справочно, совместный).

Протоколы галереи (5 фолдов, 25 % машин фолда без пары, как research/honest_eval.py):
  nodup        кадры той же машины с той же камеры убраны (наш обычный режим,
               «оракул»: знает камеру);
  neardup      так видит сервис: у запроса с парой остаются кадры его машины с той
               же камеры; у запроса без пары убраны только кадры его машины с ДРУГИХ
               камер, кадр с той же камеры остаётся. Метрики — как эталон:
               junk вычищается из присланных 10 без добора, top-1 той же машины с той
               же камеры у запроса без пары — FP.

    python research/bigallery_pp.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate_refusal_folds as crf  # noqa: E402
import dba as dbam  # noqa: E402
import honest_eval as he  # noqa: E402
import refusal_score as rsc  # noqa: E402
import reid_postprocess as pp  # noqa: E402
import service_embeddings as se  # noqa: E402

TOPK = 100
LAMBDAS = (0.05, 0.1, 0.2, 0.3, float("inf"))
DBA_KS = (1, 2, 3, 4, 5)
GALLERIES = (("2.2k", "test_all"), ("9.0k", "train_seen"), ("10.8k", "test_all+train_seen"))
TNR_FLOOR = 0.70
BIG = 1e3            # «исключить»: кандидат уходит в конец, но остаётся кадром галереи


# --------------------------------------------------------------------------
# фон и «та же камера»
# --------------------------------------------------------------------------

def load_bg(cache_dir: Path, kind: str) -> dict:
    z = np.load(cache_dir / "pseudo_camera_desc.npz")
    x = z["emb"] if kind == "emb" else z["grad"] if kind == "grad" else np.concatenate([z["emb"], z["grad"]], 1)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return dict(zip(z["ids"].tolist(), x))


def calib_pairs(D, bg: dict, n_queries: int = 3000, seed: int = 0) -> tuple:
    """Пары (кадр обучения, его top-K по внешности среди других кадров обучения):
    косинус фона и метка «та же камера» — ровно то, что видит штраф."""
    tr = [r for r in D.valid(D.train) if r["vehicle_id"] not in D.hold]
    rng = np.random.default_rng(seed)
    qi = rng.choice(len(tr), size=min(n_queries, len(tr)), replace=False)
    ids = [r["image_id"] for r in tr]
    v = D.vectors(ids)
    B = np.stack([bg[i] for i in ids])
    cam = np.array([r["camera_id"] for r in tr])
    sim = v[qi] @ v.T
    sim[np.arange(len(qi)), qi] = -np.inf
    top = np.argpartition(-sim, TOPK - 1, axis=1)[:, :TOPK]
    bgs = pair_cos(B[qi], B, top)
    same = cam[top] == cam[qi][:, None]
    return bgs.ravel(), same.ravel(), np.repeat(cam[qi], TOPK)


def pair_cos(A: np.ndarray, B: np.ndarray, top: np.ndarray, chunk: int = 256) -> np.ndarray:
    """A[q] · B[top[q, k]] кусками по запросам: (Q, K, D) целиком не держим в памяти."""
    out = np.empty(top.shape, np.result_type(A, B))
    for s in range(0, len(top), chunk):
        out[s:s + chunk] = np.einsum("qd,qkd->qk", A[s:s + chunk], B[top[s:s + chunk]])
    return out


def pr_at(bgs, same, t) -> dict:
    flag = bgs > t
    tp, fp, fn = int((flag & same).sum()), int((flag & ~same).sum()), int((~flag & same).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tau": float(t), "precision": p, "recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0}


def best_tau(bgs, same, rule: str = "f1") -> float:
    """f1 — максимум F1 «та же камера»; p90 — самый низкий порог с точностью ≥ 0.90."""
    grid = np.quantile(bgs, np.linspace(0.5, 0.9995, 400))
    pts = [pr_at(bgs, same, t) for t in grid]
    if rule == "p90":
        ok = [m for m in pts if m["precision"] >= 0.90]
        return min(ok, key=lambda m: m["tau"])["tau"] if ok else float(grid[-1])
    return max(pts, key=lambda m: m["F1"])["tau"]


def calibrate_tau(D, bg: dict, folds: int = 5) -> dict:
    """τ по двум правилам (f1, p90) на парах обучающих машин; проверка — по группам
    камер: τ выбран на 4 группах камер, точность и полнота — на камерах 5-й."""
    bgs, same, qcam = calib_pairs(D, bg)
    cams = np.array(sorted(set(qcam), key=int))
    groups = np.array_split(np.random.default_rng(1).permutation(cams), folds)
    out = {"pairs": int(len(bgs)), "same_share": float(same.mean())}
    for rule in ("f1", "p90"):
        cv = []
        for g in groups:
            test = np.isin(qcam, g)
            cv.append(pr_at(bgs[test], same[test], best_tau(bgs[~test], same[~test], rule)))
        tau = best_tau(bgs, same, rule)
        out[rule] = {"tau": tau, "all_train": pr_at(bgs, same, tau), "camera_group_cv": cv,
                     "cv_mean": {k: float(np.mean([c[k] for c in cv])) for k in ("tau", "precision", "recall", "F1")}}
    return out


# --------------------------------------------------------------------------
# галереи, маски, варианты
# --------------------------------------------------------------------------

def fold_blocks(D, kind: str, folds: list, bg: dict) -> list:
    """Галерея фолда = все кадры галереи holdout + чужие машины kind. Кадры машин
    фолда «без пары» видны только их собственным запросам с той же камеры (neardup)."""
    extra = D.valid(D.distractors(kind)) if kind != "none" else []
    qry, gal = D.valid(D.qry), D.valid(D.gal)
    rng = np.random.default_rng(D.a.seed + 1)            # тот же open-set, что honest_eval
    out = []
    for fvids in folds:
        fset = set(fvids)
        n_open = int(round(len(fvids) * D.a.openset_frac))
        opened = set(rng.choice(sorted(fvids, key=int), size=n_open, replace=False).tolist())
        q_rows = [r for r in qry if r["vehicle_id"] in fset]
        out.append(make_block(D, q_rows, gal + extra, opened, bg))
    return out


def make_block(D, q_rows, g_rows, opened: set, bg: dict) -> dict:
    Q, G = he.frame(q_rows), he.frame(g_rows)
    qv, gv = D.vectors(list(Q.index)), D.vectors(list(G.index))
    q_vid, q_cam = Q.vehicle_id.to_numpy()[:, None], Q.camera_id.to_numpy()[:, None]
    g_vid, g_cam = G.vehicle_id.to_numpy()[None, :], G.camera_id.to_numpy()[None, :]
    same_v, same_c = g_vid == q_vid, g_cam == q_cam
    g_open = np.isin(G.vehicle_id.to_numpy(), list(opened))[None, :]
    hidden = g_open & ~(same_v & same_c)                  # машины без пары — только своим запросам с той же камеры
    return {"Q": Q, "G": G, "qv": qv, "gv": gv, "same_v": same_v, "same_c": same_c, "std": ~g_open[0],
            "mask": {"neardup": hidden, "nodup": hidden | (same_v & same_c)},
            "qbg": np.stack([bg[i] for i in Q.index]), "gbg": np.stack([bg[i] for i in G.index]),
            "vehicle": Q.vehicle_id.to_numpy(), "n_gallery": len(g_rows)}


EXTRA = 5            # запасные соседи для «чистого» DBA


def dba_data(b: dict, kmax: int) -> dict:
    """Соседи каждого кадра — только из стандартной галереи (без машин «без пары»)."""
    if "_nb" not in b:
        b["_nb"] = dbam.neighbours(b["gv"], kmax + EXTRA, pool=np.flatnonzero(b["std"]))
    return b["_nb"]


def dba_scores(b: dict, k: int, clean: bool) -> np.ndarray:
    """q · DBA(g). clean — для каждого запроса его почти-дубли (та же машина с той же
    камеры) не участвуют в усреднении соседей: так в режиме без почти-дублей они не
    протекают в соседние кадры той же машины с других камер."""
    idx, sim = dba_data(b, max(DBA_KS))
    g = dbam.dba_vectors(b["gv"], idx, sim, k)
    s = b["qv"] @ g.T
    if not clean:
        return s
    junk = b["same_v"] & b["same_c"]
    inv = {}
    for i, row in enumerate(idx[:, :k]):
        for j in row:
            inv.setdefault(int(j), []).append(i)
    for qi in range(len(s)):
        bad = set(np.flatnonzero(junk[qi]).tolist())
        hit = sorted({i for j in bad for i in inv.get(j, [])})
        for i in hit:
            keep = [t for t in range(idx.shape[1]) if int(idx[i, t]) not in bad][:k]
            v = b["gv"][i] + (np.maximum(sim[i, keep], 0.0)[:, None] * b["gv"][idx[i, keep]]).sum(0)
            s[qi, i] = float(b["qv"][qi] @ (v / np.linalg.norm(v)))
    return s


def scores(b: dict, variant: dict, mode: str, tau: float, params: dict) -> np.ndarray:
    """Скор варианта с уже применённой маской протокола (-inf — кадра для запроса нет)."""
    k = variant["dba"]
    if variant["kr"]:
        gv = dbam.dba_vectors(b["gv"], *dba_data(b, max(DBA_KS)), k) if k else b["gv"]
        s = -pp.k_reciprocal_rerank(b["qv"], gv, params["rerank_k1"], params["rerank_k2"], params["rerank_lambda"])
    elif k:
        s = dba_scores(b, k, clean=(mode == "nodup"))
    else:
        s = b["qv"] @ b["gv"].T
    s = np.where(b["mask"][mode], -np.inf, s)
    lam = variant["lambda"]
    if lam:
        tau = tau[variant.get("tau", "f1")]
        top = np.argpartition(-s, TOPK - 1, axis=1)[:, :TOPK]
        bgs = pair_cos(b["qbg"], b["gbg"], top)
        flag = (bgs > tau) & np.isfinite(np.take_along_axis(s, top, 1))
        pen = np.where(flag, -BIG if np.isinf(lam) else -lam, 0.0)        # «исключить» — в конец списка
        rows = np.repeat(np.arange(len(s)), TOPK).reshape(-1, TOPK)
        s = s.copy()
        s[rows, top] = s[rows, top] + pen
    return s


def evaluate(b: dict, s: np.ndarray, mode: str) -> dict:
    """Логика эталона с масками по запросу: top-10 по скору, junk (та же машина и
    камера) вычищается без добора; позитив — та же машина с другой камеры среди
    кадров, видимых запросу по протоколу (штраф метода видимость не меняет)."""
    junk = b["same_v"] & b["same_c"]
    pos = b["same_v"] & ~b["same_c"] & ~b["mask"][mode]
    n_pos = pos.sum(1)
    order = np.argsort(-s, axis=1, kind="stable")
    aps, r1, r5 = [], [], []
    for i in np.flatnonzero(n_pos > 0):
        top10 = [j for j in order[i, :10] if np.isfinite(s[i, j])]
        rel = np.array([b["same_v"][i, j] for j in top10 if not junk[i, j]], bool)[:10]
        if rel.any():
            cum = np.cumsum(rel)
            aps.append(float((cum / (np.arange(len(rel)) + 1) * rel).sum() / min(n_pos[i], 10)))
        else:
            aps.append(0.0)
        r1.append(bool(rel[:1].any()))
        r5.append(bool(rel[:5].any()))
    top1 = order[:, 0]
    rec = {"conf": s[np.arange(len(s)), top1], "has_match": n_pos > 0, "correct": b["same_v"][np.arange(len(s)), top1],
           "same_cam_top1": junk[np.arange(len(s)), top1]}
    return {"mAP@10": float(np.mean(aps)), "Rank-1": float(np.mean(r1)), "Rank-5": float(np.mean(r5)), "rec": rec}


def merged(recs: list, f: int, folds: list, vehicles: list) -> dict:
    if len(recs) == 1:
        keep = ~np.isin(vehicles[0], list(folds[f]))
        return {k: v[keep] for k, v in recs[0].items()}
    parts = [r for i, r in enumerate(recs) if i != f]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


THR_FROM = {"8996": "9.0k"}     # у 8996 без пары 7 запросов трёх машин: у ≥ 2 фолдов TNR не определён


def run(blocks: list, folds: list, variant: dict, mode: str, tau: float, params: dict,
        thresholds: list = None) -> dict:
    """thresholds — пороги по фолдам с другой галереи (THR_FROM); иначе порог фолда f
    выбирается на запросах фолда f этой же галереи."""
    ev = [evaluate(b, scores(b, variant, mode, tau, params), mode) for b in blocks]
    recs = [e["rec"] for e in ev]
    vehicles = [b["vehicle"] for b in blocks]
    thr, per = [], []
    for f in range(len(folds)):
        if thresholds is not None:
            t = thresholds[f]
        else:
            if len(recs) > 1:
                sel = recs[f]
            else:                                               # одна галерея: выбор на запросах фолда f
                keep = np.isin(vehicles[0], list(folds[f]))
                sel = {k: v[keep] for k, v in recs[0].items()}
            t = rsc.choose_f1(sel, TNR_FLOOR)
        thr.append(t)
        test = merged(recs, f, folds, vehicles)
        m = crf.metrics(test, t)
        m["open_same_cam_top1"] = int((~test["has_match"] & test["same_cam_top1"]).sum())
        m["n_open"] = int((~test["has_match"]).sum())
        per.append(m)
    st = lambda k: rsc.stats([p[k] for p in per])
    return {"mAP@10": rsc.stats([e["mAP@10"] for e in ev]), "Rank-1": rsc.stats([e["Rank-1"] for e in ev]),
            "Rank-5": rsc.stats([e["Rank-5"] for e in ev]), "threshold": rsc.stats(thr),
            "thresholds": [float(t) for t in thr], "thresholds_from": "own" if thresholds is None else "other",
            **{k: st(k) for k in ("F1", "TNR", "recall", "precision", "open_same_cam_top1", "n_open")},
            "open_same_cam_top1_all": int(sum((~r["has_match"] & r["same_cam_top1"]).sum() for r in recs)),
            "n_open_all": int(sum((~r["has_match"]).sum() for r in recs))}


SERVICE_ROWS = (("raw | neardup", "raw (так сейчас)"),
                ("campen τ=p90 λ=inf | neardup", "штраф за ту же камеру (фон resnet18), τ точн. ≥ 0.90, исключить"),
                ("campen τ=f1 λ=inf | neardup", "штраф за ту же камеру (фон resnet18), τ по F1, исключить"),
                ("dba k=1 | neardup", "DBA k=1"),
                ("dba k=1 + campen τ=p90 λ=inf | neardup", "DBA k=1 + штраф (фон resnet18)"))
ORACLE_ROWS = (("raw | nodup", "raw"),
               ("dba k=1 | nodup", "DBA k=1 (соседи без почти-дублей запроса)"),
               ("kr | nodup", "kr (справочно: совместный, видит другие запросы)"),
               ("dba k=1 + kr | nodup", "DBA k=1 + kr (справочно)"))
FPS_TARGET = 100


def cheap_rows(res: Path, tag: str) -> tuple:
    """Строка «DBA k=1 + штраф (дешёвый фон)» из research/cheap_bg.py, если он прогнан."""
    p = res / f"cheap_bg_{tag}.json"
    if not p.exists():
        return {}, None, None
    import cheap_bg                                     # здесь: cheap_bg сам импортирует этот модуль
    C = json.loads(p.read_text(encoding="utf-8"))
    pick = cheap_bg.choose(C["kinds"])
    return C["results"], pick, {**C, "pick_label": cheap_bg.LABEL[pick]}


def report_md(res: Path, tag: str) -> str:
    R = json.loads((res / f"bigallery_pp_{tag}.json").read_text(encoding="utf-8"))
    T = json.loads((res / f"threshold_vs_gallery_{tag}.json").read_text(encoding="utf-8"))
    B = json.loads((res / "bench_bigallery_pp.json").read_text(encoding="utf-8"))
    cr, pick, C = cheap_rows(res, tag)
    r = {**R["results"], **({f"dba k=1 + campen {pick} | neardup": cr[f"dba k=1 + campen {pick} | neardup"]} if pick else {})}
    big = ("9.0k", "10.8k", "8996")
    m = lambda key, g, k="mAP@10": r[key][g][k]["mean"]
    cal = R["tau_calibration"][R["bg"]]
    gpu = B["gpu_4060_estimate"]
    cost_pen = (f"+{B['mac_ms']['resnet18_448x256_forward']:.0f} мс CPU (Мак, только прогон resnet18); или на GPU: 4060 ≈ "
                f"{gpu['fps_before']:.0f} → {gpu['fps_after']:.0f} FPS (оценка)")
    cost_dba = f"0 мс; пересчёт галереи 10.8k — {B['dba']['rebuild_10847_s']:.1f} с, кадр — {B['dba']['add_one_frame_ms']:.1f} мс"
    rows = list(SERVICE_ROWS)
    cost = {"campen τ=p90 λ=inf | neardup": cost_pen, "campen τ=f1 λ=inf | neardup": cost_pen,
            "dba k=1 | neardup": cost_dba, "dba k=1 + campen τ=p90 λ=inf | neardup": cost_pen,
            "dba k=1 | nodup": cost_dba,
            "kr | nodup": "по всем запросам сразу: 53 с на 5 фолдов 10.8k на Маке, по одному запросу не мерили",
            "dba k=1 + kr | nodup": "то же"}
    pen_fps = {"dba k=1 + campen τ=p90 λ=inf | neardup": gpu["fps_after"]}
    if pick:
        key = f"dba k=1 + campen {pick} | neardup"
        k = C["kinds"][pick]
        rows.append((key, f"DBA k=1 + штраф (дешёвый фон: {C['pick_label']})"))
        cost[key] = f"+{k['ms']:.2f} мс CPU (Мак); 4060 ≈ {1000 / C['gpu_ms']['b1']:.0f} → {k['fps']['b1']:.0f} FPS при b1 (оценка)"
        pen_fps[key] = k["fps"]["b1"]

    def verdict(key):
        if key in pen_fps:                       # штраф поверх DBA: F1 растёт на 9.0k и 10.8k и FPS ≥ 100
            up = all(m(key, g, "F1") > m("dba k=1 | neardup", g, "F1") for g in ("9.0k", "10.8k"))
            return "**да**" if up and pen_fps[key] >= FPS_TARGET else f"нет: {pen_fps[key]:.0f} FPS < {FPS_TARGET}" if up else "нет"
        return {"raw | neardup": "база", "campen τ=p90 λ=inf | neardup": "только вместе с DBA",
                "campen τ=f1 λ=inf | neardup": "нет", "dba k=1 | neardup": "**да**", "raw | nodup": "база",
                "dba k=1 | nodup": "**да**", "kr | nodup": "справочно", "dba k=1 + kr | nodup": "справочно"}.get(key, "")

    def table(rows_, base):
        out = ["| Вариант | mAP@10 9.0k / 10.8k / 8996 (Δ к raw) | F1 / TNR 9.0k | F1 / TNR 10.8k | F1 / TNR 8996 | "
               "без пары → та же машина с той же камеры, 9.0k / 10.8k | Стоимость на запрос | В прод |",
               "|---|---|---:|---:|---:|---:|---|---|"]
        for key, label in rows_:
            if key not in r:
                continue
            maps = " / ".join(f"{m(key, g):.3f}" + ("" if key == base else f" ({m(key, g) - m(base, g):+.3f})") for g in big)
            fr = lambda g: f"{m(key, g, 'F1'):.3f} / {m(key, g, 'TNR'):.3f}"
            same = " / ".join(f"{r[key][g]['open_same_cam_top1_all']} из {r[key][g]['n_open_all']}" for g in ("9.0k", "10.8k"))
            out.append(f"| {label} | {maps} | {fr('9.0k')} | {fr('10.8k')} | {fr('8996')} | {same} | "
                       f"{cost.get(key, '—')} | {verdict(key)} |")
        return out

    dba = "dba k=1 | neardup"
    lines = [f"# Постобработка по одному запросу на больших галереях: {tag}", "",
             f"Веса `{R['weights']}`, 5 фолдов по 150 машинам holdout, 25 % машин фолда без пары, метрики эталона. "
             "Главные галереи — 9.0k (holdout + 8639 кадров обучающих машин, по фолдам; состав как 8996), 10.8k и "
             "8996 (одна галерея команды; без пары в ней 7 запросов — её TNR не показатель). Порог отказа для каждой "
             "строки — максимум F1 при TNR ≥ 0.70, выбран на фолде этой же галереи и проверен на остальных; для 8996 — "
             "на том же фолде 9.0k (тот же состав; в 8996 у большинства фолдов нет запросов без пары). "
             "Воспроизвести: `python research/bigallery_pp.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag "
             f"{tag} --with-kr`, `python research/cheap_bg.py ...`, `python research/threshold_vs_gallery.py ...`, "
             "`python research/bench_bigallery_pp.py ...`, затем `--report`.", "",
             "## Как сервис: почти-дубли в галерее", "",
             "Все строки — один протокол: у запроса без пары в галерее остаётся кадр его машины с той же камеры, "
             "как у сервиса. Δ — к raw этого же протокола.", "",
             *table(rows, "raw | neardup"), "",
             "### Вклад штрафа поверх DBA k=1", "",
             "| Фон | Δ mAP@10 9.0k / 10.8k / 8996 | Δ F1 9.0k / 10.8k / 8996 | Δ TNR 9.0k / 10.8k | "
             "без пары → та же машина с той же камеры, 9.0k: DBA → DBA + штраф | доп. время на запрос | FPS b1 (оценка) |",
             "|---|---:|---:|---:|---:|---|---|"]
    gpu_add = gpu["with_background_ms"] - gpu["siglip_ms"]
    cpu_fps = 1000 / (gpu["siglip_ms"] + B["mac_ms"]["resnet18_448x256_forward"])
    pen_rows = [("dba k=1 + campen τ=p90 λ=inf | neardup", "resnet18 448×256",
                 f"CPU {B['mac_ms']['resnet18_448x256_forward']:.0f} мс (Мак, только прогон) или GPU ≈ +{gpu_add:.1f} мс (по FLOPs)",
                 f"{cpu_fps:.0f} на CPU / {gpu['fps_after']:.0f} на GPU")]
    if pick:
        k = C["kinds"][pick]
        pen_rows.append((f"dba k=1 + campen {pick} | neardup", f"дешёвый: {C['pick_label']}",
                         f"CPU {k['ms']:.2f} мс (Мак)", f"{k['fps']['b1']:.0f}"))
    for key, label, cost_, fps_ in pen_rows:
        d = lambda k, gs: " / ".join(f"{m(key, g, k) - m(dba, g, k):+.3f}" for g in gs)
        lines.append(f"| {label} | {d('mAP@10', big)} | {d('F1', big)} | {d('TNR', ('9.0k', '10.8k'))} | "
                     f"{r[dba]['9.0k']['open_same_cam_top1_all']} → {r[key]['9.0k']['open_same_cam_top1_all']} из "
                     f"{r[key]['9.0k']['n_open_all']} | {cost_} | {fps_} |")
    lines += ["", "## Без почти-дублей (оракул камер)", "",
              "Проверка, что прирост DBA — не почти-дубли: кадры той же машины с той же камеры убраны из галереи, "
              "а у DBA — и из соседей каждого кадра. Δ — к raw этого протокола.", "",
              *table(ORACLE_ROWS, "raw | nodup"), "",
              "## Порог отказа и размер галереи (без почти-дублей)", "",
              "| Правило | F1 / TNR 9.0k | F1 / TNR 10.8k | В прод |", "|---|---:|---:|---|"]
    nc = T["named_compare"]
    form_ru = "ln N" if T["form"] == "ln" else "√(2 ln N)"
    for rule, label in (("fixed", "порог 0.70 (сырой косинус), перенесён с 2.2k"),
                        ("thr(N)", f"порог thr(N) = a + b·{form_ru}"), ("z100", "z100, порог перенесён с 2.2k")):
        fr = lambda g: f"{nc[g][rule]['F1']['mean']:.3f} / {nc[g][rule]['TNR']['mean']:.3f}"
        lines.append(f"| {label} | {fr('9.0k')} | {fr('10.8k')} | {'нет' if rule == 'thr(N)' else 'см. ниже'} |")
    best = lambda prefix, g: max((k for k in r if k.startswith(prefix) and k.endswith("| nodup") and "+" not in k),
                                 key=lambda k: r[k][g]["mAP@10"]["mean"])
    lines += ["", "## Штраф за ту же камеру", "",
              f"- Сервис получает полный кадр и bbox (`/v1/search`, `/v1/gallery`: `image` + `x, y, w, h`, кроп делает "
              "сам `crop_with_pad`), так что фон доступен и для запроса, и для кадра галереи.",
              f"- «Та же камера» — косинус фоновых дескрипторов > τ. τ подобран по camera_id пар «кадр обучения — его "
              f"top-{R['topk']} по внешности» ({cal['pairs']} пар, той же камеры {cal['same_share']:.1%}), проверка по группам "
              f"камер: τ по F1 {cal['f1']['tau']:.3f} — точность {cal['f1']['cv_mean']['precision']:.2f}, полнота "
              f"{cal['f1']['cv_mean']['recall']:.2f}; τ с точностью ≥ 0.90 {cal['p90']['tau']:.3f} — "
              f"{cal['p90']['cv_mean']['precision']:.2f} / {cal['p90']['cv_mean']['recall']:.2f}. Детектор слабый: среди "
              "похожих машин фон соседних камер парковки неотличим, а τ по F1 вычитает и настоящие пары.",
              "- Сетка λ (0.05, 0.1, 0.2, 0.3, исключить) — в JSON; на больших галереях ни одно λ не поднимает mAP@10, "
              "лучшее — без потерь (τ точн. ≥ 0.90).",
              "- Эталон засчитывает top-1 той же машины с той же камеры как верный ответ у запроса с парой, поэтому "
              "штраф один снижает F1 (убирает «бесплатные» TP) и поднимает TNR; вместе с DBA F1 растёт.",
              *([f"- Дешёвый фон (`cheap_bg_{tag}.md`): пиксели уже декодированного кадра, ужатого до 128×72, машина "
                 f"выброшена маской. «{C['pick_label']}» — точность / полнота «та же камера» "
                 f"{C['kinds'][pick]['calibration']['p90']['cv_mean']['precision']:.2f} / "
                 f"{C['kinds'][pick]['calibration']['p90']['cv_mean']['recall']:.2f} (resnet18 "
                 f"{cal['p90']['cv_mean']['precision']:.2f} / {cal['p90']['cv_mean']['recall']:.2f}) за "
                 f"{C['kinds'][pick]['ms']:.2f} мс CPU вместо {B['mac_ms']['resnet18_448x256_forward']:.0f}+ мс."] if pick else []),
              ""
              "## DBA", "",
              "- Без почти-дублей соседи кадра галереи для каждого запроса берутся без его почти-дублей (иначе кадр той же "
              "машины с другой камеры впитывает почти-дубль запроса — обходной путь). Рост mAP@10 сохраняется и "
              f"так: лучший k по mAP — {best('dba k=', '9.0k').split(' |')[0]} на 9.0k, {best('dba k=', '10.8k').split(' |')[0]} "
              f"на 10.8k, {best('dba k=', '8996').split(' |')[0]} на 8996; k ≥ 4 прироста не даёт (все k — в JSON).",
              "- k выбран по тем же галереям, на которых мерится; защита — совпадение на трёх больших галереях и на 2.2k.",
              "", "## Порог от размера галереи", "",
              f"- thr(N) = {T['rule_mean']['a']:.3f} + {T['rule_mean']['b']:.4f}·{form_ru}: TNR держит около 0.70, но порог "
              f"растёт до {nc['10.8k']['rule_threshold']['mean']:.3f} и F1 на 10.8k падает до "
              f"{nc['10.8k']['thr(N)']['F1']['mean']:.3f}; z100 на тех же галереях лучше по обоим "
              f"({nc['10.8k']['z100']['F1']['mean']:.3f} / {nc['10.8k']['z100']['TNR']['mean']:.3f}). Порог зависит не только от N: "
              "при той же N на смеси с кадрами обучающих машин он выше, чем на одном тесте (`threshold_vs_gallery_"
              f"{tag}.md`).",
              "- После постобработки Андрея: `python research/threshold_vs_gallery.py --data-dir <данные> --env "
              "REID_WEIGHTS=<веса> --score-fn <файл.py>:<функция> --tag <имя>`.", "",
              "## Рекомендация", "",
              "- **Андрею и Глебу в прод — DBA k=1** по галерее (`research/dba.py`, как встроить — `HONEST_EVAL.md`): "
              "один пересчёт при старте и обновление при добавлении кадра, на запрос — ноль. Рост mAP@10 на всех больших "
              "галереях, в том числе без почти-дублей.",
              (f"- **Штраф за ту же камеру — вместе с DBA и на дешёвом фоне** («{C['pick_label']}»): поверх DBA F1 "
               f"{m(f'dba k=1 + campen {pick} | neardup', '9.0k', 'F1') - m(dba, '9.0k', 'F1'):+.3f} / "
               f"{m(f'dba k=1 + campen {pick} | neardup', '10.8k', 'F1') - m(dba, '10.8k', 'F1'):+.3f} на 9.0k / 10.8k, "
               f"+{C['kinds'][pick]['ms']:.1f} мс CPU, ≈ {C['kinds'][pick]['fps']['b1']:.0f} FPS при b1 (оценка). Фон "
               f"resnet18 не брать: пользы меньше (F1 {m('dba k=1 + campen τ=p90 λ=inf | neardup', '9.0k', 'F1') - m(dba, '9.0k', 'F1'):+.3f}"
               f" / {m('dba k=1 + campen τ=p90 λ=inf | neardup', '10.8k', 'F1') - m(dba, '10.8k', 'F1'):+.3f}) за "
               f"{B['mac_ms']['resnet18_448x256_forward']:.0f}+ мс CPU или ≈ {gpu['fps_after']:.0f} FPS на GPU. τ по F1 "
               "не брать: съедает настоящие пары." if pick else
               "- **Штраф за ту же камеру — только вместе с DBA и только если устраивает цена фона**: он почти убирает "
               "ответ «та же машина с той же камеры» запросам без пары и поднимает TNR, но сам mAP@10 не поднимает. "
               "τ по F1 не брать: съедает настоящие пары."),
              "- **Порог как функция N — не брать**: проигрывает z100. Финальный порог считать после постобработки "
              "Андрея тем же скриптом; на неизвестном размере закрытой выборки фиксированный порог держит F1, а TNR "
              "падает с ростом галереи — это надо проговорить на защите.", ""]
    return "\n".join(lines)


def data_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Данные, эмбеддинги и галереи — общие с research/cheap_bg.py."""
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--galleries", default="2.2k,9.0k,10.8k,8996")
    ap.add_argument("--pp-params", type=Path, default=he.ROOT / "artifacts" / "postprocess.json")
    ap.add_argument("--holdout", type=Path, default=HERE / "val_ids.txt")
    ap.add_argument("--exclude-frames", type=Path, default=HERE / "honest_eval_results" / "holdout_frame_dups.txt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--openset-frac", type=float, default=0.25)
    ap.add_argument("--profile", choices=se.PROFILES, default="env_example")
    ap.add_argument("--compose-ref", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--source", choices=("crops", "frames"), default="crops")
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--one-query-per-camera", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=se.DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=HERE / "honest_eval_results")
    return ap


def build_blocks(D, folds: list, want: list, bg: dict) -> dict:
    if "8996" in want and THR_FROM["8996"] not in want:
        raise SystemExit(f"8996 берёт пороги фолдов с {THR_FROM['8996']}: добавьте её в --galleries")
    blocks = {name: fold_blocks(D, kind, folds, bg) for name, kind in GALLERIES if name in want}
    if "8996" in want:
        tq, tg = he.team8996(D)
        blocks["8996"] = [make_block(D, D.valid(tq), D.valid(tg), set(), bg)]
    return blocks


def main() -> int:
    ap = data_args(argparse.ArgumentParser())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bg", default="emb", choices=("emb", "grad", "both"))
    ap.add_argument("--with-kr", action="store_true", help="добавить справочные kr и dba + kr (долго на 10.8k)")
    ap.add_argument("--report", action="store_true", help="только собрать bigallery_pp_<tag>.md из JSON")
    a = ap.parse_args()
    if a.report:
        (a.out_dir / f"bigallery_pp_{a.tag}.md").write_text(report_md(a.out_dir, a.tag), encoding="utf-8")
        print(f"-> {a.out_dir / f'bigallery_pp_{a.tag}.md'}")
        return 0

    D = he.Data(a)
    params = he.pp_params(a.pp_params)
    folds = he.folds_by_vehicle(sorted(D.hold), a.folds, a.seed)
    bg = {kind: load_bg(a.cache_dir, kind) for kind in ("emb", "grad", "both")}
    calib = {kind: calibrate_tau(D, bg[kind]) for kind in bg}
    tau = {rule: calib[a.bg][rule]["tau"] for rule in ("f1", "p90")}
    for kind, v in calib.items():
        print(f"τ по фону {kind}: " + "; ".join(
            f"{rule} τ {v[rule]['tau']:.4f} (группы камер: P {v[rule]['cv_mean']['precision']:.3f} "
            f"R {v[rule]['cv_mean']['recall']:.3f})" for rule in ("f1", "p90")), flush=True)

    blocks = build_blocks(D, folds, a.galleries.split(","), bg[a.bg])
    # сверка: nodup + raw совпадает с honest_eval (mAP@10 и записи отказа)
    if "2.2k" in blocks:
        ref = crf.fold_records(D, "test_all", "nodup")
        for b, r in zip(blocks["2.2k"], ref):
            e = evaluate(b, scores(b, {"dba": 0, "kr": False, "lambda": 0}, "nodup", tau, params), "nodup")
            if not (np.allclose(e["rec"]["conf"], r["conf"]) and np.array_equal(e["rec"]["correct"], r["correct"])
                    and np.array_equal(e["rec"]["has_match"], r["has_match"])):
                raise AssertionError("nodup + raw не совпадает с calibrate_refusal_folds")
        print("сверка с honest_eval (nodup, raw): ок", flush=True)

    variants = [("raw", {"dba": 0, "kr": False, "lambda": 0}, ("neardup", "nodup"))]
    variants += [(f"campen τ={rule} λ={l:g}", {"dba": 0, "kr": False, "lambda": l, "tau": rule}, ("neardup",))
                 for rule in ("f1", "p90") for l in LAMBDAS]
    variants += [(f"dba k={k}", {"dba": k, "kr": False, "lambda": 0}, ("nodup", "neardup")) for k in DBA_KS]
    variants += [(f"dba k={k} + campen τ={rule} λ=inf", {"dba": k, "kr": False, "lambda": float("inf"), "tau": rule},
                  ("neardup",)) for rule in ("f1", "p90") for k in DBA_KS]
    if a.with_kr:
        variants += [("kr", {"dba": 0, "kr": True, "lambda": 0}, ("nodup",))]
        variants += [(f"dba k={k} + kr", {"dba": k, "kr": True, "lambda": 0}, ("nodup",)) for k in (1,)]
    res = {}
    for name, var, modes in variants:
        for mode in modes:
            for gname, bl in blocks.items():
                t0 = time.time()
                src = res.get(f"{name} | {mode}", {}).get(THR_FROM.get(gname))
                res.setdefault(f"{name} | {mode}", {})[gname] = run(bl, folds, var, mode, tau, params,
                                                                    src["thresholds"] if src else None)
                r = res[f"{name} | {mode}"][gname]
                print(f"{name:28s} {mode:7s} {gname:5s} mAP@10 {r['mAP@10']['mean']:.4f} R1 {r['Rank-1']['mean']:.4f} "
                      f"F1 {r['F1']['mean']:.3f} TNR {r['TNR']['mean']:.3f} R {r['recall']['mean']:.3f} "
                      f"open→same-cam {r['open_same_cam_top1_all']}/{r['n_open_all']}  {time.time() - t0:.0f} с",
                      flush=True)
    w = {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~") for k, v in D.env.items()}
    report = {"tag": a.tag, "weights": w.get("REID_WEIGHTS", ""), "bg": a.bg, "tau": tau, "tau_calibration": calib,
              "topk": TOPK,
              "results": res}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"bigallery_pp_{a.tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(f"-> {a.out_dir / f'bigallery_pp_{a.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
