#!/usr/bin/env python3
"""Правило Андрея (почти-дубль = высокое сходство И та же рамка) против штрафа по фону.

1. Утверждение: доля запросов с парой и без, у которых сырой top-1 — та же машина с той
   же камеры, в условиях сервиса (у запроса без пары в галерее остаётся кадр его машины с
   той же камеры), 2.2k / 8996 / 10.8k.
2. Калибровка правила по camera_id обучающих машин: пары «кадр обучения — его top-100
   по внешности»; почти-дубль, если сырой косинус ≥ s_hi и рамки близки (IoU ≥ t_iou,
   сдвиг центра ≤ t_center, размеры ≤ exp(t_size)) — api/app/pp/dedup.py. Цель — та же
   машина + та же камера. Выбор — как у τ штрафа: наибольшая полнота при точности ≥ 0.90
   на группах камер выбора, проверка на отложенной группе камер.
3. Сравнение (a10, как сервис, DBA k=1 и без DBA): без исключения / правило Андрея /
   штраф по фону / оба (OR) / оба (AND). Почти-дубли исключаются из ранжирования (в конец
   списка) и из оценки отказа. mAP@10, R1, R5; F1 / TNR при пороге вне фолда (F1 при
   TNR ≥ 0.70 на фолде той же галереи; 8996 — с фолда 9.0k), оценка raw и z100; у
   запросов без пары top-1 той же машины с той же камеры; время на запрос.

    python research/dedup_compare.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10 --zip dataset.zip
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "api"))
import bigallery_pp as bp  # noqa: E402
import honest_eval as he  # noqa: E402
import pseudo_camera as pc  # noqa: E402
import refusal_score as rsc  # noqa: E402
import service_score as ss  # noqa: E402
import threshold_vs_gallery as tvg  # noqa: E402
from app.pp import dedup as dd  # noqa: E402
from app.pp import rerank as rr  # noqa: E402

K = 100
KINDS = (("none", "без исключения"), ("bbox", "правило Андрея (сходство + рамка)"), ("bg", "штраф по фону"),
         ("or", "оба, OR"), ("and", "оба, AND"))
GALLERIES = ("2.2k", "9.0k", "10.8k", "8996")
S_HI = tuple(np.round(np.arange(0.70, 0.995, 0.01), 3))
T_IOU = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
T_CENTER = (0.05, 0.1, 0.2, 0.3, 0.5, float("inf"))
T_SIZE = (0.05, 0.1, 0.2, 0.3, float("inf"))
P_FLOOR = 0.90


# --------------------------------------------------------------------------
# калибровка правила по обучающим машинам
# --------------------------------------------------------------------------

def train_pairs(D, boxes: dict, bg: dict, n_queries: int = 3000, seed: int = 0) -> dict:
    tr = [r for r in D.valid(D.train) if r["vehicle_id"] not in D.hold]
    qi = np.random.default_rng(seed).choice(len(tr), size=min(n_queries, len(tr)), replace=False)
    ids = [r["image_id"] for r in tr]
    v = D.vectors(ids)
    box = np.array([boxes[i] for i in ids], dtype="float64")
    B = np.stack([bg[i] for i in ids])
    cam = np.array([r["camera_id"] for r in tr])
    vid = np.array([r["vehicle_id"] for r in tr])
    sim = v[qi] @ v.T
    sim[np.arange(len(qi)), qi] = -np.inf
    top = np.argpartition(-sim, K - 1, axis=1)[:, :K]
    f = dd.box_features(box[qi][:, None, :], box[top])
    same_c, same_v = cam[top] == cam[qi][:, None], vid[top] == vid[qi][:, None]
    return {"cos": np.take_along_axis(sim, top, 1).ravel(), **{k: f[k].ravel() for k in ("iou", "center", "size", "ok")},
            "bgcos": ss.pair_cos(B[qi], B, top).ravel(), "nd": (same_v & same_c).ravel(), "same_c": same_c.ravel(),
            "pos": (same_v & ~same_c).ravel(), "qcam": np.repeat(cam[qi], K)}


def pr(flag: np.ndarray, target: np.ndarray) -> dict:
    tp, fp, fn = int((flag & target).sum()), int((flag & ~target).sum()), int((~flag & target).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0, "flagged": tp + fp}


def best_rule(P: dict, sel: np.ndarray) -> dict:
    """Наибольшая полнота почти-дублей при точности ≥ P_FLOOR на подвыборке sel."""
    nd = P["nd"][sel]
    c = {s: P["cos"][sel] >= s for s in S_HI}
    i = {t: (P["iou"][sel] >= t) & P["ok"][sel] for t in T_IOU}
    m = {t: P["center"][sel] <= t for t in T_CENTER}
    z = {t: P["size"][sel] <= t for t in T_SIZE}
    best, best_key = None, None
    for t_iou in T_IOU:
        for t_c in T_CENTER:
            geo = i[t_iou] & m[t_c]
            for t_s in T_SIZE:
                g2 = geo & z[t_s]
                for s_hi in S_HI:
                    q = pr(c[s_hi] & g2, nd)
                    if q["precision"] < P_FLOOR:
                        continue
                    key = (q["recall"], q["precision"])
                    if best_key is None or key > best_key:
                        best_key, best = key, {"s_hi": float(s_hi), "t_iou": t_iou, "t_center": t_c, "t_size": t_s}
    return best or {"s_hi": 1.01, "t_iou": 1.0, "t_center": 0.0, "t_size": 0.0}


def flag_rule(P: dict, rule: dict, sel=slice(None)) -> np.ndarray:
    f = {k: P[k][sel] for k in ("iou", "center", "size", "ok")}
    return dd.rule_from_features(P["cos"][sel], f, rule["s_hi"], rule["t_iou"], rule["t_center"], rule["t_size"])


def calibrate(D, boxes: dict, bg: dict, tau: float, folds: int = 5) -> dict:
    P = train_pairs(D, boxes, bg)
    cams = np.array(sorted(set(P["qcam"]), key=int))
    groups = np.array_split(np.random.default_rng(1).permutation(cams), folds)       # как bigallery_pp.calibrate_tau
    cv, cv_bg = [], []
    for g in groups:
        test = np.isin(P["qcam"], g)
        rule = best_rule(P, ~test)
        cv.append({"rule": rule, **pr(flag_rule(P, rule, test), P["nd"][test]),
                   "removes_true_pair": float((flag_rule(P, rule, test) & P["pos"][test]).sum() / max(P["pos"][test].sum(), 1))})
        fb = P["bgcos"][test] > tau
        cv_bg.append({**pr(fb, P["nd"][test]), "removes_true_pair": float((fb & P["pos"][test]).sum() / max(P["pos"][test].sum(), 1))})
    rule = best_rule(P, np.ones(len(P["nd"]), bool))
    mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
    keys = ("precision", "recall", "F1", "removes_true_pair")
    fx, fb = flag_rule(P, rule), P["bgcos"] > tau
    return {"pairs": int(len(P["nd"])), "near_dup_share": float(P["nd"].mean()), "rule": rule,
            "bbox_cv": {k: mean(cv, k) for k in keys}, "bbox_cv_folds": cv, "bg_cv": {k: mean(cv_bg, k) for k in keys},
            "bbox_all": pr(fx, P["nd"]), "bg_all": pr(fb, P["nd"]),
            "same_camera_target": {"bbox": pr(fx, P["same_c"]), "bg": pr(fb, P["same_c"])},
            "near_dup_geometry": {k: {"median": float(np.median(P[k][P["nd"] & P["ok"]])),
                                      "p10": float(np.percentile(P[k][P["nd"] & P["ok"]], 10)),
                                      "p90": float(np.percentile(P[k][P["nd"] & P["ok"]], 90))}
                                  for k in ("cos", "iou", "center", "size")}}


# --------------------------------------------------------------------------
# сравнение на галереях
# --------------------------------------------------------------------------

def block_flags(b: dict, boxes: dict, dba_k: int, rule: dict, tau: float) -> dict:
    raw = b["qv"] @ b["gv"].T
    s = bp.dba_scores(b, dba_k, clean=False) if dba_k else raw.copy()
    s = np.where(b["mask"]["neardup"], -np.inf, s)
    top, cos = ss.topk(s, K)
    fin = np.isfinite(cos)
    qb = np.array([boxes[i] for i in b["Q"].index], dtype="float64")
    gb = np.array([boxes[i] for i in b["G"].index], dtype="float64")
    fb = (ss.pair_cos(b["qbg"], b["gbg"], top) > tau) & fin
    fx = dd.near_dup_flags(np.take_along_axis(raw, top, 1), qb[:, None, :], gb[top], rule["s_hi"], rule["t_iou"],
                           rule["t_center"], rule["t_size"]) & fin
    return {"s": s, "top": top, "cos": cos, "flags": {"none": np.zeros_like(fb), "bbox": fx, "bg": fb, "or": fb | fx,
                                                       "and": fb & fx}}


def block_records(b: dict, bf: dict, kind: str) -> dict:
    flags = bf["flags"][kind]
    r = ss.decide(bf["top"], bf["cos"], flags)
    pen = bf["s"].copy()
    rows, cols = np.nonzero(flags)
    pen[rows, bf["top"][rows, cols]] -= bp.BIG
    ev = bp.evaluate(b, pen, "neardup")
    n = len(pen)
    top1 = np.array([o[0] if len(o) else 0 for o in r["order"]])
    junk = b["same_v"] & b["same_c"]
    pos = b["same_v"] & ~b["same_c"] & ~b["mask"]["neardup"]
    in_top = np.take_along_axis(pos, bf["top"], 1)
    base = {"has_match": ev["rec"]["has_match"], "correct": b["same_v"][np.arange(n), top1],
            "same_cam_top1": junk[np.arange(n), top1]}
    return {"mAP@10": ev["mAP@10"], "Rank-1": ev["Rank-1"], "Rank-5": ev["Rank-5"],
            "raw": {**base, "conf": r["conf_raw"]}, "z100": {**base, "conf": r["conf_z100"]},
            "flagged_per_query": float(flags.sum(1).mean()),
            "true_pairs_removed": int((flags & in_top).sum()), "true_pairs_in_topk": int(in_top.sum())}


def run_gallery(blocks: list, folds: list, recs: list, thresholds: dict = None) -> dict:
    out = {k: rsc.stats([r[k] for r in recs]) for k in ("mAP@10", "Rank-1", "Rank-5", "flagged_per_query")}
    out["true_pairs_removed"] = int(sum(r["true_pairs_removed"] for r in recs))
    out["true_pairs_in_topk"] = int(sum(r["true_pairs_in_topk"] for r in recs))
    vehicles = [b["vehicle"] for b in blocks]
    for conf in ("raw", "z100"):
        out[conf] = bp.run_records([r[conf] for r in recs], vehicles, folds,
                                   thresholds[conf]["thresholds"] if thresholds else None)
    return out


def claim(blocks: dict, flags: dict) -> dict:
    """Доля запросов с top-1 = та же машина с той же камеры, без исключения (п. 1)."""
    out = {}
    for g, bl in blocks.items():
        for dba_k in (0, 1):
            w, wo, nw, nwo = 0, 0, 0, 0
            for b, bf in zip(bl, flags[g][dba_k]):
                top1 = bf["top"][:, 0]
                junk = (b["same_v"] & b["same_c"])[np.arange(len(top1)), top1]
                pos = (b["same_v"] & ~b["same_c"] & ~b["mask"]["neardup"]).any(1)
                w += int((junk & pos).sum()); nw += int(pos.sum())
                wo += int((junk & ~pos).sum()); nwo += int((~pos).sum())
            out[f"{g} | dba {dba_k}"] = {"with_pair": w / nw if nw else float("nan"), "n_with_pair": nw,
                                          "without_pair": wo / nwo if nwo else float("nan"), "n_without_pair": nwo,
                                          "all": (w + wo) / (nw + nwo)}
    return out


def bench(rule: dict, n: int = 2000) -> dict:
    """Время правила Андрея на запрос: сырой косинус с K кандидатами (512) и геометрия рамок."""
    rng = np.random.default_rng(0)
    g = rng.standard_normal((K, 512)).astype("float32")
    q = rng.standard_normal(512).astype("float32")
    boxes = np.abs(rng.normal(500, 200, (K, 4)))
    qbox = boxes[0]
    t0 = time.perf_counter()
    for _ in range(n):
        dd.near_dup_flags(g @ q, qbox, boxes, rule["s_hi"], rule["t_iou"], rule["t_center"], rule["t_size"])
    return {"bbox_rule_ms": (time.perf_counter() - t0) * 1000 / n}


def main() -> int:
    ap = bp.data_args(argparse.ArgumentParser())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    out_json = a.out_dir / f"dedup_compare_{a.tag}.json"
    if not a.report:
        from app.config import Settings

        tau = Settings().pp_camera_tau
        D = he.Data(a)
        folds = he.folds_by_vehicle(sorted(D.hold), a.folds, a.seed)
        boxes = {r["image_id"]: r["box"] for r in pc.read_rows(a.data_dir)}
        bg = tvg.service_bg(D, a)
        cal = calibrate(D, boxes, bg, tau)
        rule = cal["rule"]
        print("правило Андрея:", rule, "CV P/R", {k: round(v, 3) for k, v in cal["bbox_cv"].items()},
              "| фон CV P/R", {k: round(v, 3) for k, v in cal["bg_cv"].items()}, flush=True)
        blocks = bp.build_blocks(D, folds, list(GALLERIES), bg)
        flags = {g: {k: [block_flags(b, boxes, k, rule, tau) for b in bl] for k in (0, 1)} for g, bl in blocks.items()}
        res = {"tag": a.tag, "tau": tau, "rule": rule, "calibration": cal, "claim": claim(blocks, flags),
               "bench": bench(rule), "results": {}}
        for dba_k in (1, 0):
            for kind, _ in KINDS:
                key = f"dba{dba_k} | {kind}"
                for g in GALLERIES:
                    recs = [block_records(b, bf, kind) for b, bf in zip(blocks[g], flags[g][dba_k])]
                    src = res["results"].get(key, {}).get(bp.THR_FROM.get(g))
                    res["results"].setdefault(key, {})[g] = run_gallery(blocks[g], folds, recs, src)
                    r = res["results"][key][g]
                    print(f"{key:16s} {g:5s} mAP@10 {r['mAP@10']['mean']:.4f} R1 {r['Rank-1']['mean']:.3f} "
                          f"R5 {r['Rank-5']['mean']:.3f} | raw F1 {r['raw']['F1']['mean']:.3f} TNR {r['raw']['TNR']['mean']:.3f}"
                          f" | z100 F1 {r['z100']['F1']['mean']:.3f} TNR {r['z100']['TNR']['mean']:.3f} | без пары→та же "
                          f"{r['raw']['open_same_cam_top1_all']}/{r['raw']['n_open_all']}", flush=True)
        # сверка: DBA k=1 + bg совпадает с cheap_bg (тот же скор и порог raw)
        ref = json.loads((a.out_dir / f"cheap_bg_{a.tag}.json").read_text(encoding="utf-8"))["results"]
        for g in GALLERIES:
            mine, theirs = res["results"]["dba1 | bg"][g], ref["dba k=1 + campen grad64+hist | neardup"][g]
            for k in ("mAP@10", "Rank-1", "Rank-5"):
                if abs(mine[k]["mean"] - theirs[k]["mean"]) > 1e-9:
                    raise AssertionError(f"dba1 | bg {g} {k}: {mine[k]['mean']} != cheap_bg {theirs[k]['mean']}")
            for k in ("F1", "TNR"):
                if abs(mine["raw"][k]["mean"] - theirs[k]["mean"]) > 1e-9:
                    raise AssertionError(f"dba1 | bg {g} {k}: {mine['raw'][k]['mean']} != cheap_bg {theirs[k]['mean']}")
        print("сверка DBA k=1 + штраф по фону с cheap_bg_a10.json — ок", flush=True)
        w = {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~") for k, v in D.env.items()}
        res["weights"] = w.get("REID_WEIGHTS", "")
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    res = json.loads(out_json.read_text(encoding="utf-8"))
    (a.out_dir / f"dedup_compare_{a.tag}.md").write_text(report_md(res), encoding="utf-8")
    print(f"-> {a.out_dir / f'dedup_compare_{a.tag}.md'}")
    return 0


def report_md(res: dict) -> str:
    tag, cal, rule, R = res["tag"], res["calibration"], res["rule"], res["results"]
    f3 = lambda x: f"{x:.3f}"
    m = lambda key, g, k: R[key][g][k]["mean"]
    rc = lambda key, g, c: f"{f3(R[key][g][c]['F1']['mean'])} / {f3(R[key][g][c]['TNR']['mean'])}"
    rule_txt = (f"сырой косинус ≥ {rule['s_hi']:.2f} и IoU ≥ {rule['t_iou']:.1f}, сдвиг центра ≤ {rule['t_center']:.2f} "
                f"среднего размера рамки" + ("" if rule["t_size"] == float("inf") else
                                               f", размеры ≤ ×{np.exp(rule['t_size']):.2f}"))
    L = [f"# Правило Андрея против штрафа по фону: {tag}", "",
         f"Веса `{res['weights']}`. Условия как у сервиса: у запроса без пары в галерее остаётся кадр его машины с той же "
         "камеры. 150 машин holdout, 5 фолдов по 30, 25 % машин фолда без пары. Порог отказа — F1 при TNR ≥ 0.70 на "
         "фолде той же галереи, проверка на остальных; для 8996 — с того же фолда 9.0k. Почти-дубли исключаются из "
         "ранжирования (в конец списка) и из оценки отказа. Воспроизвести: `python research/dedup_compare.py --data-dir "
         f"<данные> --env REID_WEIGHTS=<веса> --tag {tag} --zip dataset.zip`.", "",
         "## 1. Утверждение: сырой top-1 — та же машина с той же камеры", "",
         "В галерее holdout по одному кадру на пару (машина, камера), почти-дубль со своей камеры есть у каждого запроса.", "",
         "| Галерея | Постобработка | с парой | без пары | все |", "|---|---|---:|---:|---:|"]
    for k, v in res["claim"].items():
        g, d = k.split(" | ")
        L.append(f"| {g} | {'DBA k=1' if d.endswith('1') else 'без DBA'} | {v['with_pair']:.1%} ({v['n_with_pair']}) | "
                 f"{v['without_pair']:.1%} ({v['n_without_pair']}) | {v['all']:.1%} |".replace("%", " %"))
    L += ["", "## 2. Калибровка правила по обучающим машинам", "",
          f"Пары «кадр обучения — его top-100 по внешности»: {cal['pairs']:,} пар, почти-дублей (та же машина + та же "
          f"камера) {cal['near_dup_share']:.1%}. Выбор — наибольшая полнота при точности ≥ 0.90 на группах камер выбора, "
          "проверка на отложенной группе (5 групп).".replace(",", " "), "",
          f"- **Правило:** {rule_txt}.",
          f"- Почти-дубли в train: косинус медиана {cal['near_dup_geometry']['cos']['median']:.3f} (10 % ниже "
          f"{cal['near_dup_geometry']['cos']['p10']:.3f}), IoU медиана {cal['near_dup_geometry']['iou']['median']:.3f} "
          f"(10 % ниже {cal['near_dup_geometry']['iou']['p10']:.3f}).", "",
          "| Детектор почти-дубля (вне фолда, группы камер) | Точность | Полнота | F1 | Убирает настоящих пар |",
          "|---|---:|---:|---:|---:|"]
    for name, k in (("правило Андрея", "bbox_cv"), ("признак фона (штраф, τ)", "bg_cv")):
        v = cal[k]
        L.append(f"| {name} | {f3(v['precision'])} | {f3(v['recall'])} | {f3(v['F1'])} | {v['removes_true_pair']:.1%} |"
                 .replace("%", " %"))
    L += ["", "Фон метит «ту же камеру» (любую машину), поэтому его точность для почти-дублей ниже; чужие машины с той "
          "же камеры у запроса с парой — тоже неверные ответы, их удаление пар не трогает.", "",
          "## 3. Сравнение (a10, как сервис)", "",
          "| Постобработка | Исключение | mAP@10 2.2k / 8996 / 10.8k | R1 / R5 10.8k | F1 / TNR raw 10.8k | F1 / TNR z100 "
          "10.8k | F1 raw / z100 8996 | F1 / TNR z100 2.2k | без пары → та же машина с той же камеры, 2.2k / 10.8k | "
          "Время на запрос |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    cost = {"none": "0", "bbox": f"{res['bench']['bbox_rule_ms']:.2f} мс", "bg": "1.4 мс",
            "or": f"{1.4 + res['bench']['bbox_rule_ms']:.2f} мс", "and": f"{1.4 + res['bench']['bbox_rule_ms']:.2f} мс"}
    for d in (1, 0):
        for kind, label in KINDS:
            key = f"dba{d} | {kind}"
            L.append(f"| {'DBA k=1' if d else 'без DBA (справочно)'} | {label} | "
                     f"{f3(m(key, '2.2k', 'mAP@10'))} / {f3(m(key, '8996', 'mAP@10'))} / {f3(m(key, '10.8k', 'mAP@10'))} | "
                     f"{f3(m(key, '10.8k', 'Rank-1'))} / {f3(m(key, '10.8k', 'Rank-5'))} | {rc(key, '10.8k', 'raw')} | "
                     f"{rc(key, '10.8k', 'z100')} | {f3(R[key]['8996']['raw']['F1']['mean'])} / "
                     f"{f3(R[key]['8996']['z100']['F1']['mean'])} | {rc(key, '2.2k', 'z100')} | "
                     f"{R[key]['2.2k']['raw']['open_same_cam_top1_all']} / {R[key]['10.8k']['raw']['open_same_cam_top1_all']}"
                     f" из {R[key]['10.8k']['raw']['n_open_all']} | {cost[kind]} |")
    b, f = R["dba1 | bbox"], R["dba1 | bg"]
    L += ["", f"Разброс по фолдам: F1 ± 0.05–0.12, TNR ± 0.12–0.21 — разница правила и фона на отказе в его пределах. "
          f"На 8996 без пары 7 запросов, TNR там не показатель. Правило убирает настоящих пар из top-100: "
          f"{b['10.8k']['true_pairs_removed']} из {b['10.8k']['true_pairs_in_topk']} на 10.8k, фон — "
          f"{f['10.8k']['true_pairs_removed']}. Время: правило — сырой косинус со 100 кандидатами и геометрия рамок "
          "(Мак, в процессе), фон — признак кадра и 100 косинусов (linux-контейнер, `api/scripts/bench_bg_cpu.py`).", "",
          "## 4. Может ли сервис применить правило", "",
          "- **Рамка кадра галереи хранится:** `gallery.bbox INTEGER[4] NOT NULL` (`db/init/01_schema.sh`, "
          "`db.ensure_schema`). Её пишут `/v1/gallery` (поля x, y, w, h) и bulk-import (колонки x, y, w, h из CSV; "
          "в `crops/crops_*.csv` они есть). Если в CSV их нет, пишется `[0, 0, 0, 0]` — такая рамка почти-дублем не "
          "считается.",
          "- **Рамка запроса приходит в `/v1/search`** (x, y, w, h) и уже передаётся в постобработку; `search_pp` "
          "возвращает `bbox` кандидатов.",
          "- **Сырой косинус с DBA** берётся из исходных векторов в памяти (`DBAGallery.originals`) — лишнего запроса к "
          "базе нет; без DBA это и есть скор поиска.",
          "- **Не хранится разрешение кадра.** IoU считается в пикселях: почти-дубль по определению снят той же камерой "
          "(то же разрешение), а у разных камер рамки и так не совпадут. Если кадры одной камеры приходят в разном "
          "разрешении, нужен размер кадра в базе — сейчас в датасете все кадры 1920×1080.",
          "- **Включается** `PP_DEDUP=bbox` (или `both` — OR с фоном) в `api/app/config.py`; пороги правила — "
          "`PP_DEDUP_SIM`, `PP_DEDUP_IOU`, `PP_DEDUP_CENTER`, `PP_DEDUP_SIZE` (значения по умолчанию — из этой "
          "калибровки, тест сверяет). Порог отказа для каждого варианта — `REFUSAL_PRESETS`.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
