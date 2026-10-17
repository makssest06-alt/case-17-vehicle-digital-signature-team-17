#!/usr/bin/env python3
"""Галереи 361 и 1111 как в закрытом тесте — протокол Андрея (head_to_head.py ветки
andrew-postprocess-throughput), наши эмбеддинги и код сервиса.

- Holdout как у организаторов: первый кадр каждой пары (машина, камера) → галерея,
  остальные → запросы: 546 запросов / 361 кадр (5 побайтных дублей train↔holdout убраны).
- Галереи из невиданных машин: 361 и 1111 (+ 750 кадров test gallery). 2.2k (+ весь тест)
  и 10.8k (+ кадры обучающих машин) — справочно: Андрей показал, что они завышают DBA.
- Ранжирование: все запросы против всей галереи, top-10, junk вычищает официальный
  evaluate (research/official_evaluate.ranking_metrics).
- Отказ «как сервис»: 4 ротации; в каждой у 25 % машин в галерее остаётся кадр одной камеры
  (самой частой среди их запросов) — их запросы с этой камеры без пары, но с почти-дублем.
  Порог вне фолда: 2 фолда по машинам, на одном — максимум F1 при TNR ≥ 0.70 (сетка — 201
  квантиль оценки), проверка на другом; F1/TNR/PR-AUC — среднее по фолдам и ротациям.
  Сиды ротаций и фолдов — как у Андрея (np.random.default_rng(42), две перестановки).
- Оценка отказа z100 по кандидатам без почти-дублей. Два способа top-100:
    andrey  — top-100 после выталкивания помеченных вниз (как head_to_head.py);
    service — top-100 по скору, помеченные выброшены (как api/app/pp/state.py).
  Сверка с Андреем — andrey; пресеты сервиса — service.
- Финальная схема: ранжирование = DBA k=1 + фон (рамки в ранжирование не идут),
  отказ = z100 без почти-дублей «фон ИЛИ рамка». Порог пресета — выбор вне фолда сразу по
  361 и 1111: TNR ≥ 0.70 на фолде выбора КАЖДОЙ галереи, из таких — максимум среднего F1;
  в пресет — медиана по фолдам × ротациям, проверка на отложенных фолдах с ней.

    python research/closed_gallery.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10 --zip dataset.zip
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "api"))
import bigallery_pp as bp  # noqa: E402
import honest_eval as he  # noqa: E402
import official_evaluate as oe  # noqa: E402
import threshold_vs_gallery as tvg  # noqa: E402
from app.pp import dedup as dd  # noqa: E402
from app.pp import rerank as rr  # noqa: E402
from app.pp.dba import dba  # noqa: E402

PUSH, Z_TOP, MIN_TNR, STRICT_TNR = 1e3, 100, 0.70, 0.98
RULES = {"ours": {"s_any": float("inf"), "s_hi": 0.84, "t_iou": 0.6, "t_center": 0.3, "t_size": float("inf")},
         "andrey": {"s_any": 0.94, "s_hi": 0.85, "t_iou": 0.7, "t_center": float("inf"), "t_size": float("inf")}}
GALLERIES = ("361", "1111", "2.2k", "10.8k")
MAIN = ("361", "1111")
FINAL = "итог: DBA + фон; отказ фон ИЛИ рамка (наши пороги)"
FALLBACK = "запасной: DBA; отказ рамка (наши пороги)"
# ключ REFUSAL_PRESETS: (PP_DBA_K, PP_DEDUP_RANK, PP_DEDUP_REFUSAL, формула) — пороги рамок сервиса = наши
PRESET_KEYS = {"raw": (0, "off", "off", "z100"), "DBA": (1, "off", "off", "z100"), "DBA + фон": (1, "bg", "bg", "z100"),
               FINAL: (1, "bg", "both", "z100"), FALLBACK: (1, "off", "bbox", "z100")}
# (имя, скор, флаги ранжирования, флаги отказа)
VARIANTS = (("raw", "raw", "none", "none"), ("DBA", "dba", "none", "none"),
            ("DBA + фон", "dba", "bg", "bg"),
            ("DBA + дубли Андрея (и в ранжировании)", "dba", "dup_andrey", "dup_andrey"),
            ("итог: DBA + фон; отказ фон ИЛИ рамка (наши пороги)", "dba", "bg", "bg|dup_ours"),
            ("итог: DBA + фон; отказ фон ИЛИ рамка (пороги Андрея)", "dba", "bg", "bg|dup_andrey"),
            ("запасной: DBA; отказ рамка (наши пороги)", "dba", "none", "dup_ours"),
            ("запасной: DBA; отказ рамка (пороги Андрея)", "dba", "none", "dup_andrey"))


def decision(acc, has, correct) -> dict:
    """candidate_metrics эталона на массивах (как head_to_head.decision_metrics)."""
    tp = int((acc & has & correct).sum()); fp = int((acc & ~(has & correct)).sum())
    fn = int((~acc & has).sum()); tn = int((~acc & ~has).sum()); fp_open = int((acc & ~has).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"F1": 2 * p * r / (p + r) if p + r else 0.0, "TNR": tn / (tn + fp_open) if tn + fp_open else float("nan"),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def choose(score, has, correct, floor=MIN_TNR, strict=False) -> float:
    """Порог как у Андрея: сетка 201 квантиль; F1-максимум при TNR ≥ floor (strict — наименьший с TNR ≥ floor)."""
    fin = score[np.isfinite(score)]
    best_t, best = np.inf, -1.0
    for t in (np.unique(np.quantile(fin, np.linspace(0, 1, 201))) if len(fin) else []):
        m = decision(score >= t, has, correct)
        if strict:
            if m["TNR"] >= floor:
                return float(t)
        elif m["TNR"] >= floor and m["F1"] > best:
            best_t, best = float(t), m["F1"]
    return best_t


def choose_joint(parts: list, floor=MIN_TNR, strict=False) -> float:
    """Порог по нескольким галереям сразу: TNR ≥ floor на КАЖДОЙ, из таких — максимум среднего F1
    (strict — наименьший порог, где TNR ≥ floor на каждой). parts — [(score, has, correct)]."""
    fin = np.concatenate([p[0][np.isfinite(p[0])] for p in parts])
    best_t, best = np.inf, -1.0
    for t in (np.unique(np.quantile(fin, np.linspace(0, 1, 201))) if len(fin) else []):
        ms = [decision(sc >= t, hs, co) for sc, hs, co in parts]
        if not all(m["TNR"] >= floor for m in ms):
            continue
        if strict:
            return float(t)
        f1 = float(np.mean([m["F1"] for m in ms]))
        if f1 > best:
            best_t, best = float(t), f1
    return best_t


def rotation_records(s, fl_rank, fl_ref, keep, qv, qc, gv, gc, z_mode) -> dict:
    """Записи отказа одной ротации: оценка, есть ли пара, верен ли top-1 ответа."""
    k = np.flatnonzero(keep)
    S, FR, FF = s[:, k], fl_rank[:, k], fl_ref[:, k]
    top1 = np.argmax(S - PUSH * FR, axis=1)                         # ответ — по флагам ранжирования
    if z_mode == "andrey":
        order = np.argsort(-(S - PUSH * FF), axis=1, kind="stable")[:, :Z_TOP]
    else:                                                           # сервис: top-100 по скору, потом флаги
        order = np.argsort(-S, axis=1, kind="stable")[:, :Z_TOP]
    # refusal_score отбрасывает помеченных и берёт лучших из оставшихся — порядок уже по убыванию
    score = np.array([rr.refusal_score(S[i, order[i]], FF[i, order[i]], "z100") for i in range(len(S))])
    score_raw = np.array([rr.refusal_score(S[i, order[i]], FF[i, order[i]], "raw") for i in range(len(S))])
    gvk, gck = gv[k], gc[k]
    has = np.array([((gvk == qv[i]) & (gck != qc[i])).any() for i in range(len(S))])
    correct = gvk[top1] == qv
    return {"score": score, "score_raw": score_raw, "has": has, "correct": correct, "top1": k[top1]}


def evaluate_refusal(recs: list, qfold: np.ndarray, thresholds=None, key: str = "score") -> dict:
    """recs — по ротациям; порог выбирается на фолде tune, проверяется на held (или задан).
    key — какая оценка: score (z100) или score_raw."""
    f1, tnr, auc, thr, strict, s_tnr, s_rec = [], [], [], [], [], [], []
    recs = [{**r, "score": r[key]} for r in recs]
    for r_i, r in enumerate(recs):
        for f in (0, 1):
            tune, held = qfold != f, qfold == f
            t = thresholds[(r_i, f)] if thresholds else choose(r["score"][tune], r["has"][tune], r["correct"][tune])
            ts = thresholds[("strict", r_i, f)] if thresholds else choose(r["score"][tune], r["has"][tune],
                                                                           r["correct"][tune], STRICT_TNR, True)
            m = decision(r["score"][held] >= t, r["has"][held], r["correct"][held])
            ms = decision(r["score"][held] >= ts, r["has"][held], r["correct"][held])
            f1.append(m["F1"]); tnr.append(m["TNR"]); thr.append(t); strict.append(ts)
            s_tnr.append(ms["TNR"]); s_rec.append(ms["TP"] / max(ms["TP"] + ms["FN"], 1))
            sh = np.where(np.isfinite(r["score"][held]), r["score"][held], -np.inf)
            auc.append(oe.pr_auc(sh, r["has"][held].astype(int)))
    mean = lambda x: float(np.nanmean(x))
    return {"F1": mean(f1), "TNR": mean(tnr), "PR-AUC": mean(auc), "F1_std": float(np.std(f1)),
            "TNR_std": float(np.std(tnr)), "thresholds": thr, "strict": strict,
            "strict_TNR": mean(s_tnr), "strict_recall": mean(s_rec)}


def main() -> int:
    ap = bp.data_args(argparse.ArgumentParser())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    out_json = a.out_dir / f"closed_gallery_{a.tag}.json"
    if not a.report:
        res = run(a)
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    res = json.loads(out_json.read_text(encoding="utf-8"))
    (a.out_dir / f"closed_gallery_{a.tag}.md").write_text(report_md(res), encoding="utf-8")
    (HERE.parent / "docs" / "refusal.md").write_text(docs_md(res), encoding="utf-8")
    print(f"-> {a.out_dir / f'closed_gallery_{a.tag}.md'}, docs/refusal.md")
    return 0


def run(a) -> dict:
    import pseudo_camera as pc
    from app.config import Settings

    tau = Settings().pp_camera_tau
    D = he.Data(a)
    qrows, grows = D.valid(D.qry), D.valid(D.gal)
    assert (len(qrows), len(grows)) == (546, 361), (len(qrows), len(grows))
    boxes = {r["image_id"]: r["box"] for r in pc.read_rows(a.data_dir)}
    bg = tvg.service_bg(D, a)
    Q = he.frame(qrows)
    qids = list(Q.index)
    qv, qc = Q.vehicle_id.to_numpy(), Q.camera_id.to_numpy()
    q = D.vectors(qids)
    qbg = np.stack([bg[i] for i in qids])
    qbox = np.array([boxes[i] for i in qids], dtype="float64")
    vehicles = sorted(set(qv), key=int)
    rng = np.random.default_rng(42)                                   # как head_to_head.py
    rot_of = {v: n % 4 for n, v in enumerate(rng.permutation(vehicles))}
    fold_of = {v: n % 2 for n, v in enumerate(rng.permutation(vehicles))}
    keep_cam = {v: Counter(qc[qv == v]).most_common(1)[0][0] for v in vehicles}
    qfold = np.array([fold_of[v] for v in qv])
    extra = {"361": [], "1111": D.valid(D.distractors("test_gallery")), "2.2k": D.valid(D.distractors("test_all")),
             "10.8k": D.valid(D.distractors("test_all+train_seen"))}
    res = {"tag": a.tag, "tau": tau, "rules": RULES, "n_queries": len(qids), "galleries": {}}
    recs_all = {}
    checked = False
    for gname in GALLERIES:
        G = he.frame(grows + extra[gname])
        gids = list(G.index)
        gv, gc = G.vehicle_id.to_numpy(), G.camera_id.to_numpy()
        g = D.vectors(gids)
        raw = q @ g.T
        s = {"raw": raw, "dba": q @ dba(g, k=1, weights="cos").T}
        gbox = np.array([boxes[i] for i in gids], dtype="float64")
        fl = {"none": np.zeros(raw.shape, bool), "bg": (qbg @ np.stack([bg[i] for i in gids]).T) > tau}
        for name, rule in RULES.items():
            fl[f"dup_{name}"] = dd.near_dup_flags(raw, qbox[:, None, :], gbox[None, :, :], **rule)
        for x, y in (("bg", "dup_ours"), ("bg", "dup_andrey")):
            fl[f"{x}|{y}"] = fl[x] | fl[y]
        keeps = [np.array([not (rot_of.get(v) == r and c != keep_cam.get(v)) for v, c in zip(gv, gc)])
                 for r in range(4)]
        gres = {"N": len(gids)}
        junk = (gv[None, :] == qv[:, None]) & (gc[None, :] == qc[:, None])
        pos = (gv[None, :] == qv[:, None]) & (gc[None, :] != qc[:, None])
        for vname, sk, fr, ff in VARIANTS:
            order = np.argsort(-(s[sk] - PUSH * fl[fr]), axis=1, kind="stable")[:, :10]
            rk = oe.ranking_metrics(Q, G, {qid: [gids[j] for j in order[i]] for i, qid in enumerate(qids)}, top_k=10)
            v_out = {"mAP@10": rk["mAP@10"], "R1": rk["Rank-1"], "R5": rk["Rank-5"],
                     "flags_on_true_pairs": float(fl[ff][pos].mean()), "flags_on_junk": float(fl[ff][junk].mean())}
            for z_mode in ("andrey", "service"):
                recs = [rotation_records(s[sk], fl[fr], fl[ff], kp, qv, qc, gv, gc, z_mode) for kp in keeps]
                if not checked:                                   # одна сверка с candidate_metrics эталона
                    r0 = recs[0]; held = qfold == 0
                    t = choose(r0["score"][~held], r0["has"][~held], r0["correct"][~held])
                    gsub = G.iloc[np.flatnonzero(keeps[0])]
                    cand = {qid: [(gids[r0["top1"][i]], float(r0["score"][i]))] for i, qid in enumerate(qids)
                            if held[i] and r0["score"][i] >= t}
                    o = oe.candidate_metrics(Q[held], gsub, cand)
                    m = decision(r0["score"][held] >= t, r0["has"][held], r0["correct"][held])
                    assert all(o[k_] == m[k_] for k_ in ("TP", "FP", "FN", "TN")), (o, m)
                    print("   совпадение с candidate_metrics эталона: ок", flush=True)
                    checked = True
                v_out[z_mode] = evaluate_refusal(recs, qfold)
                recs_all[(gname, vname, z_mode)] = recs
            gres[vname] = v_out
            print(f"{gname:6s} {vname:52s} mAP {v_out['mAP@10']:.4f} R1 {v_out['R1']:.3f} R5 {v_out['R5']:.3f} | "
                  f"andrey F1 {v_out['andrey']['F1']:.3f} TNR {v_out['andrey']['TNR']:.3f} | service F1 "
                  f"{v_out['service']['F1']:.3f} TNR {v_out['service']['TNR']:.3f}", flush=True)
        res["galleries"][gname] = gres
    # пресеты: порог вне фолда по 361 и 1111 вместе (service), медиана по фолдам × ротациям
    res["presets"] = {}
    for vname, _, fr, ff in VARIANTS:
        thr, strict = {}, {}
        for r_i in range(4):
            for f in (0, 1):
                tune = qfold != f
                parts = [(p["score"][tune], p["has"][tune], p["correct"][tune])
                         for p in (recs_all[(g, vname, "service")][r_i] for g in MAIN)]
                thr[(r_i, f)] = choose_joint(parts)
                strict[(r_i, f)] = choose_joint(parts, STRICT_TNR, True)
        t_med, s_med = float(np.median(list(thr.values()))), float(np.median(list(strict.values())))
        fixed = {**{k: t_med for k in thr}, **{("strict",) + k: s_med for k in strict}}
        res["presets"][vname] = {"rank": fr, "refusal": ff, "threshold": t_med, "strict": s_med,
                                 "fold_thresholds": [round(x, 4) for x in thr.values()],
                                 "per_gallery": {g: evaluate_refusal(recs_all[(g, vname, "service")], qfold, fixed)
                                                 for g in GALLERIES}}
    res["config_presets"] = {
        repr(PRESET_KEYS[v]): {"balanced": round(p["threshold"], 4), "strict": round(p["strict"], 4),
                               "note": f"{v}: 361/1111 как закрытый тест, F1 при TNR ≥ 0.70 на каждой "
                                       f"(closed_gallery_{a.tag}.md)"}
        for v, p in res["presets"].items() if v in PRESET_KEYS and v not in ("raw", "DBA")}
    # F1-оптимум без ограничения на TNR — для итога: сколько F1 отдаём за TNR ≥ 0.70
    fin_name = FINAL
    free = {}
    for r_i in range(4):
        for f in (0, 1):
            tune = qfold != f
            parts = [(p["score"][tune], p["has"][tune], p["correct"][tune])
                     for p in (recs_all[(g, fin_name, "service")][r_i] for g in MAIN)]
            free[(r_i, f)] = choose_joint(parts, floor=-1.0)
    t_free = float(np.median(list(free.values())))
    fixed = {**{k: t_free for k in free}, **{("strict",) + k: np.inf for k in free}}
    res["free"] = {"threshold": t_free, "per_gallery": {g: evaluate_refusal(recs_all[(g, fin_name, "service")], qfold,
                                                                              fixed) for g in GALLERIES}}
    # сервис сейчас без постобработки: сырой косинус top-1, порог 0.8025
    cur = {**{(r_i, f): 0.8025 for r_i in range(4) for f in (0, 1)},
           **{("strict", r_i, f): 0.8465 for r_i in range(4) for f in (0, 1)}}
    res["current"] = {g: evaluate_refusal(recs_all[(g, "raw", "service")], qfold, cur, "score_raw") for g in GALLERIES}
    res["final"] = FINAL
    res["fallback"] = FALLBACK
    w = {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~") for k, v in D.env.items()}
    res["weights"] = w.get("REID_WEIGHTS", "")
    return res


ANDREY_REF = {  # head_to_head.json ветки andrew-postprocess-throughput (b3e7ae3), mAP@10 / F1 / TNR
    ("361", "raw"): (0.8671, 0.089, 0.694), ("361", "DBA"): (0.8835, 0.095, 0.669),
    ("361", "DBA + фон"): (0.8835, 0.813, 0.745), ("361", "DBA + дубли Андрея (и в ранжировании)"): (0.8765, 0.805, 0.710),
    ("1111", "raw"): (0.7401, 0.215, 0.679), ("1111", "DBA"): (0.7492, 0.225, 0.679),
    ("1111", "DBA + фон"): (0.7509, 0.680, 0.691), ("1111", "DBA + дубли Андрея (и в ранжировании)"): (0.7418, 0.666, 0.664),
    ("2.2k", "raw"): (0.6798, 0.270, 0.702), ("2.2k", "DBA"): (0.7388, 0.237, 0.692),
    ("2.2k", "DBA + фон"): (0.7436, 0.734, 0.712)}
f3 = lambda x: f"{x:.3f}"
f4 = lambda x: f"{x:.4f}"


def report_md(res: dict) -> str:
    G, P, tag = res["galleries"], res["presets"], res["tag"]
    L = [f"# Галереи 361 и 1111 как в закрытом тесте: {tag}", "",
         f"Веса `{res['weights']}`. Протокол Андрея (`research/local_postprocess/experiments_2026-10-17/eval/"
         "head_to_head.py` ветки `andrew-postprocess-throughput`), наши эмбеддинги и код сервиса. 546 запросов / 361 "
         "кадр holdout (первый кадр каждой пары машина-камера → галерея). Ранжирование — все запросы против всей "
         "галереи, официальный `ranking_metrics`. Отказ «как сервис»: 4 ротации по 25 % машин (у них в галерее кадр "
         "одной камеры), порог вне фолда (2 фолда по машинам), F1 при TNR ≥ 0.70. Воспроизвести: `python "
         f"research/closed_gallery.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag {tag} --zip dataset.zip`.", "",
         "## 1. Сверка с числами Андрея (оценка z100 как в head_to_head.py)", "",
         "| Галерея | Вариант | mAP@10 наш / Андрея | F1 наш / Андрея | TNR наш / Андрея |", "|---|---|---:|---:|---:|"]
    for (g, v), (am, af, at) in ANDREY_REF.items():
        x = G[g][v]
        L.append(f"| {g} | {v} | {f4(x['mAP@10'])} / {f4(am)} | {f3(x['andrey']['F1'])} / {f3(af)} | "
                 f"{f3(x['andrey']['TNR'])} / {f3(at)} |")
    dm = max(abs(G[g][v]["mAP@10"] - r[0]) for (g, v), r in ANDREY_REF.items())
    dfl = [(abs(G[g][v]["andrey"]["F1"] - r[1]), g, v) for (g, v), r in ANDREY_REF.items()]
    dtn = [(abs(G[g][v]["andrey"]["TNR"] - r[2]), g, v) for (g, v), r in ANDREY_REF.items()]
    mf, mt = max(dfl), max(dtn)
    L += ["", f"Наибольшие расхождения: mAP@10 {dm:.4f}, F1 {mf[0]:.3f} ({mf[1]}, «{mf[2]}»), TNR {mt[0]:.3f} "
          f"({mt[1]}, «{mt[2]}»). Причина — "
          "эмбеддинги: у нас из готовых JPEG-кропов `crops/` кодом сервиса, у Андрея — свой кэш a10 (`a10_*.npz`, та же "
          "предобработка squash 224, отступ 0.10, без TTA); остальное — его код один в один (сиды ротаций и фолдов, "
          "камера ротации, сетка порогов, DBA по всей галерее, совпадение с `candidate_metrics` эталона проверено). "
          "F1 и TNR считаются на половине запросов, без пары — ~37 запросов на фолд (шаг TNR ~0.03): малый сдвиг "
          "оценок двигает выбранный порог.", "",
          "## 2. Варианты на 361 и 1111 (оценка z100 как в сервисе: top-100 по скору, почти-дубли выброшены)", "",
          "Порог — один на обе галереи: TNR ≥ 0.70 на фолде выбора каждой, максимум среднего F1, в пресет — медиана "
          "по фолдам × ротациям.", "",
          "| Вариант | mAP@10 361 / 1111 | R1 / R5 361 | R1 / R5 1111 | Порог z100 | F1 / TNR 361 | F1 / TNR 1111 | "
          "PR-AUC 361 / 1111 | Почти-дубли задевают настоящих пар |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for v in P:
        a, b, pg = G["361"][v], G["1111"][v], P[v]["per_gallery"]
        L.append(f"| {'**' + v + '**' if v == res['final'] else v} | {f4(a['mAP@10'])} / {f4(b['mAP@10'])} | "
                 f"{f3(a['R1'])} / {f3(a['R5'])} | {f3(b['R1'])} / {f3(b['R5'])} | {P[v]['threshold']:.4f} | "
                 f"{f3(pg['361']['F1'])} / {f3(pg['361']['TNR'])} | {f3(pg['1111']['F1'])} / {f3(pg['1111']['TNR'])} | "
                 f"{f3(pg['361']['PR-AUC'])} / {f3(pg['1111']['PR-AUC'])} | {a['flags_on_true_pairs']:.1%} |"
                 .replace("%", " %"))
    L += ["", "## 3. 2.2k и 10.8k — справочно, завышены", "",
          "2.2k содержит пачки почти-дублей тестовых запросов, 10.8k — кадры обучающих машин: обе завышают DBA "
          f"(на 2.2k +{G['2.2k']['DBA']['mAP@10'] - G['2.2k']['raw']['mAP@10']:.3f} mAP@10 против "
          f"+{G['1111']['DBA']['mAP@10'] - G['1111']['raw']['mAP@10']:.3f} на 1111). Решения по ним не принимались.", "",
          "| Вариант | mAP@10 2.2k / 10.8k | F1 / TNR 2.2k | F1 / TNR 10.8k |", "|---|---:|---:|---:|"]
    for v in P:
        pg = P[v]["per_gallery"]
        L.append(f"| {v} | {f4(G['2.2k'][v]['mAP@10'])} / {f4(G['10.8k'][v]['mAP@10'])} | "
                 f"{f3(pg['2.2k']['F1'])} / {f3(pg['2.2k']['TNR'])} | {f3(pg['10.8k']['F1'])} / {f3(pg['10.8k']['TNR'])} |")
    return "\n".join(L) + "\n"


def docs_md(res: dict) -> str:
    """docs/refusal.md — только цифры 361 / 1111."""
    G, P, fin, fb = res["galleries"], res["presets"], res["final"], res["fallback"]
    F, B, cur, fr = P[fin], P[fb], res["current"], res["free"]
    pf = lambda v, g: f"{f3(P[v]['per_gallery'][g]['F1'])} / {f3(P[v]['per_gallery'][g]['TNR'])}"
    L = ["# Режим отказа", "",
         "Итоговая схема (команда, 18.10): **ранжирование — DBA k=1 + штраф по фону; отказ — z100 по кандидатам без "
         "почти-дублей, где почти-дубль = фон ИЛИ правило рамок**. Правило рамок в ранжирование не идёт: оно задевает "
         f"{G['361'][fin]['flags_on_true_pairs']:.1%} настоящих пар. Все числа — на галереях 361 и 1111 из невиданных "
         "машин, как в закрытом тесте (`research/honest_eval_results/closed_gallery_a10.md`).".replace("%", " %"), "",
         "## Что считается отказом", "",
         "- `/v1/search` отвечает `refusal: true` (`low_similarity`), если оценка уверенности меньше порога, и "
         "`gallery_empty`, если галерея пуста. Иначе — кандидаты.",
         "- Метрика эталона (`candidate_metrics`): у запроса без пары отказ — TN, ответ — FP; у запроса с парой ответ "
         "с верной машиной на первом месте — TP, с чужой — FP, отказ — FN. PR-AUC — площадь под кривой "
         "точность–полнота оценки для «у запроса есть пара», от порога не зависит.", "",
         "## Оценка уверенности", "",
         "- `z100` = (s1 − mean(s2..s100)) / std(s2..s100) по кандидатам top-100 (по DBA-скору) без почти-дублей.",
         "- Почти-дубль для отказа: косинус фонов > 0.68127 **или** правило рамок — сырой косинус ≥ 0.84 и IoU рамок ≥ "
         "0.6, сдвиг центра ≤ 0.3 размера рамки. Пороги рамок Андрея (≥ 0.94, или ≥ 0.85 и IoU ≥ 0.7) почти равны "
         f"нашим: F1 / TNR {pf(fin.replace('наши пороги', 'пороги Андрея'), '361')} на 361 и "
         f"{pf(fin.replace('наши пороги', 'пороги Андрея'), '1111')} на 1111 против наших {pf(fin, '361')} и "
         f"{pf(fin, '1111')} — у наших выше средний F1.",
         "- Где в конфиге: `api/app/config.py` — `REFUSAL_SCORE`, `SEARCH_THRESHOLD`, `SEARCH_THRESHOLD_STRICT`, "
         "`PP_DEDUP_RANK`, `PP_DEDUP_REFUSAL`; посчитанные пороги — `REFUSAL_PRESETS`.", "",
         "## Как выбран порог", "",
         "- Протокол Андрея: 546 запросов / 361 кадр holdout; 1111 = 361 + 750 кадров test gallery как чужие машины. "
         "Отказ «как сервис»: 4 ротации по 25 % машин — у них в галерее остаётся кадр одной камеры, запросы с этой "
         "камеры без пары, но с почти-дублем.",
         "- Порог вне фолда (2 фолда по машинам): на фолде выбора — TNR ≥ 0.70 на 361 **и** на 1111, из таких — "
         "максимум среднего F1; в конфиг — медиана по фолдам × ротациям, проверка — на отложенных фолдах с ней. "
         f"Пороги по фолдам: {', '.join(f'{x:.2f}' for x in F['fold_thresholds'])} — плато шириной ~1, берём середину.",
         "- 2.2k, 9.0k, 10.8k для выбора не годятся: в 2.2k пачки почти-дублей тестовых запросов, в 9.0k/10.8k — "
         "кадры обучающих машин; обе завышают DBA (на 2.2k "
         f"+{G['2.2k']['DBA']['mAP@10'] - G['2.2k']['raw']['mAP@10']:.3f} mAP@10, на 1111 "
         f"+{G['1111']['DBA']['mAP@10'] - G['1111']['raw']['mAP@10']:.3f}).", "",
         "## Итог на 361 / 1111", "",
         "| | mAP@10 | R1 / R5 | Порог | F1 / TNR | PR-AUC | Строгий: TNR / полнота |", "|---|---:|---:|---:|---:|---:|---:|"]
    for g in ("361", "1111"):
        x, pg = G[g][fin], F["per_gallery"][g]
        L.append(f"| {g} | {f4(x['mAP@10'])} | {f3(x['R1'])} / {f3(x['R5'])} | z100 {F['threshold']:.4f} | "
                 f"**{f3(pg['F1'])} / {f3(pg['TNR'])}** | {f3(pg['PR-AUC'])} | {f3(pg['strict_TNR'])} / "
                 f"{f3(pg['strict_recall'])} |")
    L += ["", f"Строгий порог — {F['strict']:.4f} (наименьший с TNR ≥ 0.98 на обеих).", "",
          "## Сколько F1 отдаём за TNR", "",
          "| Галерея | Сейчас без постобработки: raw 0.8025 | Итог, F1-оптимум без ограничения (порог "
          f"{fr['threshold']:.2f}) | **Итог, порог {F['threshold']:.2f} (F1 при TNR ≥ 0.70)** |", "|---|---:|---:|---:|"]
    for g in ("361", "1111"):
        L.append(f"| {g} | {f3(cur[g]['F1'])} / {f3(cur[g]['TNR'])} | {f3(fr['per_gallery'][g]['F1'])} / "
                 f"{f3(fr['per_gallery'][g]['TNR'])} | **{pf(fin, g)}** |")
    L += ["", "Значения — F1 / TNR. Без постобработки запросу без пары отвечает кадр его же машины с той же камеры, "
          "поэтому TNR низкий.", "",
          "## Запасной план без фона (если FPS с фоном не пройдёт)", "",
          f"Ранжирование — DBA k=1, отказ — z100 без почти-дублей по правилу рамок (`PP_DEDUP_RANK=off`, "
          f"`PP_DEDUP_REFUSAL=bbox`), порог {B['threshold']:.4f}: F1 / TNR {pf(fb, '361')} на 361 и {pf(fb, '1111')} на "
          f"1111, mAP@10 {f4(G['361'][fb]['mAP@10'])} / {f4(G['1111'][fb]['mAP@10'])} (с фоном "
          f"{f4(G['361'][fin]['mAP@10'])} / {f4(G['1111'][fin]['mAP@10'])}). Признак фона на этих галереях даёт мало; "
          "правило рамок стоит 0.04 мс на запрос против 1.4 мс у фона.", "",
          "## Воспроизвести", "", "```bash",
          "python research/closed_gallery.py --data-dir <данные> --env REID_WEIGHTS=<веса a10> --tag a10 --zip dataset.zip",
          "```", "", "Пишет `closed_gallery_a10.json` / `.md` и этот файл; пороги для `REFUSAL_PRESETS` — в JSON, ключ "
          "`config_presets`. Около минуты на готовых эмбеддингах.", "",
          "## Оговорки", "",
          "- 150 машин holdout: на фолд ~37 запросов без пары, шаг TNR ~0.03; разброс порогов по фолдам ~1 единица z.",
          "- Эмбеддинги — из готовых JPEG-кропов; у Андрея свой кэш, числа сходятся до 0.003 mAP@10.",
          "- Предположение протокола: в закрытой галерее у запроса без пары есть кадр его машины с той же камеры.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
