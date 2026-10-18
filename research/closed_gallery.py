#!/usr/bin/env python3
"""Порог отказа на галереях как в закрытом тесте — протокол Андрея (head_to_head.py ветки
andrew-postprocess-throughput), наши эмбеддинги и код сервиса.

- Holdout как у организаторов: первый кадр каждой пары (машина, камера) → галерея,
  остальные → запросы: 546 запросов / 361 кадр (5 побайтных дублей train↔holdout убраны).
- Ранжирование: все запросы против всей галереи, top-10, junk вычищает официальный
  evaluate (research/official_evaluate.ranking_metrics).
- Отказ «как сервис»: ротации; в каждой у части машин в галерее остаётся кадр одной камеры
  (самой частой среди их запросов) — их запросы с этой камеры без пары, но с почти-дублем.
  Порог вне фолда: 2 фолда по машинам, выбор на одном (сетка — 201 квантиль оценки), проверка
  на другом; в пресет — медиана по фолдам × ротациям. Сиды — как у Андрея (default_rng(42)).
- Оценка отказа z100 по кандидатам без почти-дублей; top-100 двумя способами:
    andrey  — top-100 после выталкивания помеченных вниз (как head_to_head.py);
    service — top-100 по скору, помеченные выброшены (как api/app/pp/state.py) — для пресетов.
- Протоколы (--protocol):
    q15 (по умолчанию, команда 18.10) — правило: максимум 0.7·F1 + 0.3·TNR (ответ организаторов
      Q15); 8 ротаций по 3/8 машин — без пары 20 % запросов (Q17); галереи 361 и 361 + 750 кадров
      обучающих машин (первый кадр пары машина-камера, train_distractors). Кадры и метки теста
      не используются; пишет и docs/refusal.md.
    andrey (прежний) — F1 при TNR ≥ 0.70 на каждой; 4 ротации по 25 % машин (13 % запросов без
      пары); галереи 361, 1111 (+750 test gallery), 2.2k, 10.8k. Даёт пороги 4.6506 / 9.6847.
- Финальная схема (команда, 18.10): ранжирование = DBA k=1 без флагов, отказ = z100 без
  почти-дублей «фон ИЛИ рамка». Прежний итог (ранжирование DBA + фон) — строкой для сравнения:
  фон в ранжировании уводит из top-1 кадр той же машины с той же камеры, а эталон засчитывает
  его верным (official_evaluate.py:290).

    python research/closed_gallery.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10_q15 --zip dataset.zip
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
W_F1, W_TNR = 0.7, 0.3                      # оценка отказа организаторов (Q15, пересказ команды)
RULES = {"ours": {"s_any": float("inf"), "s_hi": 0.84, "t_iou": 0.6, "t_center": 0.3, "t_size": float("inf")},
         "andrey": {"s_any": 0.94, "s_hi": 0.85, "t_iou": 0.7, "t_center": float("inf"), "t_size": float("inf")}}
# q15 (по умолчанию, команда 18.10): правило выбора — максимум 0.7·F1 + 0.3·TNR (Q15); без пары ~20 % запросов
#   (Q17): 8 ротаций, в каждой у 3/8 машин в галерее кадр одной камеры (каждая машина — в 3 ротациях из 8);
#   большая галерея — 361 + 750 кадров обучающих машин (по кадру на пару машина-камера), кадров теста нет.
# andrey — прежний протокол (head_to_head.py): 4 ротации по 25 % машин (~13 % запросов без пары), F1 при
#   TNR ≥ 0.70, большая галерея — 361 + 750 кадров test gallery.
# q17 — буквально по ответу Q17: у 20 % машин (5 ротаций) в галерее не остаётся ни одного кадра.
PROTOCOLS = {"q15": {"rotations": 8, "open_width": 3, "objective": "q15", "galleries": ("361", "1111tr"), "open": "keep_cam"},
             "q17": {"rotations": 5, "open_width": 1, "objective": "q15", "galleries": ("361", "1111tr"), "open": "drop"},
             "andrey": {"rotations": 4, "open_width": 1, "objective": "f1", "galleries": ("361", "1111", "2.2k", "10.8k"),
                        "open": "keep_cam"}}
N_TRAIN_DISTRACTORS, DISTRACTOR_SEED = 750, 2026
OLD = {"threshold": 4.6506, "strict": 9.6847, "fallback": 4.9039, "fallback_strict": 9.9144}   # пресеты до 18.10
# задаются в main() по --protocol
GALLERIES, MAIN, ROTATIONS, OPEN_W, OBJECTIVE, OPEN_MODE = PROTOCOLS["q15"]["galleries"], ("361", "1111tr"), 8, 3, "q15", "keep_cam"
GALLERY_TITLE = {"361": "361", "1111": "1111 (+750 test gallery)", "1111tr": "1111 (+750 кадров обучающих машин)",
                 "2.2k": "2.2k", "10.8k": "10.8k"}
FINAL = "итог: DBA; отказ фон ИЛИ рамка (наши пороги)"
PREV_FINAL = "прежний итог: DBA + фон; отказ фон ИЛИ рамка (наши пороги)"
FALLBACK = "запасной: DBA; отказ рамка (наши пороги)"
# ключ REFUSAL_PRESETS: (PP_DBA_K, PP_DEDUP_RANK, PP_DEDUP_REFUSAL, формула) — пороги рамок сервиса = наши
PRESET_KEYS = {"raw": (0, "off", "off", "z100"), "DBA": (1, "off", "off", "z100"), "DBA + фон": (1, "bg", "bg", "z100"),
               FINAL: (1, "off", "both", "z100"), PREV_FINAL: (1, "bg", "both", "z100"),
               FALLBACK: (1, "off", "bbox", "z100")}
# (имя, скор, флаги ранжирования, флаги отказа)
VARIANTS = (("raw", "raw", "none", "none"), ("DBA", "dba", "none", "none"),
            ("DBA + фон", "dba", "bg", "bg"),
            ("DBA + дубли Андрея (и в ранжировании)", "dba", "dup_andrey", "dup_andrey"),
            (FINAL, "dba", "none", "bg|dup_ours"),
            ("итог: DBA; отказ фон ИЛИ рамка (пороги Андрея)", "dba", "none", "bg|dup_andrey"),
            (PREV_FINAL, "dba", "bg", "bg|dup_ours"),
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


def objective(m: dict) -> float:
    """Оценка режима отказа организаторов (Q15): 0.7·F1 + 0.3·TNR (TNR не определён — 0)."""
    tnr = m["TNR"] if np.isfinite(m["TNR"]) else 0.0
    return W_F1 * m["F1"] + W_TNR * tnr


def choose(score, has, correct, floor=MIN_TNR, strict=False, rule=None) -> float:
    """Порог на сетке 201 квантиль. rule f1 — F1-максимум при TNR ≥ floor (как у Андрея); q15 — максимум
    0.7·F1 + 0.3·TNR без ограничения. strict — наименьший порог с TNR ≥ floor."""
    return choose_joint([(score, has, correct)], floor, strict, rule)


def choose_joint(parts: list, floor=MIN_TNR, strict=False, rule=None) -> float:
    """Порог по нескольким галереям сразу. rule f1 — TNR ≥ floor на КАЖДОЙ, из таких — максимум среднего F1;
    q15 — максимум среднего 0.7·F1 + 0.3·TNR; strict — наименьший порог, где TNR ≥ floor на каждой.
    parts — [(score, has, correct)]."""
    rule = rule or OBJECTIVE
    fin = np.concatenate([p[0][np.isfinite(p[0])] for p in parts])
    best_t, best = np.inf, -1.0
    for t in (np.unique(np.quantile(fin, np.linspace(0, 1, 201))) if len(fin) else []):
        ms = [decision(sc >= t, hs, co) for sc, hs, co in parts]
        if strict or rule == "f1":
            if not all(m["TNR"] >= floor for m in ms):
                continue
            if strict:
                return float(t)
        val = float(np.mean([m["F1"] if rule == "f1" else objective(m) for m in ms]))
        if val > best:
            best_t, best = float(t), val
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
    f1, tnr, rec, obj, auc, thr, strict, s_tnr, s_rec, s_f1, s_obj = ([] for _ in range(11))
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
            rec.append(m["TP"] / max(m["TP"] + m["FN"], 1)); obj.append(objective(m))
            s_tnr.append(ms["TNR"]); s_rec.append(ms["TP"] / max(ms["TP"] + ms["FN"], 1))
            s_f1.append(ms["F1"]); s_obj.append(objective(ms))
            sh = np.where(np.isfinite(r["score"][held]), r["score"][held], -np.inf)
            auc.append(oe.pr_auc(sh, r["has"][held].astype(int)))
    mean = lambda x: float(np.nanmean(x))
    return {"F1": mean(f1), "TNR": mean(tnr), "recall": mean(rec), "Q15": mean(obj), "PR-AUC": mean(auc),
            "F1_std": float(np.std(f1)), "TNR_std": float(np.std(tnr)), "thresholds": thr, "strict": strict,
            "strict_TNR": mean(s_tnr), "strict_recall": mean(s_rec), "strict_F1": mean(s_f1),
            "strict_Q15": mean(s_obj), "open_share": mean([1 - r["has"].mean() for r in recs])}


def main() -> int:
    ap = bp.data_args(argparse.ArgumentParser())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--protocol", choices=tuple(PROTOCOLS), default="q17",
                    help="q15 — основной (0.7·F1 + 0.3·TNR, ~20 %% запросов без пары, без кадров теста); q17 — 20 %% машин "
                         "целиком без кадров в галерее; andrey — прежний")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    set_protocol(a.protocol)
    out_json = a.out_dir / f"closed_gallery_{a.tag}.json"
    if not a.report:
        res = run(a)
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    res = json.loads(out_json.read_text(encoding="utf-8"))
    (a.out_dir / f"closed_gallery_{a.tag}.md").write_text(report_md(res), encoding="utf-8")
    if a.protocol == "q17":                          # docs/refusal.md — только по основному протоколу
        alt = a.out_dir / "closed_gallery_a10_q15.json"  # с почти-дублем своей камеры (Q22) — для сравнения
        alt = json.loads(alt.read_text(encoding="utf-8")) if alt.exists() else None
        (HERE.parent / "docs" / "refusal.md").write_text(docs_md(res, alt), encoding="utf-8")
    print(f"-> {a.out_dir / f'closed_gallery_{a.tag}.md'}" + (", docs/refusal.md" if a.protocol == "q17" else ""))
    return 0


def set_protocol(name: str) -> None:
    global GALLERIES, MAIN, ROTATIONS, OPEN_W, OBJECTIVE, OPEN_MODE
    p = PROTOCOLS[name]
    GALLERIES, ROTATIONS, OPEN_W, OBJECTIVE = p["galleries"], p["rotations"], p["open_width"], p["objective"]
    OPEN_MODE = p["open"]
    MAIN = GALLERIES[:2]


def train_distractors(D, n: int = N_TRAIN_DISTRACTORS, seed: int = DISTRACTOR_SEED) -> list:
    """Отвлекатели не из теста: кадры обучающих машин (не holdout — значит, ни одной машины из запросов),
    первый кадр каждой пары машина-камера в порядке train.csv, n пар случайно (фиксированный сид).
    Кадры из групп побайтно одинаковых полных кадров (honest_eval_results/leak_audit.json) не берутся:
    среди них пары holdout↔обучение и тест↔обучение."""
    dup = json.loads((HERE / "honest_eval_results" / "leak_audit.json").read_text(encoding="utf-8"))
    banned = {x for p in dup["exact_full_frame_duplicates"]["pairs"] for x in (p["a"], p["b"])}
    first = {}
    for r in D.distractors("train_seen"):
        if r["image_id"] not in banned:
            first.setdefault((r["vehicle_id"], r["camera_id"]), r)
    pairs = list(first.values())
    pick = np.sort(np.random.default_rng(seed).choice(len(pairs), size=n, replace=False))
    return [pairs[i] for i in pick]


def open_in(perm: dict, v, r: int) -> bool:
    """Машина v в ротации r — «без пары» (в галерее кадр одной камеры): окно OPEN_W из ROTATIONS групп.
    perm — место машины в перестановке; andrey (4, 1) — ровно rot_of = n % 4 у head_to_head.py."""
    return v in perm and (perm[v] % ROTATIONS - r) % ROTATIONS < OPEN_W


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
    perm = {v: n for n, v in enumerate(rng.permutation(vehicles))}  # ротация машины — open_in(perm, v, r)
    fold_of = {v: n % 2 for n, v in enumerate(rng.permutation(vehicles))}
    keep_cam = {v: Counter(qc[qv == v]).most_common(1)[0][0] for v in vehicles}
    qfold = np.array([fold_of[v] for v in qv])
    extra = {"361": lambda: [], "1111tr": lambda: D.valid(train_distractors(D)),
             "1111": lambda: D.valid(D.distractors("test_gallery")), "2.2k": lambda: D.valid(D.distractors("test_all")),
             "10.8k": lambda: D.valid(D.distractors("test_all+train_seen"))}     # тест читается только в andrey
    res = {"tag": a.tag, "tau": tau, "rules": RULES, "n_queries": len(qids), "galleries": {},
           "protocol": {"rotations": ROTATIONS, "open_width": OPEN_W, "objective": OBJECTIVE, "main": list(MAIN),
                        "open": OPEN_MODE,
                        "galleries": list(GALLERIES), "distractor_seed": DISTRACTOR_SEED}}
    recs_all = {}
    checked = False
    for gname in GALLERIES:
        G = he.frame(grows + extra[gname]())
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
        keeps = [np.array([not (open_in(perm, v, r) and (OPEN_MODE == "drop" or c != keep_cam.get(v)))
                           for v, c in zip(gv, gc)]) for r in range(ROTATIONS)]
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
            # доля запросов с парой, у которых ответ — кадр той же машины с той же камеры: эталон
            # (official_evaluate.candidate_metrics) сверяет только vehicle_id и засчитывает его верным
            v_out["same_cam_top1"] = float(np.mean([junk[np.arange(len(qids)), r["top1"]][r["has"]].mean()
                                                    for r in recs_all[(gname, vname, "service")]]))
            # запросы без пары: лучший кандидат оценки отказа (top-100 по скору, почти-дубли выброшены, как в
            # сервисе) — доля, когда это отвлекатель (не кадр holdout), и его средний скор
            dist, s1 = [], []
            for kp, r in zip(keeps, recs_all[(gname, vname, "service")]):
                k = np.flatnonzero(kp)
                for i in np.flatnonzero(~r["has"]):
                    o = np.argsort(-s[sk][i, k], kind="stable")[:Z_TOP]
                    c = o[~fl[ff][i, k[o]]]
                    if len(c):
                        dist.append(k[c[0]] >= len(grows)); s1.append(s[sk][i, k[c[0]]])
            v_out["open_best_distractor"] = float(np.mean(dist)) if dist else float("nan")
            v_out["open_best_s1"] = float(np.mean(s1)) if s1 else float("nan")
            gres[vname] = v_out
            print(f"{gname:6s} {vname:52s} mAP {v_out['mAP@10']:.4f} R1 {v_out['R1']:.3f} R5 {v_out['R5']:.3f} | "
                  f"andrey F1 {v_out['andrey']['F1']:.3f} TNR {v_out['andrey']['TNR']:.3f} | service F1 "
                  f"{v_out['service']['F1']:.3f} TNR {v_out['service']['TNR']:.3f}", flush=True)
        res["galleries"][gname] = gres
    # пресеты: порог вне фолда по 361 и 1111 вместе (service), медиана по фолдам × ротациям
    res["presets"] = {}
    for vname, _, fr, ff in VARIANTS:
        thr, strict = {}, {}
        for r_i in range(ROTATIONS):
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
    how = ("361 и 1111 (+750 кадров обучающих машин), без кадров теста; максимум 0.7·F1 + 0.3·TNR (Q15), "
           "~20 % запросов без пары" if OBJECTIVE == "q15" else "361/1111 как закрытый тест, F1 при TNR ≥ 0.70 на каждой")
    res["config_presets"] = {
        repr(PRESET_KEYS[v]): {"balanced": round(p["threshold"], 4), "strict": round(p["strict"], 4),
                               "note": f"{v}: {how} (closed_gallery_{a.tag}.md)"}
        for v, p in res["presets"].items() if v in PRESET_KEYS and v not in ("raw", "DBA")}
    # для сравнения на этом же протоколе: прежние пресеты (до 18.10) и прежнее правило (F1 при TNR ≥ 0.70)
    fixed_at = lambda t, ts: {**{(r_i, f): t for r_i in range(ROTATIONS) for f in (0, 1)},
                              **{("strict", r_i, f): ts for r_i in range(ROTATIONS) for f in (0, 1)}}
    res["old"], res["rule_f1"] = {}, {}
    for vname, t_old, s_old in ((FINAL, OLD["threshold"], OLD["strict"]), (FALLBACK, OLD["fallback"], OLD["fallback_strict"])):
        res["old"][vname] = {"threshold": t_old, "strict": s_old, "per_gallery": {
            g: evaluate_refusal(recs_all[(g, vname, "service")], qfold, fixed_at(t_old, s_old)) for g in GALLERIES}}
        thr = [choose_joint([(p["score"][qfold != f], p["has"][qfold != f], p["correct"][qfold != f])
                             for p in (recs_all[(g, vname, "service")][r_i] for g in MAIN)], rule="f1")
               for r_i in range(ROTATIONS) for f in (0, 1)]
        t_f1 = float(np.median(thr))
        res["rule_f1"][vname] = {"threshold": t_f1, "per_gallery": {
            g: evaluate_refusal(recs_all[(g, vname, "service")], qfold, fixed_at(t_f1, res["presets"][vname]["strict"]))
            for g in GALLERIES}}
    # F1-оптимум без ограничения на TNR — для итога: сколько F1 отдаём за TNR ≥ 0.70
    fin_name = FINAL
    free = {}
    for r_i in range(ROTATIONS):
        for f in (0, 1):
            tune = qfold != f
            parts = [(p["score"][tune], p["has"][tune], p["correct"][tune])
                     for p in (recs_all[(g, fin_name, "service")][r_i] for g in MAIN)]
            free[(r_i, f)] = choose_joint(parts, floor=-1.0, rule="f1")
    t_free = float(np.median(list(free.values())))
    fixed = {**{k: t_free for k in free}, **{("strict",) + k: np.inf for k in free}}
    res["free"] = {"threshold": t_free, "per_gallery": {g: evaluate_refusal(recs_all[(g, fin_name, "service")], qfold,
                                                                              fixed) for g in GALLERIES}}
    # сервис сейчас без постобработки: сырой косинус top-1, порог 0.8025
    cur = {**{(r_i, f): 0.8025 for r_i in range(ROTATIONS) for f in (0, 1)},
           **{("strict", r_i, f): 0.8465 for r_i in range(ROTATIONS) for f in (0, 1)}}
    res["current"] = {g: evaluate_refusal(recs_all[(g, "raw", "service")], qfold, cur, "score_raw") for g in GALLERIES}
    res["score_choice"] = score_choice(recs_all, qfold)
    res["final"] = FINAL
    res["prev_final"] = PREV_FINAL
    res["fallback"] = FALLBACK
    w = {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~") for k, v in D.env.items()}
    res["weights"] = w.get("REID_WEIGHTS", "")
    return res


def score_choice(recs_all: dict, qfold: np.ndarray) -> dict:
    """z100 против raw для итога: порог по правилу пресета (протокол: q15 — максимум среднего 0.7·F1 + 0.3·TNR,
    andrey — TNR ≥ 0.70 на каждой и максимум среднего F1; медиана по фолдам × ротациям), метрики на обеих
    галереях MAIN при нём. Источник из одной галереи — порог выбран по ней и перенесён на другую."""
    out = {}
    for key, formula in (("score", "z100"), ("score_raw", "raw")):
        recs = {g: recs_all[(g, FINAL, "service")] for g in MAIN}
        out[formula] = {}
        for src in (MAIN, MAIN[:1], MAIN[1:]):
            thr = [choose_joint([(recs[g][r_i][key][qfold != f], recs[g][r_i]["has"][qfold != f],
                                  recs[g][r_i]["correct"][qfold != f]) for g in src])
                   for r_i in range(ROTATIONS) for f in (0, 1)]
            t = float(np.median(thr))
            fixed = {**{(r_i, f): t for r_i in range(ROTATIONS) for f in (0, 1)},
                     **{("strict", r_i, f): np.inf for r_i in range(ROTATIONS) for f in (0, 1)}}
            out[formula][" + ".join(src)] = {"threshold": t, **{
                g: {k: evaluate_refusal(recs[g], qfold, fixed, key)[k] for k in ("F1", "TNR", "recall", "Q15")}
                for g in MAIN}}
    return out


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
    G, P, tag, pr = res["galleries"], res["presets"], res["tag"], res["protocol"]
    g1, g2 = pr["main"]
    rule = ("максимум 0.7·F1 + 0.3·TNR (Q15)" if pr["objective"] == "q15" else "F1 при TNR ≥ 0.70")
    open_pct = P[res["final"]]["per_gallery"][g1]["open_share"]
    L = [f"# Галереи {GALLERY_TITLE[g1]} и {GALLERY_TITLE[g2]}: {tag}", "",
         f"Веса `{res['weights']}`. Протокол Андрея (`head_to_head.py` ветки `andrew-postprocess-throughput`), наши "
         "эмбеддинги и код сервиса. 546 запросов / 361 кадр holdout (первый кадр каждой пары машина-камера → галерея). "
         f"Ранжирование — все запросы против всей галереи, официальный `ranking_metrics`. Отказ «как сервис»: "
         f"{pr['rotations']} ротаций, в каждой у {pr['open_width']}/{pr['rotations']} машин в галерее кадр одной "
         f"камеры — без пары {open_pct:.1%} запросов; порог вне фолда (2 фолда по машинам), правило — {rule}, в "
         "пресет — медиана по фолдам × ротациям. Воспроизвести: `python research/closed_gallery.py --data-dir <данные> "
         f"--env REID_WEIGHTS=<веса> --tag {tag} --zip dataset.zip`.".replace("%", " %"), ""]
    if pr["objective"] != "q15":
        L += ["## Сверка с числами Андрея (оценка z100 как в head_to_head.py)", "",
              "| Галерея | Вариант | mAP@10 наш / Андрея | F1 наш / Андрея | TNR наш / Андрея |", "|---|---|---:|---:|---:|"]
        for (g, v), (am, af, at) in ANDREY_REF.items():
            x = G[g][v]
            L.append(f"| {g} | {v} | {f4(x['mAP@10'])} / {f4(am)} | {f3(x['andrey']['F1'])} / {f3(af)} | "
                     f"{f3(x['andrey']['TNR'])} / {f3(at)} |")
        L.append("")
    L += [f"## Варианты на {g1} и {g2} (оценка z100 как в сервисе: top-100 по скору, почти-дубли выброшены)", "",
          f"Порог — один на обе галереи, правило — {rule}.", "",
          f"| Вариант | mAP@10 {g1} / {g2} | R1 / R5 {g1} | R1 / R5 {g2} | Порог z100 | F1 / TNR / полнота {g1} | "
          f"F1 / TNR / полнота {g2} | 0.7·F1+0.3·TNR {g1} / {g2} | Строгий | Своя камера в top-1 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in P:
        a, b, pg = G[g1][v], G[g2][v], P[v]["per_gallery"]
        L.append(f"| {'**' + v + '**' if v == res['final'] else v} | {f4(a['mAP@10'])} / {f4(b['mAP@10'])} | "
                 f"{f3(a['R1'])} / {f3(a['R5'])} | {f3(b['R1'])} / {f3(b['R5'])} | {P[v]['threshold']:.4f} | "
                 f"{f3(pg[g1]['F1'])} / {f3(pg[g1]['TNR'])} / {f3(pg[g1]['recall'])} | "
                 f"{f3(pg[g2]['F1'])} / {f3(pg[g2]['TNR'])} / {f3(pg[g2]['recall'])} | "
                 f"{f3(pg[g1]['Q15'])} / {f3(pg[g2]['Q15'])} | {P[v]['strict']:.4f} | "
                 f"{a['same_cam_top1']:.0%} / {b['same_cam_top1']:.0%} |".replace("%", " %"))
    if "old" in res:
        L += ["", "## Прежние пресеты и прежнее правило на этом протоколе", "",
              f"| Вариант | Порог | F1 / TNR / полнота / 0.7·F1+0.3·TNR {g1} | то же {g2} |", "|---|---|---:|---:|"]
        for v in (res["final"], res["fallback"]):
            rows = (("прежний пресет", res["old"][v]), ("правило «F1 при TNR ≥ 0.70»", res["rule_f1"][v]),
                    ("новый (Q15)", P[v]))
            for lab, x in rows:
                q = lambda g: " / ".join(f3(x["per_gallery"][g][k]) for k in ("F1", "TNR", "recall", "Q15"))
                L.append(f"| {v}: {lab} | {x['threshold']:.4f} | {q(g1)} | {q(g2)} |")
    if "2.2k" in G:
        L += ["", "## 2.2k и 10.8k — справочно, завышены", "",
              "| Вариант | mAP@10 2.2k / 10.8k | F1 / TNR 2.2k | F1 / TNR 10.8k |", "|---|---:|---:|---:|"]
        for v in P:
            pg = P[v]["per_gallery"]
            L.append(f"| {v} | {f4(G['2.2k'][v]['mAP@10'])} / {f4(G['10.8k'][v]['mAP@10'])} | "
                     f"{f3(pg['2.2k']['F1'])} / {f3(pg['2.2k']['TNR'])} | {f3(pg['10.8k']['F1'])} / {f3(pg['10.8k']['TNR'])} |")
    return "\n".join(L) + "\n"


def docs_md(res: dict, alt: dict = None) -> str:
    """docs/refusal.md — протокол q17 (alt — q15 для сравнения): галереи 361 и 361 + 750 кадров обучающих машин,
    без кадров теста."""
    G, P, fin, prev, fb = res["galleries"], res["presets"], res["final"], res["prev_final"], res["fallback"]
    F, B, sc, pr = P[fin], P[fb], res["score_choice"], res["protocol"]
    g1, g2 = pr["main"]
    t2 = GALLERY_TITLE[g2]
    pg = lambda x, g: x["per_gallery"][g]
    m4 = lambda x, g: " / ".join(f3(pg(x, g)[k]) for k in ("F1", "TNR", "recall", "Q15"))
    mp = lambda v: f"{f4(G[g1][v]['mAP@10'])} / {f4(G[g2][v]['mAP@10'])}"
    pct = lambda x: f"{x:.0%}".replace("%", " %")
    open_pct = f"{pg(F, g1)['open_share']:.1%}".replace("%", " %")
    L = ["# Режим отказа", "",
         "Итоговая схема (команда, 18.10): **ранжирование — DBA k=1 без флагов; отказ — z100 по кандидатам без "
         "почти-дублей, где почти-дубль = фон ИЛИ правило рамок** (`PP_DEDUP_RANK=off`, `PP_DEDUP_REFUSAL=both`), "
         f"**порог {F['threshold']:.4f}** (строгий {F['strict']:.4f}). Правило рамок в ранжирование не идёт: оно задевает "
         f"{G[g1][fin]['flags_on_true_pairs']:.1%} настоящих пар; фон не идёт — см. «Кадр той же камеры в top-1».".replace(
             "%", " %"), "",
         "## Правило выбора порога — оценка организаторов", "",
         "Источник — таблица ответов организаторов «Вопросы и ответы Бизнес 7. Фалькон Тех», лист «Объединённые "
         "вопросы участников»; номера — оттуда.", "",
         "- Балл за режим кандидатов — **10 % × (0.7·F1 + 0.3·TNR)** (Q14, Q15). F1 и TNR — micro по запросам, по "
         "суммарной матрице ошибок (Q25); решение по запросу — верхний кандидат по confidence (Q19, Q23); "
         "TNR = TN / (TN + FP) только по запросам без пары (Q19). Отказ — нет строк query в `candidates.csv` (Q18, "
         "Q20); confidence — любая шкала, монотонная по уверенности (Q20, Q26).",
         "- В закрытом тесте у **20 % query соответствующего ТС в gallery нет**; калибровать порог — на своём "
         "val-сплите из `train.csv` с ~20 % отложенных ID без пары (Q17, Q6).",
         "- Порог — максимум среднего 0.7·F1 + 0.3·TNR по двум галереям, вне фолда; в конфиг — медиана по фолдам × "
         f"ротациям. Без пары в протоколе {open_pct} запросов.",
         f"- **Данные теста при подборе порога не использовались**: ни кадры, ни метки. Галереи — {g1} кадр holdout и "
         f"{t2}; порог фона τ = {res['tau']} и правило рамок подобраны на парах обучающих машин "
         "(`research/honest_eval_results/cheap_bg_a10.md`, `dedup_compare_a10.md`).", "",
         "## Что считается отказом", "",
         "- `/v1/search` отвечает `refusal: true` (`low_similarity`), если оценка уверенности меньше порога, и "
         "`gallery_empty`, если галерея пуста. Иначе — кандидаты.",
         "- Метрика эталона (`candidate_metrics`): у запроса без пары отказ — TN, ответ — FP; у запроса с парой ответ "
         "с верной машиной на первом месте — TP, с чужой — FP, отказ — FN. Полнота — TP / (TP + FN). PR-AUC — площадь "
         "под кривой точность–полнота оценки для «у запроса есть пара», от порога не зависит.", "",
         "## Кадр той же камеры в top-1 считается верным", "",
         "В эталоне (`research/official_evaluate.py`, копия скрипта организаторов) кадр той же машины с той же камеры "
         "(junk) вычищается только из ранжирования: п. 1 протокола, `strip_junk` до усечения до 10 — поэтому он не "
         "влияет на mAP@10. В `candidate_metrics` его нет: «есть ли пара» считает `valid_positives` (та же машина "
         "с другой камеры), а верность ответа — строка 290:", "",
         "```python", "correct = gal_vid.get(top_gid, None) == row.vehicle_id", "```", "",
         "— только `vehicle_id`, камера не сверяется. У запроса с парой ответ кадром своей машины со своей камеры — TP. "
         "Штраф по фону в ранжировании (прежний итог) уводил такие кадры из top-1: их доля у запросов с парой "
         f"{pct(G[g1][prev]['same_cam_top1'])} / {pct(G[g2][prev]['same_cam_top1'])} против "
         f"{pct(G[g1][fin]['same_cam_top1'])} / {pct(G[g2][fin]['same_cam_top1'])} без штрафа ({g1} / {g2}). "
         "Оценка отказа у обоих одна и та же, поэтому TNR совпадает, а F1 — нет:", "",
         f"| Ранжирование | mAP@10 {g1} / {g2} | Порог | F1 / TNR / полнота / 0.7·F1+0.3·TNR {g1} | то же {g2} |",
         "|---|---:|---:|---:|---:|",
         f"| DBA k=1 (итог) | {mp(fin)} | {F['threshold']:.4f} | **{m4(F, g1)}** | **{m4(F, g2)}** |",
         f"| DBA k=1 + фон (прежний итог) | {mp(prev)} | {P[prev]['threshold']:.4f} | {m4(P[prev], g1)} | {m4(P[prev], g2)} |",
         "", "## Оценка уверенности", "",
         "- `z100` = (s1 − mean(s2..s100)) / std(s2..s100) по кандидатам top-100 (по DBA-скору) без почти-дублей.",
         f"- Почти-дубль для отказа: косинус фонов > {res['tau']} **или** правило рамок — сырой косинус ≥ 0.84 и IoU "
         "рамок ≥ 0.6, сдвиг центра ≤ 0.3 размера рамки. С порогами рамок Андрея (≥ 0.94, или ≥ 0.85 и IoU ≥ 0.7) — "
         f"{m4(P[fin.replace('наши пороги', 'пороги Андрея')], g1)} на {g1} и "
         f"{m4(P[fin.replace('наши пороги', 'пороги Андрея')], g2)} на {g2} (F1 / TNR / полнота / оценка).",
         "- Где в конфиге: `api/app/config.py` — `REFUSAL_SCORE`, `SEARCH_THRESHOLD`, `SEARCH_THRESHOLD_STRICT`, "
         "`PP_DEDUP_RANK`, `PP_DEDUP_REFUSAL`; посчитанные пороги — `REFUSAL_PRESETS`.", "",
         "## Почему z100, а не raw", "",
         "raw — косинус лучшего кандидата без почти-дублей. Его порог зависит от размера галереи: чем больше чужих "
         "кадров, тем выше лучший косинус у запроса без пары. z100 меряет, насколько лучший выделяется среди ста "
         f"ближайших, и переносится между размерами. Порог по тому же правилу; «{g1} → {g2}» — выбран только по {g1} "
         "и применён к обеим:", "",
         f"| Оценка | Порог выбран по | Порог | 0.7·F1+0.3·TNR {g1} | 0.7·F1+0.3·TNR {g2} | TNR {g1} / {g2} |",
         "|---|---|---:|---:|---:|---:|"]
    for formula in ("z100", "raw"):
        for src, lab in ((f"{g1} + {g2}", f"{g1} и {g2}"), (g1, f"{g1} → {g2}"), (g2, f"{g2} → {g1}")):
            x = sc[formula][src]
            L.append(f"| {formula} | {lab} | {x['threshold']:.4f} | {f3(x[g1]['Q15'])} | {f3(x[g2]['Q15'])} | "
                     f"{f3(x[g1]['TNR'])} / {f3(x[g2]['TNR'])} |")
    z, r = sc["z100"], sc["raw"]
    both, a_b = f"{g1} + {g2}", (lambda x: (x[g1]["Q15"] + x[g2]["Q15"]) / 2)
    raw_wins = a_b(r[both]) > a_b(z[both]) and r[g1][g2]["Q15"] > z[g1][g2]["Q15"]
    L += ["", f"Порог raw уезжает с размером галереи: {r[g1]['threshold']:.3f} по {g1} и {r[g2]['threshold']:.3f} "
          f"по {g2} (z100 — {z[g1]['threshold']:.3f} и {z[g2]['threshold']:.3f}); при переносе с {g1} TNR на {g2} у raw "
          f"{f3(r[g1][g2]['TNR'])}, у z100 {f3(z[g1][g2]['TNR'])}."]
    if raw_wins:
        L += ["", f"**Но по оценке организаторов raw на этом протоколе выше z100** — и по обеим галереям сразу "
              f"({f3(r[both][g1]['Q15'])} / {f3(r[both][g2]['Q15'])} против {f3(z[both][g1]['Q15'])} / "
              f"{f3(z[both][g2]['Q15'])}), и при переносе порога с {g1} ({f3(r[g1][g2]['Q15'])} против "
              f"{f3(z[g1][g2]['Q15'])} на {g2}). z100 выбран на прежнем протоколе, где большая галерея была из кадров "
              "test gallery и порог raw при переносе терял TNR сильнее (`closed_gallery_a10.md`). Оставлен z100 "
              "(решение команды); переход на raw — открытый вопрос."]
    else:
        L += ["", "Размер закрытой галереи неизвестен, поэтому — z100."]
    L += ["",
          "## Как выбран порог", "",
          f"- Протокол Андрея (`head_to_head.py`): 546 запросов / {g1} кадр holdout (первый кадр каждой пары машина-камера "
          f"→ галерея); большая галерея — {g1} + 750 кадров обучающих машин: первый кадр пары машина-камера, 750 пар "
          f"случайно (сид {pr['distractor_seed']}), без кадров из групп побайтных дублей полных кадров "
          "(`leak_audit.json`). Машин из запросов среди них нет.",
          (f"- Без пары — как в Q17: {pr['rotations']} ротаций, в каждой у {pr['open_width']}/{pr['rotations']} "
           "машин в галерее не остаётся ни одного кадра — все их запросы без пары; каждая машина без пары в "
           f"{pr['open_width']} ротации из {pr['rotations']}. Без пары {open_pct} запросов."
           if pr.get("open") == "drop" else
           f"- Отказ «как сервис»: {pr['rotations']} ротаций; в каждой у {pr['open_width']}/{pr['rotations']} машин в "
           "галерее остаётся кадр одной камеры (самой частой среди их запросов) — их запросы с этой камеры без пары, но "
           f"с почти-дублем; каждая машина без пары в {pr['open_width']} ротациях из {pr['rotations']}. Без пары "
           f"{open_pct} запросов."),
          f"- Порог вне фолда (2 фолда по машинам): на фолде выбора — максимум среднего 0.7·F1 + 0.3·TNR на {g1} и "
          f"{g2}; в конфиг — медиана по {2 * pr['rotations']} порогам (фолды × ротации), проверка — на отложенных "
          f"фолдах с ней. Пороги по фолдам: {', '.join(f'{x:.2f}' for x in F['fold_thresholds'])}.", "",
          f"## Итог на {g1} / {g2}", "",
          "| Галерея | mAP@10 | R1 / R5 | Порог | F1 / TNR / полнота | 0.7·F1+0.3·TNR | PR-AUC | Строгий: TNR / полнота |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in (g1, g2):
        x, y = G[g][fin], pg(F, g)
        L.append(f"| {GALLERY_TITLE[g]} | {f4(x['mAP@10'])} | {f3(x['R1'])} / {f3(x['R5'])} | z100 {F['threshold']:.4f} | "
                 f"**{f3(y['F1'])} / {f3(y['TNR'])} / {f3(y['recall'])}** | **{f3(y['Q15'])}** | {f3(y['PR-AUC'])} | "
                 f"{f3(y['strict_TNR'])} / {f3(y['strict_recall'])} |")
    o, f1r = res["old"][fin], res["rule_f1"][fin]
    L += ["", f"Строгий порог — {F['strict']:.4f} (наименьший с TNR ≥ 0.98 на обеих).", "",
          "## Прежний порог против нового (тот же протокол)", "",
          f"| Порог | Как выбран | F1 / TNR / полнота / 0.7·F1+0.3·TNR {g1} | то же {g2} |", "|---|---|---:|---:|",
          f"| {o['threshold']:.4f} | прежний пресет (F1 при TNR ≥ 0.70, 4 ротации, большая галерея с test gallery) | "
          f"{m4(o, g1)} | {m4(o, g2)} |",
          f"| {f1r['threshold']:.4f} | правило «F1 при TNR ≥ 0.70» на этом протоколе | {m4(f1r, g1)} | {m4(f1r, g2)} |",
          f"| **{F['threshold']:.4f}** | **максимум 0.7·F1 + 0.3·TNR (итог)** | **{m4(F, g1)}** | **{m4(F, g2)}** |",
          f"| {o['strict']:.4f} | прежний строгий | {' / '.join(f3(pg(o, g1)[k]) for k in ('strict_F1', 'strict_TNR', 'strict_recall', 'strict_Q15'))} | "
          f"{' / '.join(f3(pg(o, g2)[k]) for k in ('strict_F1', 'strict_TNR', 'strict_recall', 'strict_Q15'))} |",
          f"| **{F['strict']:.4f}** | **строгий: наименьший с TNR ≥ 0.98 на обеих** | "
          f"{' / '.join(f3(pg(F, g1)[k]) for k in ('strict_F1', 'strict_TNR', 'strict_recall', 'strict_Q15'))} | "
          f"{' / '.join(f3(pg(F, g2)[k]) for k in ('strict_F1', 'strict_TNR', 'strict_recall', 'strict_Q15'))} |", "",
          f"Требование «TNR ≥ 0.70» на {g2} даёт порог {f1r['threshold']:.2f} и отдаёт полноту; оценка организаторов "
          "весит F1 больше, чем TNR, поэтому её максимум — ниже по порогу.", "",
          *(["## С почти-дублем своей камеры (Q22)", "",
             "У запроса без пары в галерее остаётся кадр его машины с той же камеры — почти-дубль, который отказ "
             f"выбрасывает (протокол Андрея: {alt['protocol']['rotations']} ротаций по {alt['protocol']['open_width']}/"
             f"{alt['protocol']['rotations']} машин, `closed_gallery_a10_q15.md`). Порог по тому же правилу — "
             f"{alt['presets'][fin]['threshold']:.4f} (строгий {alt['presets'][fin]['strict']:.4f}):", "",
             f"| Порог | F1 / TNR / полнота / 0.7·F1+0.3·TNR {g1} | то же {g2} |", "|---|---:|---:|",
             f"| {alt['presets'][fin]['threshold']:.4f} (выбран на этом протоколе) | {m4(alt['presets'][fin], g1)} | "
             f"{m4(alt['presets'][fin], g2)} |", ""] if alt else []),
          f"## Почему на {g2} TNR ниже", "",
          f"Отвлекатели — машины, которые модель видела при обучении. У запроса без пары лучший кандидат оценки отказа "
          f"(почти-дубли выброшены) — такой кадр в {pct(G[g2][fin]['open_best_distractor'])} случаев, его средний скор "
          f"{f3(G[g2][fin]['open_best_s1'])} против {f3(G[g1][fin]['open_best_s1'])} на {g1}: обученные машины — сильные "
          f"отвлекатели, z100 у запроса без пары выше, отказов меньше (TNR {f3(pg(F, g2)['TNR'])} против "
          f"{f3(pg(F, g1)['TNR'])}). В закрытом тесте отвлекатели — невиданные машины; эта галерея, вероятно, строже к "
          "TNR, чем закрытый тест.", "",
          "## Запасной план без фона вообще (если FPS с фоном не пройдёт)", "",
          f"Ранжирование то же — DBA k=1, отказ — z100 без почти-дублей только по правилу рамок (`PP_DEDUP_RANK=off`, "
          f"`PP_DEDUP_REFUSAL=bbox`, `PP_CAMERA_PENALTY=off`), порог {B['threshold']:.4f} (строгий {B['strict']:.4f}): "
          f"{m4(B, g1)} на {g1} и {m4(B, g2)} на {g2} против {m4(F, g1)} и {m4(F, g2)} у итога (F1 / TNR / полнота / "
          f"оценка); mAP@10 {mp(fb)} — то же ранжирование. Правило рамок стоит 0.04 мс на запрос против 1.4 мс у фона.", "",
          "## Воспроизвести", "", "```bash",
          "# с почти-дублем своей камеры (Q22) — сначала, его числа идут в раздел выше:",
          "python research/closed_gallery.py --data-dir <данные> --env REID_WEIGHTS=<веса a10> --zip dataset.zip --protocol q15 --tag a10_q15",
          f"python research/closed_gallery.py --data-dir <данные> --env REID_WEIGHTS=<веса a10> --zip dataset.zip --tag {res['tag']}",
          "# прежний протокол (4 ротации, F1 при TNR ≥ 0.70, 361 + 750 test gallery) — пороги 4.6506 / 9.6847:",
          "python research/closed_gallery.py ... --protocol andrey --tag a10",
          "```", "", f"Пишет `closed_gallery_{res['tag']}.json` / `.md` и этот файл; пороги для `REFUSAL_PRESETS` — в "
          "JSON, ключ `config_presets`. Около минуты на готовых эмбеддингах.", "",
          "## Оговорки", "",
          "- 150 машин holdout: на фолд ~55 запросов без пары, шаг TNR ~0.02; разброс порогов по фолдам ~0.5–1 единица z.",
          "- Отвлекатели большой галереи — обучающие машины (модель их видела), см. выше.",
          "- Эмбеддинги — из готовых JPEG-кропов; у Андрея свой кэш, числа сходятся до 0.003 mAP@10.",
          "- Закрытый тест: у запроса без пары машины в галерее нет (Q17) — основной протокол; запрос, у которого "
          "единственный позитив — кадр своей камеры (Q22), — раздел выше.",
          "- Таблица ответов организаторов в репозитории не хранится; номера ответов — по ней.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
