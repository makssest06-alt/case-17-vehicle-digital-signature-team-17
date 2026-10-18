#!/usr/bin/env python3
"""Отказ под формулу организаторов 0.7·F1 + 0.3·TNR (FAQ Q15, 20 % open-set — Q17): протокол Андрея
(research/local_postprocess/experiments_2026-10-17/eval/refusal_objective.py ветки
andrew-postprocess-throughput), наши эмбеддинги и код сервиса, галереи 361 / 1111.

- 5 ротаций по 20 % машин без пары (у них в галерее кадр одной камеры — самой частой среди их
  запросов), сид np.random.default_rng(7): первая перестановка — ротации, вторая — 2 фолда по машинам.
- Порог вне фолда: на фолде выбора — максимум 0.7·F1 + 0.3·TNR по сетке 201 квантиль оценки
  (TNR не ограничен; TNR без запросов без пары — 0), проверка на другом; среднее по 10 точкам.
- Протоколы: «как тест» (почти-дубль своей камеры в галерее есть) и «чистый» (скрыт).
- Варианты Андрея (флаги и в ранжировании, и в отказе; top-100 после выталкивания помеченных) — для
  сверки с его README. Наш итог: ранжирование DBA k=1 + фон, отказ — оценка по кандидатам без
  почти-дублей «фон ИЛИ рамка» (рамка — наши пороги и пороги Андрея); запасной — без фона.
  Оценка raw (косинус лучшего без флага) и z100. Пресеты сервиса (z100 4.6506 и 4.9039) — как в
  сервисе: top-100 по скору, помеченные выброшены.
- Перенос порога: порог, выбранный на одной галерее, применён к другой — риск, если закрытая
  галерея другого размера.

    python research/refusal_formula.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10 --zip dataset.zip
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
import closed_gallery as cg  # noqa: E402
import honest_eval as he  # noqa: E402
import threshold_vs_gallery as tvg  # noqa: E402
from app.pp import dedup as dd  # noqa: E402
from app.pp import rerank as rr  # noqa: E402
from app.pp.dba import dba  # noqa: E402

ROT, SEED, PUSH, Z_TOP = 5, 7, 1e3, 100
GALLERIES = ("361", "1111")
LARGE = ("2.2k", "10.8k")                             # справочно: где порог окажется на галерее крупнее
PRESETS = {"итог": 4.6506, "запасной": 4.9039}       # z100, api/app/config.py REFUSAL_PRESETS
README = {  # README экспериментов Андрея: 0.7·F1 + 0.3·TNR, «как тест» / «чистый»
    ("raw (без флага)", "raw"): {"361": (0.627, 0.805), "1111": (0.618, 0.695)},
    ("DBA + фон", "raw"): {"361": (0.805, 0.842), "1111": (0.702, 0.708)},
    ("DBA + фон", "z100"): {"361": (0.785, 0.826), "1111": (0.691, 0.696)},
    ("DBA + дубли Андрея", "raw"): {"361": (0.804, 0.839), "1111": (0.701, 0.699)}}
# (имя, скор, флаги ранжирования, флаги отказа, семантика top-100)
VARIANTS = (("raw (без флага)", "raw", "none", "none", "andrey"),
            ("DBA + фон", "dba", "bg", "bg", "andrey"),
            ("DBA + дубли Андрея", "dba", "dup_andrey", "dup_andrey", "andrey"),
            ("DBA + фон ИЛИ дубли Андрея", "dba", "bg|dup_andrey", "bg|dup_andrey", "andrey"),
            ("итог: DBA + фон; отказ фон ИЛИ рамка (наши пороги)", "dba", "bg", "bg|dup_ours", "service"),
            ("итог: DBA + фон; отказ фон ИЛИ рамка (пороги Андрея)", "dba", "bg", "bg|dup_andrey", "service"),
            ("запасной: DBA; отказ рамка (наши пороги)", "dba", "none", "dup_ours", "service"),
            ("DBA без флага в ранжировании; отказ фон ИЛИ рамка (наши пороги)", "dba", "none", "bg|dup_ours", "service"))
FINAL, FALLBACK, NOFLAG = VARIANTS[4][0], VARIANTS[6][0], VARIANTS[7][0]
PLATEAU = 0.005                                       # плато — пороги, где цель не ниже максимума на столько


def objective(acc, has, correct) -> tuple:
    m = cg.decision(acc, has, correct)
    tnr = m["TNR"] if np.isfinite(m["TNR"]) else 0.0
    return 0.7 * m["F1"] + 0.3 * tnr, m["F1"], tnr


def best_threshold(score, has, correct) -> float:
    fin = score[np.isfinite(score)]
    cand = np.unique(np.quantile(fin, np.linspace(0, 1, 201))) if len(fin) else np.array([np.inf])
    return float(max(cand, key=lambda t: objective(score >= t, has, correct)[0]))


def records(s, fl_rank, fl_ref, keep, junk, qv, qc, gv, gc, formula, semantics, clean) -> dict:
    k = np.flatnonzero(keep)
    S, FR, FF = s[:, k].copy(), fl_rank[:, k], fl_ref[:, k]
    if clean:
        S[junk[:, k]] = -np.inf
    top1 = np.argmax(np.where(np.isfinite(S), S - PUSH * FR, -np.inf), axis=1)
    if semantics == "andrey":                                           # как refusal_objective.py
        order = np.argsort(-(S - PUSH * FF), axis=1, kind="stable")[:, :Z_TOP]
        top1 = order[:, 0] if np.array_equal(FR, FF) else top1
    else:                                                               # сервис: top-100 по скору, потом флаги
        order = np.argsort(-S, axis=1, kind="stable")[:, :Z_TOP]
    sc = np.array([rr.refusal_score(np.where(np.isfinite(S[i, order[i]]), S[i, order[i]], -9.0),
                                    FF[i, order[i]] | ~np.isfinite(S[i, order[i]]), formula) for i in range(len(S))])
    gvk, gck = gv[k], gc[k]
    has = np.array([((gvk == qv[i]) & (gck != qc[i])).any() for i in range(len(S))])
    # верный top-1 у эталона — та же машина без учёта камеры: кадр своей камеры наверху засчитан
    return {"score": sc, "has": has, "correct": gvk[top1] == qv, "same_cam_top1": (gvk[top1] == qv) & (gck[top1] == qc)}


def evaluate(recs: list, qfold, fixed: float = None) -> dict:
    """recs — по ротациям; fixed — заданный порог, иначе лучший по формуле на фолде выбора."""
    obj, f1, tnr, thr = [], [], [], []
    for r in recs:
        for f in (0, 1):
            tune, held = qfold != f, qfold == f
            t = fixed if fixed is not None else best_threshold(r["score"][tune], r["has"][tune], r["correct"][tune])
            o, a, b = objective(r["score"][held] >= t, r["has"][held], r["correct"][held])
            obj.append(o); f1.append(a); tnr.append(b); thr.append(t)
    return {"objective": float(np.mean(obj)), "objective_std": float(np.std(obj)), "F1": float(np.mean(f1)),
            "TNR": float(np.mean(tnr)), "thr_min": float(np.min(thr)), "thr_max": float(np.max(thr)),
            "thr_median": float(np.median(thr)), "thresholds": thr}


def best_joint(per_gallery: dict) -> dict:
    """Один порог на обе галереи по всем запросам (как выбирался пресет, только цель — формула):
    максимум средней по галереям и ротациям цели; плато — пороги в PLATEAU от максимума."""
    fin = np.concatenate([r["score"][np.isfinite(r["score"])] for recs in per_gallery.values() for r in recs])
    cand = np.unique(np.quantile(fin, np.linspace(0, 1, 201)))
    mean_obj = lambda recs, t: [np.mean(x) for x in zip(*[objective(r["score"] >= t, r["has"], r["correct"])
                                                            for r in recs])]
    table = [(float(t), {g: mean_obj(recs, t) for g, recs in per_gallery.items()}) for t in cand]
    joint = [np.mean([v[0] for v in by_g.values()]) for _, by_g in table]
    i = int(np.argmax(joint))
    plateau = [t for (t, _), o in zip(table, joint) if o >= joint[i] - PLATEAU]
    pick = lambda t: next(by_g for tt, by_g in table if tt == t)
    centre = float(min(plateau, key=lambda t: abs(t - np.median(plateau))))
    fmt = lambda by_g: {g: dict(zip(("objective", "F1", "TNR"), map(float, v))) for g, v in by_g.items()}
    return {"best": table[i][0], "best_at": fmt(table[i][1]), "plateau": [min(plateau), max(plateau)],
            "centre": centre, "centre_at": fmt(pick(centre))}


def run(a) -> dict:
    import pseudo_camera as pc
    from app.config import Settings

    tau = Settings().pp_camera_tau
    D = he.Data(a)
    qrows, grows = D.valid(D.qry), D.valid(D.gal)
    boxes = {r["image_id"]: r["box"] for r in pc.read_rows(a.data_dir)}
    bg = tvg.service_bg(D, a)
    Q = he.frame(qrows)
    qids = list(Q.index)
    qv, qc = Q.vehicle_id.to_numpy(), Q.camera_id.to_numpy()
    q = D.vectors(qids)
    qbg = np.stack([bg[i] for i in qids])
    qbox = np.array([boxes[i] for i in qids], dtype="float64")
    vehicles = sorted(set(qv), key=int)
    rng = np.random.default_rng(SEED)
    rot_of = {v: n % ROT for n, v in enumerate(rng.permutation(vehicles))}
    fold_of = {v: n % 2 for n, v in enumerate(rng.permutation(vehicles))}
    keep_cam = {v: Counter(qc[qv == v]).most_common(1)[0][0] for v in vehicles}
    qfold = np.array([fold_of[v] for v in qv])
    extra = {"361": [], "1111": D.valid(D.distractors("test_gallery")), "2.2k": D.valid(D.distractors("test_all")),
             "10.8k": D.valid(D.distractors("test_all+train_seen"))}
    res = {"tag": a.tag, "protocol": f"{ROT} ротаций по {100 // ROT} % без пары, rng({SEED}), 2 фолда, 201 квантиль",
           "presets": PRESETS, "galleries": {}}
    recs_all = {}

    def arrays(gname):
        G = he.frame(grows + extra[gname])
        gids = list(G.index)
        gv, gc = G.vehicle_id.to_numpy(), G.camera_id.to_numpy()
        g = D.vectors(gids)
        raw = q @ g.T
        s = {"raw": raw, "dba": q @ dba(g, k=1, weights="cos").T}
        gbox = np.array([boxes[i] for i in gids], dtype="float64")
        fl = {"none": np.zeros(raw.shape, bool), "bg": (qbg @ np.stack([bg[i] for i in gids]).T) > tau}
        for name, rule in cg.RULES.items():
            fl[f"dup_{name}"] = dd.near_dup_flags(raw, qbox[:, None, :], gbox[None, :, :], **rule)
        for y in ("dup_ours", "dup_andrey"):
            fl[f"bg|{y}"] = fl["bg"] | fl[y]
        junk = (qv[:, None] == gv[None, :]) & (qc[:, None] == gc[None, :])
        keeps = [np.array([not (rot_of.get(v) == r and c != keep_cam.get(v)) for v, c in zip(gv, gc)])
                 for r in range(ROT)]
        return len(gids), gv, gc, s, fl, junk, keeps

    for gname in GALLERIES:
        n, gv, gc, s, fl, junk, keeps = arrays(gname)
        out = {"N": n}
        for vname, sk, fr, ff, sem in VARIANTS:
            out[vname] = {}
            for formula in ("raw", "z100"):
                for proto in ("как тест", "чистый"):
                    recs = [records(s[sk], fl[fr], fl[ff], kp, junk, qv, qc, gv, gc, formula, sem, proto == "чистый")
                            for kp in keeps]
                    recs_all[(gname, vname, formula, proto)] = recs
                    e = evaluate(recs, qfold)
                    e["same_cam_top1"] = float(np.mean([r["same_cam_top1"][r["has"]].mean() for r in recs]))
                    if formula == "z100" and proto == "как тест":
                        e["at_presets"] = {p: evaluate(recs, qfold, t) for p, t in PRESETS.items()}
                    out[vname][f"{formula} | {proto}"] = e
                r = out[vname][f"{formula} | как тест"]
                print(f"{gname:5s} {vname:52s} {formula:4s} как тест: цель {r['objective']:.3f} F1 {r['F1']:.3f} "
                      f"TNR {r['TNR']:.3f} порог {r['thr_min']:.3f}..{r['thr_max']:.3f} | чистый: цель "
                      f"{out[vname][f'{formula} | чистый']['objective']:.3f}", flush=True)
        res["galleries"][gname] = out
    res["joint"] = {f"{v} | {formula}": best_joint({g: recs_all[(g, v, formula, "как тест")] for g in GALLERIES})
                    for v in (FINAL, FALLBACK, NOFLAG) for formula in ("raw", "z100")}
    # перенос порога: выбран на одной галерее (медиана по 10 точкам), применён к другой
    res["transfer"] = {}
    for vname in (FINAL, FALLBACK):
        for formula in ("raw", "z100"):
            for src, dst in (("361", "1111"), ("1111", "361")):
                t = res["galleries"][src][vname][f"{formula} | как тест"]["thr_median"]
                own = res["galleries"][dst][vname][f"{formula} | как тест"]
                moved = evaluate(recs_all[(dst, vname, formula, "как тест")], qfold, t)
                res["transfer"][f"{vname} | {formula} | {src}→{dst}"] = {
                    "threshold": t, "own_threshold": own["thr_median"], "own_objective": own["objective"],
                    "moved_objective": moved["objective"], "moved_F1": moved["F1"], "moved_TNR": moved["TNR"]}
    # галереи крупнее: лучший порог вне фолда и цель при порогах, выбранных на 361 + 1111
    res["large"] = {}
    variants = {v[0]: v for v in VARIANTS}
    for gname in LARGE:
        n, gv, gc, s, fl, junk, keeps = arrays(gname)
        res["large"][gname] = {"N": n}
        for vname in (FINAL, NOFLAG):
            _, sk, fr, ff, sem = variants[vname]
            for formula in ("raw", "z100"):
                recs = [records(s[sk], fl[fr], fl[ff], kp, junk, qv, qc, gv, gc, formula, sem, False) for kp in keeps]
                fixed = {"центр плато 361 + 1111": res["joint"][f"{vname} | {formula}"]["centre"]}
                if formula == "z100":
                    fixed["пресет итога"] = PRESETS["итог"]
                e = evaluate(recs, qfold)
                e["at"] = {k: {"threshold": t, **evaluate(recs, qfold, t)} for k, t in fixed.items()}
                res["large"][gname][f"{vname} | {formula}"] = e
                print(f"{gname:5s} {vname:52s} {formula:4s} лучший вне фолда: цель {e['objective']:.3f} порог "
                      f"{e['thr_median']:.3f} | " + " | ".join(f"{k} {x['threshold']:.4f}: цель {x['objective']:.3f} "
                                                              f"TNR {x['TNR']:.3f}" for k, x in e["at"].items()),
                      flush=True)
    return res


def report_md(res: dict) -> str:
    G = res["galleries"]
    f3 = lambda x: f"{x:.3f}"
    cell = lambda v, k, g: G[g][v][k]
    L = [f"# Отказ под формулу 0.7·F1 + 0.3·TNR: {res['tag']}", "",
         f"Протокол Андрея (`refusal_objective.py`): {res['protocol']}; порог — максимум формулы на фолде выбора "
         "без ограничения TNR, проверка на другом фолде. Галереи 361 / 1111. Воспроизвести: `python "
         f"research/refusal_formula.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag {res['tag']} --zip dataset.zip`.", "",
         "## 1. Сверка с README Андрея (цель, «как тест» / «чистый»)", "",
         "| Вариант | Оценка | 361 наш / Андрея | 1111 наш / Андрея |", "|---|---|---:|---:|"]
    for (v, formula), ref in README.items():
        row = []
        for g in GALLERIES:
            a = cell(v, f"{formula} | как тест", g)["objective"]
            b = cell(v, f"{formula} | чистый", g)["objective"]
            row.append(f"{f3(a)} / {f3(b)} — {f3(ref[g][0])} / {f3(ref[g][1])}")
        L.append(f"| {v} | {formula} | {row[0]} | {row[1]} |")
    diffs = [abs(cell(v, f"{fm} | {p}", g)["objective"] - ref[g][i]) for (v, fm), ref in README.items()
             for g in GALLERIES for i, p in enumerate(("как тест", "чистый"))]
    L += ["", f"Наибольшее расхождение с README — {max(diffs):.3f}, среднее {np.mean(diffs):.3f}. Отличие — "
          "эмбеддинги (наши — по JPEG-кропам, его — свой кэш a10); код протокола — его один в один.", "",
          "## 2. Наш итог: raw против z100 (порог вне фолда)", "",
          "Цель «как тест» / «чистый»; F1 / TNR и пороги (min..max по 10 точкам) — «как тест». «Своя камера "
          "в top-1» — доля запросов с парой, у которых верхний кандидат — кадр той же машины с той же камеры: "
          "эталон (`official_evaluate.candidate_metrics`) сверяет только `vehicle_id` и засчитывает его верным.", "",
          "| Вариант | Оценка | Цель 361 | F1 / TNR 361 | Порог 361 | Цель 1111 | F1 / TNR 1111 | Порог 1111 | "
          "Своя камера в top-1, 361 / 1111 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for v in (FINAL, VARIANTS[5][0], FALLBACK, NOFLAG):
        for formula in ("raw", "z100"):
            a, b = G["361"][v][f"{formula} | как тест"], G["1111"][v][f"{formula} | как тест"]
            ca, cb = G["361"][v][f"{formula} | чистый"], G["1111"][v][f"{formula} | чистый"]
            L.append(f"| {v} | {formula} | **{f3(a['objective'])}** / {f3(ca['objective'])} | {f3(a['F1'])} / "
                     f"{f3(a['TNR'])} | {a['thr_min']:.3f}..{a['thr_max']:.3f} | **{f3(b['objective'])}** / "
                     f"{f3(cb['objective'])} | {f3(b['F1'])} / {f3(b['TNR'])} | {b['thr_min']:.3f}..{b['thr_max']:.3f} | "
                     f"{a['same_cam_top1']:.3f} / {b['same_cam_top1']:.3f} |")
    L += ["", "## 3. Цель при пресетах сервиса (z100, «как тест»)", "",
          "Пресеты подобраны на тех же 361 / 1111 (F1 при TNR ≥ 0.70, протокол `head_to_head.py`), поэтому их "
          "строки — в выборке и выше честной оценки; «лучший по формуле» — вне фолда.", "",
          "| Вариант | Порог | Цель 361 | F1 / TNR 361 | Цель 1111 | F1 / TNR 1111 |", "|---|---:|---:|---:|---:|---:|"]
    for v, p in ((FINAL, "итог"), (FALLBACK, "запасной")):
        a, b = G["361"][v]["z100 | как тест"]["at_presets"][p], G["1111"][v]["z100 | как тест"]["at_presets"][p]
        L.append(f"| {v} | пресет {res['presets'][p]} | {f3(a['objective'])} | {f3(a['F1'])} / {f3(a['TNR'])} | "
                 f"{f3(b['objective'])} | {f3(b['F1'])} / {f3(b['TNR'])} |")
        for formula in ("z100", "raw"):
            a, b = G["361"][v][f"{formula} | как тест"], G["1111"][v][f"{formula} | как тест"]
            L.append(f"| {v} | лучший по формуле, {formula} (медиана {a['thr_median']:.3f} / {b['thr_median']:.3f}) | "
                     f"{f3(a['objective'])} | {f3(a['F1'])} / {f3(a['TNR'])} | {f3(b['objective'])} | "
                     f"{f3(b['F1'])} / {f3(b['TNR'])} |")
    L += ["", "## 4. Один порог по формуле на обе галереи («как тест», в выборке — как пресет)", "",
          f"Максимум средней по 361 и 1111 цели по всем запросам; плато — пороги, где цель не ниже максимума на "
          f"{PLATEAU}, центр плато — порог в его середине (Андрей: «брать из середины плато»).", "",
          "| Вариант | Оценка | Порог-максимум | Плато | Центр плато | Цель 361 | F1 / TNR 361 | Цель 1111 | "
          "F1 / TNR 1111 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, j in res["joint"].items():
        name, formula = k.split(" | ")
        a, b = j["centre_at"]["361"], j["centre_at"]["1111"]
        L.append(f"| {name} | {formula} | {j['best']:.4f} | {j['plateau'][0]:.4f}..{j['plateau'][1]:.4f} | "
                 f"**{j['centre']:.4f}** | {f3(a['objective'])} | {f3(a['F1'])} / {f3(a['TNR'])} | "
                 f"{f3(b['objective'])} | {f3(b['F1'])} / {f3(b['TNR'])} |")
    L += ["", "## 5. Галереи крупнее (справочно, «как тест»)", "",
          "Дистракторы 2.2k — весь тест, 10.8k — ещё и кадры train (модель их видела, поэтому TNR на ней "
          "оптимистичен). Строка «вне фолда» — где оказался бы лучший порог, остальные — цель при порогах, "
          "выбранных на 361 + 1111.", "",
          "| Галерея | Вариант | Оценка | Порог | Цель | F1 / TNR |", "|---|---|---|---|---:|---:|"]
    for gname, by_v in res.get("large", {}).items():
        for k, e in by_v.items():
            if k == "N":
                continue
            name, formula = k.split(" | ")
            L.append(f"| {gname} ({by_v['N']}) | {name} | {formula} | лучший вне фолда: медиана {e['thr_median']:.3f} "
                     f"({e['thr_min']:.3f}..{e['thr_max']:.3f}) | {f3(e['objective'])} | {f3(e['F1'])} / {f3(e['TNR'])} |")
            for lab, x in e["at"].items():
                L.append(f"| {gname} ({by_v['N']}) | {name} | {formula} | {lab}: {x['threshold']:.4f} | "
                         f"{f3(x['objective'])} | {f3(x['F1'])} / {f3(x['TNR'])} |")
    L += ["", "## 6. Перенос порога между 361 и 1111 («как тест»)", "",
          "Медиана порогов, выбранных вне фолда на одной галерее, применена ко всем точкам другой.", "",
          "| Вариант | Оценка | Откуда → куда | Порог источника | Свой порог | Цель со своим | Цель с перенесённым | "
          "F1 / TNR с перенесённым |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for k, v in res["transfer"].items():
        name, formula, way = k.split(" | ")
        L.append(f"| {name} | {formula} | {way} | {v['threshold']:.3f} | {v['own_threshold']:.3f} | "
                 f"{f3(v['own_objective'])} | {f3(v['moved_objective'])} | {f3(v['moved_F1'])} / {f3(v['moved_TNR'])} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = bp.data_args(argparse.ArgumentParser())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    out = a.out_dir / f"refusal_formula_{a.tag}.json"
    if not a.report:
        out.write_text(json.dumps(run(a), ensure_ascii=False, indent=2), encoding="utf-8")
    res = json.loads(out.read_text(encoding="utf-8"))
    (a.out_dir / f"refusal_formula_{a.tag}.md").write_text(report_md(res), encoding="utf-8")
    print(f"-> {a.out_dir / f'refusal_formula_{a.tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
