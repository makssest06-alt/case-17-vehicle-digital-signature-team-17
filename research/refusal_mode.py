#!/usr/bin/env python3
"""Режим отказа: оценка уверенности, порог и его поведение на разных размерах галереи.

Одна команда — все числа для docs/refusal.md и REFUSAL_PRESETS в api/app/config.py:

    python research/refusal_mode.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10 [--zip dataset.zip]

Протокол «как сервис» (neardup: у запроса без пары в галерее остаётся кадр его машины
с той же камеры), 150 машин holdout, 5 фолдов по 30 машин, 25 % машин фолда без пары, метрики
эталона. Галереи — сетка N (holdout + случайная доля чужих кадров: тест и обучающие
машины) и именованные 2.2k / 9.0k / 10.8k; тест — только чужие кадры, ничего на нём не
подбирается. Скор считает код сервиса (research/threshold_vs_gallery.records_service →
api/app/pp): без постобработки и DBA k=1 + штраф за ту же камеру; оценка raw и z100.

Порог: на фолде выбора — максимум F1 при TNR ≥ 0.70 по записям ВСЕХ размеров сетки
сразу (размер закрытой выборки неизвестен), проверка — на остальных четырёх фолдах на
каждом размере. Строгая точка — наименьший порог с TNR ≥ 0.98 на фолде выбора.
Итоговая формула выбирается правилом, записанным до прогона (choose_formula): TNR ≥ 0.70
при одном пороге на всех размерах, из прошедших — больше средний F1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "api"))
import calibrate_refusal_folds as crf  # noqa: E402
import honest_eval as he  # noqa: E402
import official_evaluate as oe  # noqa: E402
import refusal_score as rsc  # noqa: E402
import service_embeddings as se  # noqa: E402
import threshold_vs_gallery as tvg  # noqa: E402

VARIANTS = (("none", "без постобработки", {"dba_k": 0, "dedup": "off"}),
            ("dba1+camera", "DBA k=1 + штраф за ту же камеру", {"dba_k": 1, "dedup": "bg"}),
            ("dba1+bbox", "DBA k=1 + правило Андрея (сходство + рамка)", {"dba_k": 1, "dedup": "bbox"}),
            ("dba1+both", "DBA k=1 + оба (OR)", {"dba_k": 1, "dedup": "both"}))
FORMULAS = ("raw", "z100")
TNR_FLOOR, STRICT_TNR = 0.70, 0.98
PRESET_KEY = {"none": (0, "off"), "dba1+camera": (1, "bg"), "dba1+bbox": (1, "bbox"), "dba1+both": (1, "both")}
# Итоговый вариант (предварительно, решение за Андреем): DBA k=1 + оба исключения (OR) — выше TNR при том же F1 и
# меньше всего ответов «та же машина с той же камеры» у запросов без пары (dedup_compare_a10.md).
FINAL = "dba1+both"
SHORT = {"dba1+camera": "DBA + штраф по фону", "dba1+bbox": "DBA + правило Андрея", "dba1+both": "DBA + фон OR рамка"}


CURRENT = 0.8025                         # SEARCH_THRESHOLD сервиса сейчас (raw, без постобработки)


def fstats(vals: list) -> dict:
    """Статистика по конечным порогам; ∞ — фолд, где ни один порог не дал нужный TNR."""
    fin = [v for v in vals if np.isfinite(v)]
    return {**(rsc.stats(fin) if fin else {"mean": float("inf"), "std": float("nan"), "n": 0}),
            "inf_folds": len(vals) - len(fin)}


def as_rec(r: dict, formula: str) -> dict:
    return {"conf": r[f"conf_{formula}"], "has_match": r["has_match"], "correct": r["correct"]}


def pooled(recs: list) -> dict:
    return {k: np.concatenate([r[k] for r in recs]) for k in ("conf", "has_match", "correct")}


def choose_tnr(rec: dict, floor: float) -> float:
    """Наименьший порог с TNR ≥ floor (строгая точка)."""
    for t in np.concatenate((np.unique(rec["conf"][np.isfinite(rec["conf"])]), [np.inf])):
        m = crf.metrics(rec, float(t))
        if m["TNR"] == m["TNR"] and m["TNR"] >= floor:
            return float(t)
    return float("inf")


def size_table(by_size: dict, thr: list, strict: list, formula: str, folds: int) -> dict:
    """by_size: {ключ: (N, [записи по фолдам])} -> метрики вне фолда при выбранных порогах."""
    out = {}
    for key, (n, recs) in by_size.items():
        rf = [as_rec(r, formula) for r in recs]
        per, per_s = [], []
        for f in range(folds):
            rest = pooled([rf[i] for i in range(folds) if i != f])
            per.append(crf.metrics(rest, thr[f]))
            per_s.append(crf.metrics(rest, strict[f]))
        st = lambda rows, k: rsc.stats([p[k] for p in rows])
        out[str(key)] = {
            "N": int(n),
            "own_threshold": fstats([rsc.choose_f1(rf[f], TNR_FLOOR) for f in range(folds)]),
            **{k: st(per, k) for k in ("F1", "TNR", "precision", "recall")},
            "strict": {k: st(per_s, k) for k in ("TNR", "recall", "precision")},
            "PR-AUC": rsc.stats([oe.pr_auc(r["conf"], r["has_match"].astype(int)) for r in rf]),
            "mAP@10": rsc.stats([r["mAP@10"] for r in recs]), "Rank-1": rsc.stats([r["Rank-1"] for r in recs]),
            "Rank-5": rsc.stats([r["Rank-5"] for r in recs]),
            "mINP": rsc.stats([r["mINP"] for r in recs]),
            "open_s1": float(np.mean(np.concatenate([r["s1"][~r["has_match"]] for r in recs]))),
        }
    return out


def thr_by_n(out: dict, ns: list, nsize: dict, folds: int) -> dict:
    """raw с порогом от размера галереи: thr(N) = a + b·√(2 ln N), N — сколько кадров в
    галерее сервиса сейчас. a, b — по лучшим порогам для каждого N на фолде выбора
    (research/threshold_vs_gallery.py), проверка — на остальных фолдах."""
    form = "sqrt2ln"
    own = {n: [rsc.choose_f1(as_rec(out["grid"][n][f], "raw"), TNR_FLOOR) for f in range(folds)] for n in ns}
    fits = []
    for f in range(folds):
        pts = [(n, own[n][f]) for n in ns if np.isfinite(own[n][f])]
        fits.append(tvg.fit([p[0] for p in pts], [p[1] for p in pts], form))
    rule = lambda f, n: fits[f][0] + fits[f][1] * float(tvg.FORMS[form](n))
    sizes = {}
    for key, (n, recs) in {**{n: (n, out["grid"][n]) for n in ns},
                           **{k: (nsize[k], out["named"][k]) for k in out["named"]}}.items():
        t = [rule(f, n) for f in range(folds)]
        sizes.update(size_table({key: (n, recs)}, t, t, "raw", folds))
        sizes[str(key)]["threshold"] = rsc.stats(t)
    return {"form": form, "a": rsc.stats([p[0] for p in fits]), "b": rsc.stats([p[1] for p in fits]), "sizes": sizes}


def choose_formula(res: dict, variant: str) -> str:
    """Правило, записанное до прогона: TNR ≥ 0.70 на всех размерах сетки при одном пороге,
    из прошедших — больший средний F1; если не прошла ни одна — больший минимум TNR."""
    def grid(f):
        return [v for v in res[variant][f]["sizes"].values() if not v.get("named")]
    ok = [f for f in FORMULAS if min(v["TNR"]["mean"] for v in grid(f)) >= TNR_FLOOR]
    if ok:
        return max(ok, key=lambda f: np.mean([v["F1"]["mean"] for v in grid(f)]))
    return max(FORMULAS, key=lambda f: min(v["TNR"]["mean"] for v in grid(f)))


def run(D, a, folds: list) -> dict:
    bg = tvg.service_bg(D, a)
    rule = tvg.service_rule(a)
    from app.config import Settings

    st = Settings()
    res = {"tag": a.tag, "tnr_floor": TNR_FLOOR, "strict_tnr": STRICT_TNR, "tau": st.pp_camera_tau,
           "topk": st.pp_camera_topk}
    for key, label, v in VARIANTS:
        sp = {"mode": "neardup", "tau": st.pp_camera_tau, "topk": st.pp_camera_topk, "bg": bg, "extra": True,
              **rule, **v}
        out = tvg.build(D, None, "s1", folds, sp)
        ns = sorted(n for n in out["grid"] if len(out["grid"][n]) == a.folds)
        nsize = {name: int(round(np.mean([out["base"][f] + len(D.valid(D.distractors(kind))) for f in range(a.folds)])))
                 for name, kind in tvg.NAMED}
        res[key] = {"label": label, "grid": ns, "base_gallery": out["base"]}
        for formula in FORMULAS:
            thr = [rsc.choose_f1(pooled([as_rec(out["grid"][n][f], formula) for n in ns]), TNR_FLOOR)
                   for f in range(a.folds)]
            strict = [choose_tnr(pooled([as_rec(out["grid"][n][f], formula) for n in ns]), STRICT_TNR)
                      for f in range(a.folds)]
            sizes = size_table({n: (n, out["grid"][n]) for n in ns}, thr, strict, formula, a.folds)
            named = size_table({name: (nsize[name], out["named"][name]) for name in out["named"]}, thr, strict,
                               formula, a.folds)
            for v_ in named.values():
                v_["named"] = True
            res[key][formula] = {"threshold": fstats(thr), "thresholds": thr, "strict_threshold": fstats(strict),
                                 "strict_thresholds": strict, "sizes": {**sizes, **named}}
            # F1-оптимум без ограничения на TNR — сколько F1 отдаём за TNR ≥ 0.70
            free = [rsc.choose_f1(pooled([as_rec(out["grid"][n][f], formula) for n in ns]), None) for f in range(a.folds)]
            res[key][formula]["free"] = {
                "threshold": fstats(free), "thresholds": free,
                "sizes": {**size_table({n: (n, out["grid"][n]) for n in ns}, free, free, formula, a.folds),
                          **size_table({name: (nsize[name], out["named"][name]) for name in out["named"]}, free, free,
                                       formula, a.folds)}}
            if formula == "raw":                                      # порог от размера галереи, thr(N)
                res[key]["thr_n"] = thr_by_n(out, ns, nsize, a.folds)
            if key == "none" and formula == "raw":                    # как работает сервис сейчас
                cur = [CURRENT] * a.folds
                res[key]["current"] = {**size_table({n: (n, out["grid"][n]) for n in ns}, cur, cur, formula, a.folds),
                                       **size_table({name: (nsize[name], out["named"][name]) for name in out["named"]},
                                                    cur, cur, formula, a.folds)}
            print(f"{label:34s} {formula:5s} порог {np.mean(thr):.4f} строгий {np.mean(strict):.4f}  " +
                  "  ".join(f"{k}: F1 {x['F1']['mean']:.3f} TNR {x['TNR']['mean']:.3f}"
                            for k, x in res[key][formula]["sizes"].items() if k in ("1000", "2.2k", "10.8k")),
                  flush=True)
        res[key]["chosen_formula"] = choose_formula(res, key)
    return res


def presets(res: dict) -> dict:
    """Строки для REFUSAL_PRESETS в api/app/config.py."""
    out = {}
    for key, _, _ in VARIANTS:
        for formula in FORMULAS:
            if (key, formula) == ("none", "raw"):
                continue                       # текущий пресет сервиса (Глеб, без почти-дублей) не переписываем
            r = res[key][formula]
            k, cam = PRESET_KEY[key]
            out[f"({k}, \"{cam}\", \"{formula}\")"] = {
                "balanced": round(r["threshold"]["mean"], 4), "strict": round(r["strict_threshold"]["mean"], 4),
                "note": f"предварительно: {res[key]['label']}, {formula}, F1 при TNR ≥ {TNR_FLOOR:.2f} по сетке "
                        f"N {res[key]['grid'][0]}–{res[key]['grid'][-1]}, как сервис (refusal_mode_{res['tag']}.md)"}
    return out


def markdown(res: dict) -> str:
    tag = res["tag"]
    L = [f"# Режим отказа: {tag}", "",
         "Протокол «как сервис» (у запроса без пары в галерее остаётся кадр его машины с той же камеры), 5 фолдов по "
         "150 машинам holdout, 25 % машин фолда без пары. Порог — максимум F1 при TNR ≥ 0.70 на фолде выбора по "
         "всем размерам сетки сразу, метрики — на остальных фолдах. Строгая точка — наименьший порог с TNR ≥ 0.98. "
         f"Штраф: τ = {res['tau']:.5f}, top-{res['topk']}. Воспроизвести: `python research/refusal_mode.py "
         f"--data-dir <данные> --env REID_WEIGHTS=<веса> --tag {tag}`.", "",
         "## Пороги", "",
         "| Постобработка | Оценка | Порог (± по фолдам) | Строгий | TNR по размерам, мин–макс | F1 по размерам, мин–макс | "
         "Выбор по правилу |", "|---|---|---:|---:|---:|---:|---|"]
    for key, label, _ in VARIANTS:
        for f in FORMULAS:
            r = res[key][f]
            g = [v for v in r["sizes"].values() if not v.get("named")]
            tn, f1 = [v["TNR"]["mean"] for v in g], [v["F1"]["mean"] for v in g]
            inf_note = lambda st: f" (∞ на {st['inf_folds']} фолд.)" if st["inf_folds"] else ""
            L.append(f"| {label} | {f} | {r['threshold']['mean']:.4f} ± {r['threshold']['std']:.4f}"
                     f"{inf_note(r['threshold'])} | {r['strict_threshold']['mean']:.4f}{inf_note(r['strict_threshold'])} | "
                     f"{min(tn):.3f}–{max(tn):.3f} | {min(f1):.3f}–{max(f1):.3f} | "
                     f"{'**да**' if res[key]['chosen_formula'] == f else ''} |")
    cur = res["none"]["current"]
    L += ["", f"## Как сейчас: raw, порог {CURRENT}, без постобработки", "",
          "| Галерея | Кадров | F1 | TNR | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in cur.items():
        L.append(f"| {'сетка' if k.isdigit() else k} | {v['N']} | {v['F1']['mean']:.3f} | {v['TNR']['mean']:.3f} | "
                 f"{v['precision']['mean']:.3f} | {v['recall']['mean']:.3f} |")
    for key, label, _ in VARIANTS:
        for f in FORMULAS:
            r = res[key][f]
            L += ["", f"## {label}, {f}", "",
                  "| Галерея | Кадров | Лучший порог для этого N | F1 | TNR | Precision | Recall | Строгий: TNR / Recall | "
                  "PR-AUC | mAP@10 | mINP | s1 у запросов без пары |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for k, v in r["sizes"].items():
                own = v["own_threshold"]
                L.append(f"| {'сетка' if k.isdigit() else k} | {v['N']} | {own['mean']:.4f}"
                         f"{' (∞ на ' + str(own['inf_folds']) + ' фолд.)' if own['inf_folds'] else ''} | "
                         f"{v['F1']['mean']:.3f} | {v['TNR']['mean']:.3f} | {v['precision']['mean']:.3f} | "
                         f"{v['recall']['mean']:.3f} | {v['strict']['TNR']['mean']:.3f} / {v['strict']['recall']['mean']:.3f} | "
                         f"{v['PR-AUC']['mean']:.3f} | {v['mAP@10']['mean']:.3f} | {v['mINP']['mean']:.3f} | "
                         f"{v['open_s1']:.4f} |")
    return "\n".join(L) + "\n"


def docs_md(res: dict) -> str:
    """docs/refusal.md — текст с числами из refusal_mode_<tag>.json (тот же прогон)."""
    tag, fin = res["tag"], res.get("final", "dba1+camera")
    lab, sh = res[fin]["label"], SHORT[fin]
    pick = res[fin]["chosen_formula"]
    R, Z, raw = res[fin][pick], res[fin]["z100"], res[fin]["raw"]
    cur, tn = res["none"]["current"], res[fin]["thr_n"]
    grid = lambda r: {k: v for k, v in r["sizes"].items() if k.isdigit()} if "sizes" in r else r
    first, last = res[fin]["grid"][0], res[fin]["grid"][-1]
    g = lambda d, n, k: d[str(n)][k]["mean"] if isinstance(d[str(n)][k], dict) else d[str(n)][k]
    rng = lambda d, k: (min(v[k]["mean"] for kk, v in d.items() if kk.isdigit()),
                        max(v[k]["mean"] for kk, v in d.items() if kk.isdigit()))
    f3 = lambda x: f"{x:.3f}"
    t_lo, t_hi = rng(R["sizes"], "TNR")
    f_lo, f_hi = rng(R["sizes"], "F1")
    n_lo, n_hi = rng(tn["sizes"], "TNR")
    keys = [str(n) for n in res[fin]["grid"]] + ["2.2k", "9.0k", "10.8k"]
    num = lambda n: f"{int(n):,}".replace(",", " ")
    name = lambda k: num(k) if k.isdigit() else f"{k} ({num(R['sizes'][k]['N'])})"
    L = ["# Режим отказа", "",
         f"> **Предварительно.** Итог ниже посчитан для постобработки «{lab}», которую команда "
         "ещё не приняла (решение за Андреем). После решения пересчитать одной командой (раздел «Воспроизвести») и "
         "сменить одно число в `api/app/config.py`.", "",
         "## Что считается отказом", "",
         "- `/v1/search` отвечает `refusal: true` с `reason: low_similarity`, если оценка уверенности лучшего кандидата "
         "меньше порога, и `gallery_empty`, если галерея пуста. Иначе — кандидаты.",
         "- В метрике эталона (`research/official_evaluate.py`, `candidate_metrics`): у запроса **без пары** (в галерее "
         "нет кадра его машины с другой камеры) отказ — TN, ответ — FP; у запроса **с парой** ответ с верной машиной на "
         "первом месте — TP, с чужой — FP, отказ — FN. TNR = TN / (TN + FP без пары), F1 — по TP, FP, FN. PR-AUC — "
         "площадь под кривой точность–полнота оценки уверенности для «у запроса есть пара», от порога не зависит. "
         "mAP@10 и mINP — качество ранжирования (mINP: доля позитивов до последнего позитива включительно).", "",
         "## Какая оценка уверенности", "",
         "- `raw` — косинус лучшего кандидата.",
         "- `z100` = (s1 − mean(s2..s100)) / std(s2..s100): насколько лучший кандидат выделяется среди ста ближайших.",
         "- С постобработкой обе считаются после DBA и штрафа, по кандидатам без флага «та же камера». Код — "
         "`api/app/pp/rerank.py`; исследование считает тем же кодом.",
         "- Где в конфиге: блок «режим отказа» в `api/app/config.py` — `REFUSAL_SCORE` (формула) и `SEARCH_THRESHOLD` "
         "(+ `SEARCH_THRESHOLD_STRICT` для строгого режима). Посчитанные значения — `REFUSAL_PRESETS` там же. "
         "`docker-compose.yml` и `.env.example` своих чисел не держат.", "",
         "## Как выбран порог", "",
         "- **Протокол «как сервис»:** у запроса без пары в галерее остаётся кадр его машины с той же камеры — так "
         "выглядит галерея сервиса. 150 машин holdout, 5 фолдов по 30 машин (в обучение модели не входили), 25 % машин фолда — "
         "без пары.",
         f"- **Размеры галереи:** кадры holdout плюс случайная доля чужих кадров (весь тест и кадры обучающих машин), "
         f"N от {num(first)} до {num(last)}, и именованные 2.2k, 9.0k, 10.8k. Тест — только как чужие кадры, ничего "
         "на нём не подбирается.",
         "- **Порог:** на фолде выбора — максимум F1 при TNR ≥ 0.70 по записям **всех размеров сразу** (размер "
         "закрытой выборки неизвестен), проверка — на остальных четырёх фолдах на каждом размере. Строгая точка — "
         "наименьший порог с TNR ≥ 0.98.",
         "- **Формула:** выбирается правилом, записанным до прогона. Сначала оставить формулы, у которых при одном "
         "пороге TNR ≥ 0.70 на всех размерах, и из них взять ту, где больше средний F1. Если не прошла ни одна, взять "
         "ту, у которой больше минимальный TNR.", "",
         "## Почему сырой порог зависит от размера галереи", "",
         f"- У запроса без пары лучший кандидат — самая похожая чужая машина, то есть максимум из N сходств. С ростом N "
         f"он растёт: средний s1 у запросов без пары ({sh}) — {g(raw['sizes'], first, 'open_s1'):.3f} при N = "
         f"{num(first)} и {g(raw['sizes'], last, 'open_s1'):.3f} при N = {num(last)}. Рост ложится на a + b·√(2 ln N), "
         "как у максимума N случайных величин с гауссовым хвостом.",
         f"- Лучший порог raw для каждого N отдельно растёт так же: {g(raw['sizes'], first, 'own_threshold'):.3f} → "
         f"{g(raw['sizes'], last, 'own_threshold'):.3f}. Один порог raw на все размеры ({raw['threshold']['mean']:.4f}) "
         f"даёт TNR {f3(g(raw['sizes'], first, 'TNR'))} на малой галерее и {f3(g(raw['sizes'], last, 'TNR'))} на "
         f"10.8k: чужие проходят.",
         f"- У z100 лучший порог от N почти не зависит ({min(v['own_threshold']['mean'] for k, v in Z['sizes'].items() if k.isdigit() and v['own_threshold']['n']):.2f}–"
         f"{max(v['own_threshold']['mean'] for k, v in Z['sizes'].items() if k.isdigit() and v['own_threshold']['n']):.2f}): "
         "оценка меряет отрыв лучшего от фона, а фон растёт вместе с N.",
         f"- Без постобработки картина другая и хуже. Решает почти-дубль: у запроса без пары лучший кандидат — его же "
         f"машина с той же камеры. Сервис сейчас (raw, порог 0.8025) на этом протоколе даёт TNR "
         f"{f3(cur['2.2k']['TNR']['mean'])} на 2.2k и {f3(cur['10.8k']['TNR']['mean'])} на 10.8k при F1 "
         f"{f3(cur['2.2k']['F1']['mean'])}. F1 у него выше итогового только потому, что он почти не отказывает, а "
         "запросов с парой 75 %. Прежние 0.8025 / TNR 0.889 посчитаны без почти-дублей.", "",
         "## Что делаем", "",
         f"- **Итог (предварительно): {lab}, оценка `{pick}`, порог "
         f"{R['threshold']['mean']:.4f} ± {R['threshold']['std']:.4f} (по фолдам), строгий "
         f"{R['strict_threshold']['mean']:.4f}.** TNR на размерах {f3(t_lo)}–{f3(t_hi)}, F1 {f3(f_lo)}–{f3(f_hi)}. "
         f"Правило TNR ≥ 0.70 на всех размерах при одном пороге не выполнила ни одна формула; z100 выбран по "
         "запасной ветке правила (больший минимум TNR).",
         (lambda fr, a2, b2, a8, b8: f"- **Цена TNR.** F1-оптимум `{pick}` без ограничения на TNR — порог "
          f"{fr['threshold']['mean']:.4f}: F1 / TNR {f3(a2['F1']['mean'])} / {f3(a2['TNR']['mean'])} на 2.2k и "
          f"{f3(a8['F1']['mean'])} / {f3(a8['TNR']['mean'])} на 10.8k. С нашим порогом F1 меняется на "
          f"{b2['F1']['mean'] - a2['F1']['mean']:+.3f} на 2.2k и {b8['F1']['mean'] - a8['F1']['mean']:+.3f} на "
          f"10.8k, TNR — на {b2['TNR']['mean'] - a2['TNR']['mean']:+.3f} и "
          f"{b8['TNR']['mean'] - a8['TNR']['mean']:+.3f} (таблица ниже).")(
             R["free"], R["free"]["sizes"]["2.2k"], R["sizes"]["2.2k"], R["free"]["sizes"]["10.8k"], R["sizes"]["10.8k"]),
         f"- **Альтернатива — raw с порогом от размера галереи:** thr(N) = {tn['a']['mean']:.3f} + "
         f"{tn['b']['mean']:.4f}·√(2 ln N), где N — сколько кадров в галерее сервиса в момент запроса. Держит TNR "
         f"{f3(n_lo)}–{f3(n_hi)} на всех размерах, но F1 на 10.8k {f3(tn['sizes']['10.8k']['F1']['mean'])} против "
         f"{f3(R['sizes']['10.8k']['F1']['mean'])} у `{pick}`. В сервисе не реализовано: нужно два числа, а не одно.",
         *([f"- **Другие исключения почти-дублей** (разбор — `dedup_compare_a10.md`), та же оценка `{pick}`: "
            + "; ".join(f"«{res[v]['label']}» — порог {res[v][pick]['threshold']['mean']:.4f}, F1 / TNR "
                        f"{f3(res[v][pick]['sizes']['2.2k']['F1']['mean'])} / {f3(res[v][pick]['sizes']['2.2k']['TNR']['mean'])} "
                        f"на 2.2k и {f3(res[v][pick]['sizes']['10.8k']['F1']['mean'])} / "
                        f"{f3(res[v][pick]['sizes']['10.8k']['TNR']['mean'])} на 10.8k"
                        for v in ("dba1+camera", "dba1+bbox", "dba1+both") if v in res and v != fin) + ". Таблица ниже."]
           if "dba1+both" in res else []),
         "- **Порог не переносится между вариантами постобработки.** Включили её — смените `SEARCH_THRESHOLD` (и "
         "`STRICT`) на значение из `REFUSAL_PRESETS`. Иначе сервис предупредит в логе при старте.", "",
         f"## Цифры по размерам галереи: итог, `{pick}`", "",
         "| Галерея (кадров) | F1 | TNR | Precision | Recall | Строгий: TNR / Recall | PR-AUC | mAP@10 | R1 | R5 | mINP |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        v = R["sizes"][k]
        L.append(f"| {name(k)} | {f3(v['F1']['mean'])} | {f3(v['TNR']['mean'])} | {f3(v['precision']['mean'])} | "
                 f"{f3(v['recall']['mean'])} | {f3(v['strict']['TNR']['mean'])} / {f3(v['strict']['recall']['mean'])} | "
                 f"{f3(v['PR-AUC']['mean'])} | {f3(v['mAP@10']['mean'])} | {f3(v['Rank-1']['mean'])} | "
                 f"{f3(v['Rank-5']['mean'])} | {f3(v['mINP']['mean'])} |")
    fr = R["free"]
    L += ["", f"## Сколько F1 отдаём за TNR: {sh}, `{pick}`", "",
          f"Две точки одной оценки: F1-оптимум без ограничения на TNR (порог {fr['threshold']['mean']:.4f} ± "
          f"{fr['threshold']['std']:.4f}) и наш порог {R['threshold']['mean']:.4f} (F1 при TNR ≥ 0.70); рядом — "
          "сервис сейчас. Оба порога выбраны вне фолда по всем размерам сразу.", "",
          f"| Галерея (кадров) | Сейчас: raw 0.8025, F1 / TNR | `{pick}` F1-оптимум, F1 / TNR | `{pick}` наш порог, "
          "F1 / TNR | ΔF1 (наш − оптимум) | ΔTNR (наш − оптимум) |", "|---|---:|---:|---:|---:|---:|"]
    for k in keys:
        a_, b_ = fr["sizes"][k], R["sizes"][k]
        L.append(f"| {name(k)} | {f3(cur[k]['F1']['mean'])} / {f3(cur[k]['TNR']['mean'])} | "
                 f"{f3(a_['F1']['mean'])} / {f3(a_['TNR']['mean'])} | **{f3(b_['F1']['mean'])} / {f3(b_['TNR']['mean'])}** | "
                 f"{b_['F1']['mean'] - a_['F1']['mean']:+.3f} | {b_['TNR']['mean'] - a_['TNR']['mean']:+.3f} |")
    other = "raw" if pick == "z100" else "z100"
    extra = [v for v in ("dba1+camera", "dba1+bbox", "dba1+both") if v in res and v != fin]
    L += ["", "## Сравнение вариантов, F1 / TNR", "",
          f"| Галерея (кадров) | Сейчас: raw 0.8025 без постобработки | {sh}, `{pick}` (итог) | {sh}, "
          f"`{other}`, один порог | {sh}, raw thr(N) | "
          + " | ".join(f"{res[v]['label']}, `{pick}`" for v in extra) + " |",
          "|---|---:|---:|---:|---:|" + "---:|" * len(extra)]
    ft = lambda v: f"{f3(v['F1']['mean'])} / {f3(v['TNR']['mean'])}"
    for k in keys:
        L.append(f"| {name(k)} | {ft(cur[k])} | **{ft(R['sizes'][k])}** | {ft(res[fin][other]['sizes'][k])} | "
                 f"{ft(tn['sizes'][k])} | " + " | ".join(ft(res[v][pick]["sizes"][k]) for v in extra) + " |")
    L += ["", "## Воспроизвести", "",
          "```bash",
          f"python research/refusal_mode.py --data-dir <данные> --env REID_WEIGHTS=<веса a10> --tag {tag} --zip dataset.zip",
          "```", "",
          f"Пишет `research/honest_eval_results/refusal_mode_{tag}.json` / `.md` (все таблицы, ± по фолдам) и этот "
          "файл. Значения для `REFUSAL_PRESETS` — в JSON, ключ `presets`. На готовых эмбеддингах — около минуты; первый "
          "прогон считает эмбеддинги моделью. Отдельный разбор «порог от N» — `research/threshold_vs_gallery.py --score "
          "service --mode neardup` (`threshold_vs_gallery_a10_service.md`).", "",
          "## Оговорки", "",
          "- 150 машин holdout × 5 фолдов. Запросов без пары около 146 на все фолды, шаг TNR — около 0.7 %. Разброс по "
          "фолдам — в JSON.",
          "- Галерея команды 8996 не включена: в ней 7 запросов без пары, TNR там не показатель.",
          "- Протокол «как сервис» предполагает, что в закрытой галерее есть кадры той же машины с той же камеры. Если "
          "их нет, картина ближе к режиму без почти-дублей (`honest_eval_a10_suite.md`), и порог надо брать оттуда.",
          f"- Штраф: τ = {res['tau']:.5f} по camera_id обучающих машин, top-{res['topk']} кандидатов, признак фона — "
          "`research/honest_eval_results/cheap_bg_a10.md`.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--zip", type=Path, default=None, help="dataset.zip, если признаков фона нет в кэше")
    ap.add_argument("--holdout", type=Path, default=HERE / "val_ids.txt")
    ap.add_argument("--exclude-frames", type=Path, default=HERE / "honest_eval_results" / "holdout_frame_dups.txt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--openset-frac", type=float, default=0.25)
    ap.add_argument("--profile", choices=se.PROFILES, default="env_example")
    ap.add_argument("--compose-ref", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--source", choices=("crops", "frames"), default="crops")
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--one-query-per-camera", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=se.DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=HERE / "honest_eval_results")
    ap.add_argument("--report", action="store_true", help="только собрать .md из JSON")
    ap.add_argument("--docs", type=Path, default=HERE.parent / "docs" / "refusal.md")
    a = ap.parse_args()
    stem = a.out_dir / f"refusal_mode_{a.tag}"
    if not a.report:
        D = he.Data(a)
        res = run(D, a, he.folds_by_vehicle(sorted(D.hold), a.folds, a.seed))
        res["presets"] = presets(res)
        res["final"] = FINAL
        w = {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~") for k, v in D.env.items()}
        res["weights"] = w.get("REID_WEIGHTS", "")
        stem.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    res = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    stem.with_suffix(".md").write_text(markdown(res), encoding="utf-8")
    a.docs.write_text(docs_md(res), encoding="utf-8")
    print(f"-> {stem}.md, {a.docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
