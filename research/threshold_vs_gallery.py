#!/usr/bin/env python3
"""Порог отказа как функция размера галереи: thr(N) = a + b·ln N.

Галерея фолда = кадры галереи holdout (без машин «без пары») + случайная доля
чужих кадров (весь тест и кадры обучающих машин), N — по сетке. Для каждого N
и фолда выбирается порог: максимум F1 при TNR ≥ 0.70 (как в
research/refusal_score.py), режим «без почти-дублей», метрики эталона. Правило
подбирается на фолде f (по всем N сразу) и проверяется на остальных четырёх —
на сетке и на именованных галереях 2.2k / 9.0k / 10.8k. Рядом — фиксированный
порог 0.70 и z100 (порог выбран на 2.2k).

Форма: у запроса без пары top-1 — максимум из N похожестей на чужие машины.
Для хвоста экспоненциального типа (домен Гумбеля) этот максимум растёт как
a + b·ln N, для гауссова — как a + b·√(2 ln N). Обе формы подгоняются, выбор —
по ошибке на фолдах выбора; рост максимума у запросов без пары показан отдельно.

Любой скор: встроенный (--score raw | dba1) или функция
`--score-fn файл.py:функция`, функция(q, g, Q, G) → матрица скоров (запрос ×
кадр галереи, больше — увереннее), q, g — L2-векторы, Q, G — pandas с
vehicle_id / camera_id по image_id. Почти-дубли маскируются после скора.

`--score service` — скор сервиса с постобработкой тем же кодом, что api/app/pp
(DBA k=--pp-dba-k, штраф за ту же камеру по признаку фона, оценка отказа
rerank.refusal_score), протокол --mode: nodup или neardup («как сервис»: у
запроса без пары в галерее остаётся кадр его машины с той же камеры).

    python research/threshold_vs_gallery.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10
    python research/threshold_vs_gallery.py ... --score-fn andrey_pp.py:rerank --tag a10_andrey
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "api"))          # app.pp — код сервиса
import calibrate_refusal_folds as crf  # noqa: E402
import honest_eval as he  # noqa: E402
import refusal_score as rsc  # noqa: E402
import service_embeddings as se  # noqa: E402

GRID = (500, 750, 1000, 1500, 2203, 3000, 5000, 8000, 10847)
NAMED = (("2.2k", "test_all"), ("9.0k", "train_seen"), ("10.8k", "test_all+train_seen"))
TNR_FLOOR = 0.70
FIXED = 0.70
FORMS = {"ln": lambda n: np.log(n), "sqrt2ln": lambda n: np.sqrt(2 * np.log(n))}
SURFACE, INK, INK2, MUTED, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FONT = "PT Sans"


def builtin(name: str):
    if name == "raw":
        return lambda q, g, Q, G: q @ g.T
    if name.startswith("dba"):
        import bigallery_pp as bp
        k = int(name[3:])

        def f(q, g, Q, G):
            b = {"qv": q, "gv": g, "std": np.ones(len(g), bool),
                 "same_v": G.vehicle_id.to_numpy()[None, :] == Q.vehicle_id.to_numpy()[:, None],
                 "same_c": G.camera_id.to_numpy()[None, :] == Q.camera_id.to_numpy()[:, None]}
            return bp.dba_scores(b, k, clean=True)
        return f
    raise SystemExit(f"неизвестный --score {name}")


def plugin(spec: str):
    path, _, fn = spec.rpartition(":")
    sp = importlib.util.spec_from_file_location("score_plugin", path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return getattr(mod, fn)


# ---------------------------------------------------------------------------
# скор сервиса: тот же код, что api/app/pp
# ---------------------------------------------------------------------------

def service_bg(D, a) -> dict:
    """Признак фона сервиса (градиенты 64×36 + гистограммы) для всех кадров — из кэша cheap_bg."""
    import cheap_bg as cb
    import pseudo_camera as pc

    rows = pc.read_rows(a.data_dir)
    parts = cb.descriptors(rows, a.zip, a.cache_dir / "cheap_bg_desc.npz", 4)
    return cb.kind_bg(parts, [r["image_id"] for r in rows], "+".join(cb.bgm.SERVICE_PARTS))


def records_service(D, q_rows, g_rows, opened: set, bg: dict, conf: str, sp: dict, extra: bool = False) -> dict:
    """Записи отказа для скора сервиса. conf: s1 → оценка raw, z100 → z100 (обе — в conf_raw / conf_z100).
    Ранжирование и оценка — research/service_score.rank_queries (код сервиса, точный top-K).
    extra — ещё mAP@10, Rank-1, Rank-5 и mINP ранжирования после постобработки (для research/refusal_mode.py)."""
    import bigallery_pp as bp
    import service_score as ss
    from app.pp import rerank as rr

    from app.pp import dedup as dd

    b = bp.make_block(D, q_rows, g_rows, opened, bg)
    mask = b["mask"][sp["mode"]]
    raw = b["qv"] @ b["gv"].T
    s = bp.dba_scores(b, sp["dba_k"], clean=sp["mode"] == "nodup") if sp["dba_k"] else raw.copy()
    s = np.where(mask, -np.inf, s)
    dedup = sp.get("dedup") or ("bg" if sp.get("camera") else "off")
    top, cos = ss.topk(s, rr.needed_k(10, dedup != "off", sp["topk"], "z100"))
    fin = np.isfinite(cos)
    flags = np.zeros(cos.shape, bool)
    if dedup in ("bg", "both"):
        flags |= (ss.pair_cos(b["qbg"], b["gbg"], top) > sp["tau"]) & fin
    if dedup in ("bbox", "both"):
        qb = np.array([sp["boxes"][i] for i in b["Q"].index], dtype="float64")
        gb = np.array([sp["boxes"][i] for i in b["G"].index], dtype="float64")
        flags |= dd.near_dup_flags(np.take_along_axis(raw, top, 1), qb[:, None, :], gb[top], **sp["rule"]) & fin
    r = ss.decide(top, cos, flags)
    n = len(s)
    top1 = np.array([o[0] if len(o) else 0 for o in r["order"]])
    rows_ = np.arange(n)
    rec = {"conf": r["conf_raw"] if conf == "s1" else r["conf_z100"], "conf_raw": r["conf_raw"],
           "conf_z100": r["conf_z100"], "has_match": (b["same_v"] & ~b["same_c"] & ~mask).any(1),
           "correct": b["same_v"][rows_, top1], "s1": r["conf_raw"]}
    if extra:
        pen = s.copy()
        for i in range(n):
            pen[i, r["order"][i][r["flags"][i]]] -= bp.BIG       # штраф виден и в полном ранжировании
        ev = bp.evaluate(b, pen, sp["mode"])
        rec["mAP@10"], rec["Rank-1"], rec["Rank-5"] = ev["mAP@10"], ev["Rank-1"], ev["Rank-5"]
        rec["mINP"] = minp(b, pen, sp["mode"])
    return rec


def minp(b: dict, s: np.ndarray, mode: str) -> float:
    """mINP по полному ранжированию (логика honest_eval.full_metrics): почти-дубли и
    скрытые протоколом кадры не участвуют, позитив — та же машина с другой камеры."""
    junk = b["same_v"] & b["same_c"]
    out = []
    for i in range(len(s)):
        keep = ~junk[i] & ~b["mask"][mode][i] & np.isfinite(s[i])
        rel = b["same_v"][i][keep][np.argsort(-s[i][keep], kind="stable")]
        if rel.any():
            out.append(int(rel.sum()) / (int(np.max(np.nonzero(rel)[0])) + 1))
    return float(np.mean(out)) if out else float("nan")


def service_params(D, a) -> dict:
    """Параметры скора сервиса: τ и top-K — значения по умолчанию из api/app/config.py."""
    from app.config import Settings

    st = Settings()
    return {"mode": a.mode, "dba_k": a.pp_dba_k, "dedup": a.pp_dedup or ("bg" if a.pp_camera == "on" else "off"),
            "tau": st.pp_camera_tau, "topk": st.pp_camera_topk, "bg": service_bg(D, a), **service_rule(a)}


def service_rule(a) -> dict:
    """Правило почти-дубля сервиса (PP_DEDUP_* из api/app/config.py) и рамки всех кадров."""
    import pseudo_camera as pc
    from app.config import Settings

    st = Settings()
    return {"rule": {"s_hi": st.pp_dedup_sim, "t_iou": st.pp_dedup_iou, "t_center": st.pp_dedup_center,
                     "t_size": st.pp_dedup_size},
            "boxes": {r["image_id"]: r["box"] for r in pc.read_rows(a.data_dir)}}


def records(D, q_rows, g_rows, score_fn, conf: str) -> dict:
    Q, G = he.frame(q_rows), he.frame(g_rows)
    s = np.asarray(score_fn(D.vectors(list(Q.index)), D.vectors(list(G.index)), Q, G), dtype="float64")
    mask = he.mode_mask(Q, G, "nodup")
    s = np.where(mask, -np.inf, s)
    top = rsc.top_desc(s, 100)
    top1 = s.argmax(1)
    q_vid, g_vid = Q.vehicle_id.to_numpy(), G.vehicle_id.to_numpy()
    has = ((g_vid[None, :] == q_vid[:, None]) & ~mask).any(1)
    c = top[:, 0] if conf == "s1" else rsc.z_of(top)
    return {"conf": c, "has_match": has, "correct": g_vid[top1] == q_vid, "s1": top[:, 0]}


def build(D, score_fn, conf: str, folds: list, service: dict = None) -> dict:
    """Записи по фолдам: сетка N и именованные галереи. Open-set — как в honest_eval.
    service — параметры скора сервиса (records_service), тогда score_fn не нужен."""
    qry, gal = D.valid(D.qry), D.valid(D.gal)
    pool = D.valid(D.distractors("test_all+train_seen"))
    named = {name: D.valid(D.distractors(kind)) for name, kind in NAMED}
    rng = np.random.default_rng(D.a.seed + 1)
    out = {"grid": {}, "named": {name: [] for name in named}, "base": []}
    for f, fvids in enumerate(folds):
        fset = set(fvids)
        n_open = int(round(len(fvids) * D.a.openset_frac))
        opened = set(rng.choice(sorted(fvids, key=int), size=n_open, replace=False).tolist())
        q_rows = [r for r in qry if r["vehicle_id"] in fset]
        neardup = service is not None and service["mode"] == "neardup"
        base = gal if neardup else [r for r in gal if r["vehicle_id"] not in opened]    # «как сервис»: кадры машин без пары в галерее
        if service is not None:
            rec = lambda g_rows: records_service(D, q_rows, g_rows, opened, service["bg"], conf, service,
                                                 service.get("extra", False))
        else:
            rec = lambda g_rows: records(D, q_rows, g_rows, score_fn, conf)
        out["base"].append(len(base))
        sizes = [len(base)] + [n for n in GRID if n > len(base)]
        for n in sizes:
            take = min(n - len(base), len(pool))                # у самой большой галереи — весь пул
            sub = np.random.default_rng(1000 * n + f).choice(len(pool), size=take, replace=False) if take > 0 else []
            out["grid"].setdefault(n, []).append(rec(base + [pool[i] for i in sub]))
        for name, extra in named.items():
            out["named"][name].append(rec(base + extra))
    return out


def fit(ns, ts, form: str) -> tuple:
    x = FORMS[form](np.asarray(ns, dtype="float64"))
    b, a = np.polyfit(x, np.asarray(ts), 1)
    return float(a), float(b)


def evaluate(recs: list, thr_by_fold: list) -> dict:
    per = []
    for f, t in enumerate(thr_by_fold):
        rest = {k: np.concatenate([r[k] for i, r in enumerate(recs) if i != f]) for k in ("conf", "has_match", "correct")}
        per.append(crf.metrics(rest, t))
    return {k: rsc.stats([p[k] for p in per]) for k in ("F1", "TNR", "recall", "precision")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--score", default="raw", help="raw | dba1 .. dba5 | service")
    ap.add_argument("--mode", default="nodup", choices=("nodup", "neardup"), help="протокол галереи (для --score service)")
    ap.add_argument("--pp-dba-k", type=int, default=1, help="для --score service: 0 — без DBA")
    ap.add_argument("--pp-camera", default="on", choices=("on", "off"), help="для --score service: штраф за ту же камеру")
    ap.add_argument("--pp-dedup", default=None, choices=("off", "bbox", "bg", "both"),
                    help="для --score service: исключение почти-дублей (перекрывает --pp-camera)")
    ap.add_argument("--score-fn", default=None, help="файл.py:функция(q, g, Q, G) → скоры; заменяет --score")
    ap.add_argument("--conf", default="s1", choices=("s1", "z100"), help="оценка уверенности из строки скоров")
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
    a = ap.parse_args()

    D = he.Data(a)
    folds = he.folds_by_vehicle(sorted(D.hold), a.folds, a.seed)
    score_name = a.score_fn or a.score
    service = None
    if a.score == "service" and not a.score_fn:
        service = service_params(D, a)
        score_name = f"service (DBA k={a.pp_dba_k}, исключение {service['dedup']}, τ={service['tau']:.5f}, {a.mode})"
        score_fn = None
    else:
        score_fn = plugin(a.score_fn) if a.score_fn else builtin(a.score)
    main_ = build(D, score_fn, a.conf, folds, service)
    z = build(D, score_fn, "z100", folds, service) if a.conf == "s1" else main_
    ns = sorted(main_["grid"])
    common = [n for n in ns if len(main_["grid"][n]) == a.folds]          # N, посчитанные во всех фолдах

    # пороги по N: выбор на фолде f
    thr = {n: [rsc.choose_f1(main_["grid"][n][f], TNR_FLOOR) for f in range(a.folds)] for n in common}
    # форма и параметры — по фолду выбора
    fits = {form: [fit(common, [thr[n][f] for n in common], form) for f in range(a.folds)] for form in FORMS}
    rmse = {form: float(np.mean([np.sqrt(np.mean([(fits[form][f][0] + fits[form][f][1] * FORMS[form](n) - thr[n][f]) ** 2
                                                  for n in common])) for f in range(a.folds)])) for form in FORMS}
    form = min(rmse, key=rmse.get)
    rule = lambda f, n: fits[form][f][0] + fits[form][f][1] * float(FORMS[form](n))
    # z100: порог выбран на 2.2k (именованная галерея) на фолде f
    zthr = [rsc.choose_f1(z["named"]["2.2k"][f], TNR_FLOOR) for f in range(a.folds)]

    def compare(recs_by_n: dict, zrecs_by_n: dict, nsize: dict) -> dict:
        out = {}
        for key, recs in recs_by_n.items():
            n = nsize[key]
            out[str(key)] = {"N": n,
                             f"thr(N)": evaluate(recs, [rule(f, n) for f in range(a.folds)]),
                             "fixed": evaluate(recs, [FIXED] * a.folds),
                             "z100": evaluate(zrecs_by_n[key], zthr),
                             "rule_threshold": rsc.stats([rule(f, n) for f in range(a.folds)])}
        return out

    grid_cmp = compare({n: main_["grid"][n] for n in common}, {n: z["grid"][n] for n in common}, {n: n for n in common})
    nsize = {name: int(round(np.mean([main_["base"][f] + len(D.valid(D.distractors(kind))) for f in range(a.folds)])))
             for name, kind in NAMED}
    named_cmp = compare(main_["named"], z["named"], nsize)
    # обоснование формы: максимум похожести у запросов без пары от N
    open_max = {n: float(np.mean(np.concatenate([r["s1"][~r["has_match"]] for r in main_["grid"][n]]))) for n in common}
    om_fit = {f_: np.polyfit(FORMS[f_](np.array(common, float)), [open_max[n] for n in common], 1).tolist() for f_ in FORMS}
    om_r2 = {}
    for f_, (b_, a_) in om_fit.items():
        pred = a_ + b_ * FORMS[f_](np.array(common, float))
        y = np.array([open_max[n] for n in common])
        om_r2[f_] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    report = {"tag": a.tag, "score": score_name, "conf": a.conf, "mode": a.mode, "tnr_floor": TNR_FLOOR, "grid": common,
              "base_gallery": main_["base"], "thresholds_by_N": {str(n): rsc.stats(thr[n]) for n in common},
              "form": form, "rmse": rmse, "fits": {f_: fits[f_] for f_ in FORMS},
              "rule_mean": {"a": float(np.mean([p[0] for p in fits[form]])), "b": float(np.mean([p[1] for p in fits[form]]))},
              "z100_threshold": rsc.stats(zthr), "open_set_mean_s1": {str(n): v for n, v in open_max.items()},
              "open_set_max_fit_R2": om_r2, "grid_compare": grid_cmp, "named_compare": named_cmp}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"threshold_vs_gallery_{a.tag}"
    (a.out_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(report, a.out_dir / f"{stem}.png")
    (a.out_dir / f"{stem}.md").write_text(markdown(report, stem), encoding="utf-8")
    print(markdown(report, stem))
    return 0


def plot(r: dict, path: Path) -> None:
    ns = r["grid"]
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    fig.text(0.04, 0.94, "Порог отказа от размера галереи", fontsize=32, color=INK, family=FONT, weight="bold", va="top")
    a, b = r["rule_mean"]["a"], r["rule_mean"]["b"]
    form = "ln N" if r["form"] == "ln" else "√(2 ln N)"
    fig.text(0.04, 0.865, f"{r['tag']}, скор {r['score']}, без почти-дублей, 5 фолдов. Порог для каждого N: максимум F1 "
             f"при TNR ≥ {r['tnr_floor']:.2f} на фолде выбора.\nПравило thr(N) = {a:.3f} + {b:.4f}·{form} подобрано на "
             "фолде выбора и проверено на остальных.", fontsize=16, color=INK2, family=FONT, va="top", linespacing=1.4)
    ax = fig.add_axes([0.07, 0.24, 0.4, 0.46])
    style(ax, "Выбранный порог", "Кадров в галерее")
    m = [r["thresholds_by_N"][str(n)]["mean"] for n in ns]
    s = [r["thresholds_by_N"][str(n)]["std"] for n in ns]
    ax.errorbar(ns, m, yerr=s, fmt="o", color="#2a78d6", ms=9, capsize=5, lw=2, label="порог по N (± разброс фолдов)")
    xx = np.geomspace(min(ns), max(ns), 100)
    ax.plot(xx, a + b * FORMS[r["form"]](xx), color="#eb6834", lw=3, label=f"thr(N) = a + b·{form}")
    ax.axhline(FIXED, color=MUTED, lw=2, ls=(0, (5, 4)), label=f"фиксированный {FIXED:.2f}")
    ticks(ax, ns)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
              prop={"family": FONT, "size": 14}, labelcolor=INK2)
    ax2 = fig.add_axes([0.55, 0.24, 0.4, 0.46])
    style(ax2, "F1 и TNR на отложенных фолдах", "Кадров в галерее")
    for rule, color, label in (("thr(N)", "#eb6834", "thr(N)"), ("fixed", MUTED, f"{FIXED:.2f}"), ("z100", "#2a78d6", "z100")):
        g = r["grid_compare"]
        ax2.plot(ns, [g[str(n)][rule]["F1"]["mean"] for n in ns], color=color, lw=3, marker="o", label=f"F1, {label}")
        ax2.plot(ns, [g[str(n)][rule]["TNR"]["mean"] for n in ns], color=color, lw=2, ls=(0, (2, 2)), marker="s",
                 label=f"TNR, {label}")
    ticks(ax2, ns)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
               prop={"family": FONT, "size": 14}, labelcolor=INK2)
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def ticks(ax, ns):
    """Логарифмическая ось с делениями по сетке N обычными числами."""
    ax.set_xscale("log")
    shown = [n for n in ns if n in (500, 1000, 2203, 5000, 10847)]
    ax.set_xticks(shown)
    ax.set_xticklabels([f"{n:,}".replace(",", " ") for n in shown], fontsize=14, family=FONT, color=INK2)
    ax.minorticks_off()


def style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRIDC, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=14, length=0)
    ax.set_title(title, fontsize=21, color=INK, family=FONT, weight="bold", loc="left", pad=12)
    ax.set_xlabel(xlabel, fontsize=16, color=INK2, family=FONT)


def markdown(r: dict, stem: str) -> str:
    form = "ln N" if r["form"] == "ln" else "√(2 ln N)"
    cell = lambda d: f"{d['F1']['mean']:.3f} / {d['TNR']['mean']:.3f}"
    lines = [f"# Порог отказа от размера галереи: {r['tag']}", "",
             f"Скор `{r['score']}`, оценка уверенности `{r['conf']}`, "
             f"{'как сервис (почти-дубли в галерее)' if r.get('mode') == 'neardup' else 'режим без почти-дублей'}, 5 фолдов. Для каждого N "
             f"порог — максимум F1 при TNR ≥ {r['tnr_floor']:.2f} на фолде выбора; правило подобрано на фолде выбора, "
             "проверено на остальных четырёх. Галерея размера N — кадры holdout фолда плюс случайная доля всего "
             "теста и кадров обучающих машин. Картинка — `" + stem + ".png`.", "",
             f"**Форма:** ошибка подгонки по фолдам — ln N {r['rmse']['ln']:.4f}, √(2 ln N) {r['rmse']['sqrt2ln']:.4f}; "
             f"выбрана {form}. Средние параметры: thr(N) = {r['rule_mean']['a']:.4f} + {r['rule_mean']['b']:.4f}·{form}. "
             f"Средний top-1 у запросов без пары (максимум похожести на чужие машины) от N: R² {r['open_set_max_fit_R2']['ln']:.3f} "
             f"для ln N, {r['open_set_max_fit_R2']['sqrt2ln']:.3f} для √(2 ln N).", "",
             "| N | порог по N (выбор) | thr(N) | F1 / TNR: thr(N) | F1 / TNR: 0.70 | F1 / TNR: z100 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for n in r["grid"]:
        g = r["grid_compare"][str(n)]
        t = r["thresholds_by_N"][str(n)]
        lines.append(f"| {n} | {t['mean']:.3f} ± {t['std']:.3f} | {g['rule_threshold']['mean']:.3f} | {cell(g['thr(N)'])} | "
                     f"{cell(g['fixed'])} | {cell(g['z100'])} |")
    lines += ["", "Именованные галереи (правило и пороги не подбирались на них):", "",
              "| Галерея | N | thr(N) | F1 / TNR: thr(N) | F1 / TNR: 0.70 | F1 / TNR: z100 |", "|---|---:|---:|---:|---:|---:|"]
    for name, g in r["named_compare"].items():
        lines.append(f"| {name} | {g['N']} | {g['rule_threshold']['mean']:.3f} | {cell(g['thr(N)'])} | {cell(g['fixed'])} | "
                     f"{cell(g['z100'])} |")
    lines += ["", f"z100: порог {r['z100_threshold']['mean']:.3f} ± {r['z100_threshold']['std']:.3f}, выбран на 2.2k. "
              "Пересчитать для другого скора — та же команда с `--score-fn файл.py:функция`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
