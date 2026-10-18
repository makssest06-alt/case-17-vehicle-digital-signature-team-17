#!/usr/bin/env python3
"""Постобработка и отказ вместе: перебор ранжирований на 361 / 1111 по протоколу Андрея
(head_to_head.py, как research/closed_gallery.py): 4 ротации по 25 % машин без пары, rng(42), 2 фолда.

- Ранжирование — только то, что сервис посчитает по одному запросу и статичной галерее: raw; DBA k=1/2/3
  (веса cos); k-reciprocal только по галерее (граф соседей галереи, без других запросов —
  online_gallery_rerank.py Андрея); hubness Андрея (k=5, β=0.5 — head_to_head.py); DBA + фон;
  DBA + hubness (combo_check.py Андрея: hubness по галерее после DBA); DBA + hubness + фон; hubness + фон.
- Отказ у всех один: оценка по top-100 без почти-дублей «фон ИЛИ рамка» (наши пороги), семантика сервиса;
  скор z100 и raw по баллам самого ранжирования; порог заново для каждого варианта так же, как выбирался
  пресет итога PR #2 (closed_gallery.py): на фолде выбора каждой ротации — максимум среднего F1 при TNR ≥ 0.70
  на 361 и на 1111, медиана 8 порогов; F1 / TNR — при ней. Справочно — порог вне фолда по каждой галерее
  отдельно (closed_gallery.evaluate_refusal).
- Время — постобработка одного запроса на галерее 1111 (numpy, медиана по запросам) без извлечения
  эмбеддинга и поиска; то, что строится по галерее заранее (DBA, hubness, граф соседей), — отдельно.

    python research/joint_pp_refusal.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10 --zip dataset.zip
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "api"))
import bigallery_pp as bp  # noqa: E402
import closed_gallery as cg  # noqa: E402
import honest_eval as he  # noqa: E402
import official_evaluate as oe  # noqa: E402
import threshold_vs_gallery as tvg  # noqa: E402
from app.pp import dedup as dd  # noqa: E402
from app.pp import rerank as rr  # noqa: E402
from app.pp.dba import dba  # noqa: E402

GALLERIES = cg.MAIN
REFUSAL = "bg|dup_ours"                                   # почти-дубли — только в отказе
FINAL = "DBA k=1 + фон (итог PR #2)"
HUB_K, HUB_BETA = 5, 0.5
RECIP_GRID = [(k, m) for k in (5, 10, 20) for m in (0.5, 0.7, 0.85, 0.95)]   # как online_gallery_rerank.py
MAP_GAIN, F1_GAIN = 0.01, 0.02                            # порог «лучше» на 1111 для нового пресета


def hub(g: np.ndarray, k: int = HUB_K) -> np.ndarray:
    """Hubness кадра галереи — средний косинус к k ближайшим соседям по галерее (head_to_head.hub)."""
    S = g @ g.T
    np.fill_diagonal(S, -np.inf)
    return np.partition(S, -k, axis=1)[:, -k:].mean(1).astype("float32")


def topk_rows(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    val = np.take_along_axis(scores, idx, axis=1)
    return np.take_along_axis(idx, np.argsort(-val, axis=1), axis=1)


def gallery_graph(g: np.ndarray, k: int) -> np.ndarray:
    """k соседей каждого кадра по одной галерее (online_gallery_rerank.gallery_graph)."""
    S = g @ g.T
    np.fill_diagonal(S, -np.inf)
    return topk_rows(S, min(k, len(g) - 1))


def reciprocal(raw_row: np.ndarray, graph: np.ndarray, k: int, mix: float) -> np.ndarray:
    """Балл одного запроса: mix·косинус + (1 − mix)·Jaccard соседей (online_gallery_rerank.reciprocal_scores)."""
    nb = topk_rows(raw_row[None, :], k)[0]
    overlap = np.isin(graph, nb).sum(axis=1)
    return (mix * raw_row + (1.0 - mix) * overlap / np.maximum(2.0 * k - overlap, 1.0)).astype("float32")


def rank_metrics(s, fl, Q, G, gids, qids, mask=None) -> dict:
    order = np.argsort(-(s - cg.PUSH * fl), axis=1, kind="stable")[:, :10]
    take = range(len(qids)) if mask is None else np.flatnonzero(mask)
    sub = Q if mask is None else Q[mask]
    rk = oe.ranking_metrics(sub, G, {qids[i]: [gids[j] for j in order[i]] for i in take}, top_k=10)
    return {"mAP@10": rk["mAP@10"], "R1": rk["Rank-1"], "R5": rk["Rank-5"]}


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
    rng = np.random.default_rng(42)                                   # как head_to_head.py / closed_gallery.py
    rot_of = {v: n % 4 for n, v in enumerate(rng.permutation(vehicles))}
    fold_of = {v: n % 2 for n, v in enumerate(rng.permutation(vehicles))}
    keep_cam = {v: Counter(qc[qv == v]).most_common(1)[0][0] for v in vehicles}
    qfold = np.array([fold_of[v] for v in qv])
    extra = {"361": [], "1111": D.valid(D.distractors("test_gallery"))}
    res = {"tag": a.tag, "tau": tau, "hub": {"k": HUB_K, "beta": HUB_BETA}, "galleries": {}, "timing_ms": {}}
    ctx = {}
    for gname in GALLERIES:
        G = he.frame(grows + extra[gname])
        gids = list(G.index)
        g = D.vectors(gids)
        gbg = np.stack([bg[i] for i in gids])
        gbox = np.array([boxes[i] for i in gids], dtype="float64")
        ctx[gname] = {"G": G, "gids": gids, "g": g, "gbg": gbg, "gbox": gbox, "raw": q @ g.T,
                      "gv": G.vehicle_id.to_numpy(), "gc": G.camera_id.to_numpy()}

    # k-reciprocal: (k, mix) по mAP@10 на 1111 на половине машин (фолд 0), проверка на другой
    c = ctx["1111"]
    graphs = {k: gallery_graph(c["g"], k) for k, _ in RECIP_GRID}
    none = np.zeros(c["raw"].shape, bool)
    sel = []
    for k, mix in RECIP_GRID:
        s = np.stack([reciprocal(c["raw"][i], graphs[k], k, mix) for i in range(len(qids))])
        tune = rank_metrics(s, none, Q, c["G"], c["gids"], qids, qfold == 0)["mAP@10"]
        test = rank_metrics(s, none, Q, c["G"], c["gids"], qids, qfold == 1)["mAP@10"]
        sel.append({"k": k, "mix": mix, "tune_mAP": tune, "test_mAP": test})
    best = max(sel, key=lambda r: r["tune_mAP"])
    RK, RMIX = best["k"], best["mix"]
    res["reciprocal_selection"] = {"grid": sel, "selected": best}
    print(f"k-reciprocal: выбран k={RK} mix={RMIX} (фолд выбора {best['tune_mAP']:.4f}, "
          f"другой {best['test_mAP']:.4f})", flush=True)
    recip = f"k-reciprocal по галерее (k={RK}, mix={RMIX})"
    hubv = f"hubness (k={HUB_K}, β={HUB_BETA})"
    # (имя, функция баллов по контексту галереи, флаги ранжирования)
    variants = [
        ("raw", lambda c: c["raw"], "none"),
        ("DBA k=1", lambda c: q @ c["dba1"].T, "none"),
        ("DBA k=2", lambda c: q @ dba(c["g"], k=2, weights="cos").T, "none"),
        ("DBA k=3", lambda c: q @ dba(c["g"], k=3, weights="cos").T, "none"),
        (recip, lambda c: np.stack([reciprocal(c["raw"][i], c["graph"], RK, RMIX) for i in range(len(qids))]), "none"),
        (hubv, lambda c: c["raw"] - HUB_BETA * c["hub"][None, :], "none"),
        (f"{hubv} + фон", lambda c: c["raw"] - HUB_BETA * c["hub"][None, :], "bg"),
        (FINAL, lambda c: q @ c["dba1"].T, "bg"),
        (f"DBA k=1 + {hubv}", lambda c: q @ c["dba1"].T - HUB_BETA * c["hub_dba"][None, :], "none"),
        (f"DBA k=1 + {hubv} + фон", lambda c: q @ c["dba1"].T - HUB_BETA * c["hub_dba"][None, :], "bg"),
        (f"DBA k=1 + hubness (k={HUB_K}, β=1, combo_check)", lambda c: q @ c["dba1"].T - c["hub_dba"][None, :], "none"),
    ]
    res["variants"] = [v[0] for v in variants]
    for gname in GALLERIES:
        c = ctx[gname]
        c["dba1"] = dba(c["g"], k=1, weights="cos")
        c["hub"], c["hub_dba"] = hub(c["g"]), hub(c["dba1"])
        c["graph"] = gallery_graph(c["g"], RK)
        gv, gc = c["gv"], c["gc"]
        fl = {"none": np.zeros(c["raw"].shape, bool), "bg": (qbg @ c["gbg"].T) > tau}
        fl["dup_ours"] = dd.near_dup_flags(c["raw"], qbox[:, None, :], c["gbox"][None, :, :], **cg.RULES["ours"])
        fl[REFUSAL] = fl["bg"] | fl["dup_ours"]
        keeps = [np.array([not (rot_of.get(v) == r and cc != keep_cam.get(v)) for v, cc in zip(gv, gc)])
                 for r in range(4)]
        out = {"N": len(c["gids"])}
        for vname, fn, fr in variants:
            s = fn(c)
            v_out = rank_metrics(s, fl[fr], Q, c["G"], c["gids"], qids)
            recs = [cg.rotation_records(s, fl[fr], fl[REFUSAL], kp, qv, qc, gv, gc, "service") for kp in keeps]
            for key, formula in (("score", "z100"), ("score_raw", "raw")):
                v_out[formula] = cg.evaluate_refusal(recs, qfold, key=key)
            ctx.setdefault("recs", {})[(gname, vname)] = recs
            out[vname] = v_out
            print(f"{gname:5s} {vname:46s} mAP {v_out['mAP@10']:.4f} R1 {v_out['R1']:.3f} | z100 F1/TNR "
                  f"{v_out['z100']['F1']:.3f}/{v_out['z100']['TNR']:.3f} | raw {v_out['raw']['F1']:.3f}/"
                  f"{v_out['raw']['TNR']:.3f}", flush=True)
        res["galleries"][gname] = out
    res["preset"] = {f"{v} | {formula}": preset({g: ctx["recs"][(g, v)] for g in GALLERIES}, qfold, key)
                     for v, _, _ in variants for key, formula in (("score", "z100"), ("score_raw", "raw"))}
    res["timing_ms"] = timing(ctx["1111"], q, qbg, qbox, tau, variants, RK, RMIX)
    return res


def preset(recs_by_g: dict, qfold, key: str) -> dict:
    """Порог как у пресета (closed_gallery.run): по каждой ротации и фолду выбора — choose_joint по галереям,
    медиана; F1 / TNR каждой галереи при ней."""
    thr, strict = {}, {}
    for r_i in range(4):
        for f in (0, 1):
            tune = qfold != f
            parts = [(recs[r_i][key][tune], recs[r_i]["has"][tune], recs[r_i]["correct"][tune])
                     for recs in recs_by_g.values()]
            thr[(r_i, f)] = cg.choose_joint(parts)
            strict[(r_i, f)] = cg.choose_joint(parts, cg.STRICT_TNR, True)
    t_med, s_med = float(np.median(list(thr.values()))), float(np.median(list(strict.values())))
    fixed = {**{k: t_med for k in thr}, **{("strict",) + k: s_med for k in strict}}
    per_g = {g: cg.evaluate_refusal(recs, qfold, fixed, key=key) for g, recs in recs_by_g.items()}
    return {"threshold": t_med, "strict": s_med, "per_gallery": {g: {k: per_g[g][k] for k in ("F1", "TNR", "PR-AUC")}
                                                                 for g in per_g}}


def timing(c, q, qbg, qbox, tau, variants, rk, rmix) -> dict:
    """Постобработка одного запроса на 1111: баллы, флаги, top-10 и оценка отказа (z100). Медиана, мс."""
    gT, dT, gbgT = c["g"].T.copy(), c["dba1"].T.copy(), c["gbg"].T.copy()
    per_query = {
        "raw": lambda i: q[i] @ gT,
        "DBA": lambda i: q[i] @ dT,
        "reciprocal": lambda i: reciprocal(q[i] @ gT, c["graph"], rk, rmix),
        "hub": lambda i: q[i] @ gT - HUB_BETA * c["hub"],
        "DBA+hub": lambda i: q[i] @ dT - HUB_BETA * c["hub_dba"],
    }
    kind = lambda name: ("reciprocal" if "reciprocal" in name else "DBA+hub" if name.startswith("DBA") and "hub" in name
                         else "DBA" if name.startswith("DBA") else "hub" if "hub" in name else "raw")
    out = {}
    for vname, _, fr in variants:
        f = per_query[kind(vname)]
        ts = []
        for i in range(len(q)):
            t0 = time.perf_counter()
            s = f(i)
            raw_row = q[i] @ gT
            fbg = (qbg[i] @ gbgT) > tau
            fx = dd.near_dup_flags(raw_row, qbox[i][None, :], c["gbox"], **cg.RULES["ours"])
            fref = fbg | fx
            frank = fbg if fr == "bg" else np.zeros_like(fbg)
            top10 = np.argsort(-(s - cg.PUSH * frank), kind="stable")[:10]
            o = np.argsort(-s, kind="stable")[:cg.Z_TOP]
            rr.refusal_score(s[o], fref[o], "z100")
            ts.append(time.perf_counter() - t0)
            assert len(top10) == 10
        out[vname] = float(np.median(ts) * 1e3)
    t0 = time.perf_counter(); dba(c["g"], k=1, weights="cos"); t_dba = time.perf_counter() - t0
    t0 = time.perf_counter(); hub(c["g"]); t_hub = time.perf_counter() - t0
    t0 = time.perf_counter(); gallery_graph(c["g"], rk); t_graph = time.perf_counter() - t0
    out["_gallery_build_ms"] = {"DBA": t_dba * 1e3, "hubness": t_hub * 1e3, "граф соседей": t_graph * 1e3}
    return out


def verdict(res: dict) -> dict:
    """По порогу пресета: «не хуже» — mAP@10 и F1 на 1111 не ниже итога, TNR ≥ 0.70 на 361 и 1111."""
    G1 = res["galleries"]["1111"]
    pf = lambda v, f, g: res["preset"][f"{v} | {f}"]["per_gallery"][g]
    rows = [(v, f) for v in res["variants"] for f in ("z100", "raw")]
    ok_tnr = [(v, f) for v, f in rows if all(pf(v, f, g)["TNR"] >= cg.MIN_TNR for g in GALLERIES)]
    base_map, base_f1 = G1[FINAL]["mAP@10"], pf(FINAL, "z100", "1111")["F1"]
    best_map = max(res["variants"], key=lambda v: G1[v]["mAP@10"])
    best_f1 = max(ok_tnr, key=lambda vf: pf(*vf, "1111")["F1"])
    not_worse = [(v, f) for v, f in ok_tnr if (v, f) != (FINAL, "z100")
                 and G1[v]["mAP@10"] >= base_map and pf(v, f, "1111")["F1"] >= base_f1]
    better = [(v, f) for v, f in not_worse if G1[v]["mAP@10"] - base_map >= MAP_GAIN
              or pf(v, f, "1111")["F1"] - base_f1 >= F1_GAIN]
    return {"best_mAP_1111": best_map, "best_F1_1111": list(best_f1), "not_worse": [list(x) for x in not_worse],
            "better_enough": [list(x) for x in better]}


def report_md(res: dict) -> str:
    G, V = res["galleries"], verdict(res)
    PF = lambda vf, g: res["preset"][f"{vf[0]} | {vf[1]}"]["per_gallery"][g]
    f3 = lambda x: f"{x:.3f}"
    sel = res["reciprocal_selection"]["selected"]
    L = [f"# Постобработка и отказ вместе: {res['tag']}", "",
         "Протокол Андрея `head_to_head.py` (как `research/closed_gallery.py`): 546 запросов, галереи 361 / 1111, "
         "4 ротации по 25 % машин без пары, rng(42), 2 фолда по машинам; почти-дубли «фон ИЛИ рамка» (наши пороги) "
         "исключаются только в отказе; порог — заново для каждого варианта и скора. Воспроизвести: `python research/joint_pp_refusal.py --data-dir <данные> --env "
         f"REID_WEIGHTS=<веса> --tag {res['tag']} --zip dataset.zip`.", "",
         f"k-reciprocal — только по галерее; (k, mix) выбраны по mAP@10 на 1111 на половине машин: k={sel['k']}, "
         f"mix={sel['mix']} (на фолде выбора {sel['tune_mAP']:.4f}, на другом {sel['test_mAP']:.4f}). Hubness — "
         f"k={res['hub']['k']}, β={res['hub']['beta']} (`head_to_head.py`), у DBA + hubness — по галерее после DBA "
         "(`combo_check.py`). Отказ считается по баллам самого ранжирования.", "",
         "F1 / TNR — при пороге, выбранном как пресет итога (медиана по ротациям и фолдам, TNR ≥ 0.70 на 361 и "
         "на 1111); «вне фолда» — порог по каждой галерее отдельно, справочно.", "",
         "| Ранжирование | Скор отказа | mAP@10 361 | mAP@10 1111 | R1 361 | R1 1111 | Порог | F1 / TNR 361 | "
         "F1 / TNR 1111 | вне фолда F1 / TNR 1111 | мс на запрос |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    bf = tuple(V["best_F1_1111"])
    for v in res["variants"]:
        for formula in ("z100", "raw"):
            a, b = G["361"][v], G["1111"][v]
            mark = (" **[лучший mAP 1111]**" if v == V["best_mAP_1111"] and formula == "z100" else "") + \
                   (" **[лучший F1 1111]**" if (v, formula) == bf else "") + \
                   (" — итог PR #2" if (v, formula) == (FINAL, "z100") else "")
            p = res["preset"][f"{v} | {formula}"]
            pa, pb = p["per_gallery"]["361"], p["per_gallery"]["1111"]
            L.append(f"| {v}{mark} | {formula} | {a['mAP@10']:.4f} | {b['mAP@10']:.4f} | {f3(a['R1'])} | {f3(b['R1'])} | "
                     f"{p['threshold']:.4f} | {f3(pa['F1'])} / {f3(pa['TNR'])} | {f3(pb['F1'])} / {f3(pb['TNR'])} | "
                     f"{f3(b[formula]['F1'])} / {f3(b[formula]['TNR'])} | {res['timing_ms'][v]:.2f} |")
    gb = res["timing_ms"]["_gallery_build_ms"]
    L += ["", "Время — постобработка одного запроса на 1111 (numpy на этой машине, без эмбеддинга и поиска). "
          "Строится по галерее заранее и при каждом изменении галереи: " +
          ", ".join(f"{k} {v:.0f} мс" for k, v in gb.items()) + ".", "",
          "## Вывод", "",
          f"- Лучший по mAP@10 на 1111: **{V['best_mAP_1111']}** ({G['1111'][V['best_mAP_1111']]['mAP@10']:.4f}; "
          f"итог {G['1111'][FINAL]['mAP@10']:.4f}).",
          f"- Лучший по F1 на 1111 среди вариантов с TNR ≥ 0.70 на обеих галереях: **{bf[0]}, {bf[1]}** "
          f"({f3(PF(bf, '1111')['F1'])} / {f3(PF(bf, '1111')['TNR'])}; итог "
          f"{f3(PF((FINAL, 'z100'), '1111')['F1'])} / {f3(PF((FINAL, 'z100'), '1111')['TNR'])}).",
          "- Не хуже итога PR #2 и по mAP@10, и по F1 на 1111 (TNR ≥ 0.70 на обеих): " +
          ("; ".join(f"{v}, {f}" for v, f in V["not_worse"]) if V["not_worse"] else "**нет**") + ".",
          f"- Из них лучше на 1111 хотя бы на {MAP_GAIN} mAP или {F1_GAIN} F1: " +
          ("; ".join(f"{v}, {f}" for v, f in V["better_enough"]) if V["better_enough"] else "**нет**") + "."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = bp.data_args(argparse.ArgumentParser())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    out = a.out_dir / f"joint_pp_refusal_{a.tag}.json"
    if not a.report:
        res = run(a)
        res["verdict"] = verdict(res)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    res = json.loads(out.read_text(encoding="utf-8"))
    (a.out_dir / f"joint_pp_refusal_{a.tag}.md").write_text(report_md(res), encoding="utf-8")
    print(f"-> {a.out_dir / f'joint_pp_refusal_{a.tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
