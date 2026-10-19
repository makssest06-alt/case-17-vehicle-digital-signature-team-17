#!/usr/bin/env python3
"""Кадр той же машины с той же камеры на первом месте ответа: три трактовки, протокол Q17.

Тот же прогон, что research/closed_gallery.py --protocol q17 (галереи 361 и 1111 = 361 + 750 кадров обучающих машин,
5 ротаций, у 20 % машин в галерее нет ни одного кадра, порог вне фолда — максимум 0.7·F1 + 0.3·TNR по обеим,
медиана 10 порогов). Для вариантов «DBA» (отказ без исключения почти-дублей) и прежнего итога (отказ без
почти-дублей «фон ИЛИ рамка») верхний ответ, если это кадр той же машины с той же камеры (junk: тот же vehicle_id и
camera_id), трактуется тремя способами:
  a — засчитывается верным: так считают closed_gallery.py и эталон research/official_evaluate.py (candidate_metrics
      сверяет только vehicle_id верхнего кандидата);
  b — выкидывается до выбора верхнего: решение «ответить / отказать» — по оценке сервиса (её не меняем: сервис
      camera_id не знает), верность — по первому не-junk кандидату из top-10 ответа;
  c — считается ошибкой.
Порог в каждой трактовке выбирается заново тем же правилом; рядом — метрики при пресете варианта.

    python research/refusal_topcam.py --data-dir <данные> --env REID_WEIGHTS=<веса a10> --zip dataset.zip --tag a10_q17
    python research/refusal_topcam.py ... --no-model      # только из кэша эмбеддингов: при промахе — ошибка, не прогон

Пишет research/honest_eval_results/refusal_topcam_<tag>.json; таблицу из него берёт docs/refusal.md
(research/closed_gallery.py --report). Перед расчётом сверяет пресеты вариантов с closed_gallery_<tag>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "api"))
import closed_gallery as cg  # noqa: E402
import service_embeddings as se  # noqa: E402
from app.pp import rerank as rr  # noqa: E402

MODES = {"a": "засчитывается верным", "b": "выкидывается до выбора верхнего", "c": "считается ошибкой"}


def records(args, mode: str) -> dict:
    """Записи отказа одной ротации, как closed_gallery.rotation_records (z_mode service), с трактовкой junk."""
    s, fl_rank, fl_ref, keep, qv, qc, gv, gc, _ = args
    k = np.flatnonzero(keep)
    S, FR, FF = s[:, k], fl_rank[:, k], fl_ref[:, k]
    gvk, gck = gv[k], gc[k]
    junk = (gvk[None, :] == qv[:, None]) & (gck[None, :] == qc[:, None])
    has = ((gvk[None, :] == qv[:, None]) & (gck[None, :] != qc[:, None])).any(axis=1)
    order = np.argsort(-S, axis=1, kind="stable")[:, :cg.Z_TOP]
    score = np.array([rr.refusal_score(S[i, order[i]], FF[i, order[i]], "raw") for i in range(len(S))])
    top1 = np.argmax(S - cg.PUSH * FR, axis=1)
    top10 = np.argsort(-(S - cg.PUSH * FR), axis=1, kind="stable")[:, :10]
    jt = junk[np.arange(len(S)), top1]
    same = gvk[top1] == qv
    none_left = np.zeros(len(S), bool)
    if mode == "a":
        correct = same
    elif mode == "c":
        correct = same & ~jt
    else:
        correct = np.zeros(len(S), bool)
        for i in range(len(S)):
            nj = [j for j in top10[i] if not junk[i, j]]
            none_left[i] = not nj
            correct[i] = bool(nj) and gvk[nj[0]] == qv[i]
    return {"score": score, "score_raw": score, "has": has, "correct": correct, "junk_top1": jt,
            "none_left": none_left, "top1": k[top1]}


def main() -> int:
    ap = cg.bp.data_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-model", action="store_true", help="только кэш эмбеддингов; при промахе — ошибка")
    a = ap.parse_args()
    if a.no_model:
        run = se.subprocess.run

        def guard(cmd, *x, **kw):
            if any("python" in str(c).lower() for c in (cmd if isinstance(cmd, (list, tuple)) else [cmd])):
                raise SystemExit(f"--no-model: промах кэша эмбеддингов, нужен прогон модели ({cmd})")
            return run(cmd, *x, **kw)
        se.subprocess.run = guard

    calls, qfolds = [], []
    rot, ev = cg.rotation_records, cg.evaluate_refusal
    cg.rotation_records = lambda *x: calls.append(x) or rot(*x)
    cg.evaluate_refusal = lambda recs, qfold, *x, **kw: (qfolds.append(qfold) if not qfolds else None) or ev(recs, qfold, *x, **kw)
    cg.set_protocol("q17")
    res = cg.run(a)
    qfold = qfolds[0]

    names = [v[0] for v in cg.VARIANTS]
    it = iter(calls)
    by = {(g, v, z, r): next(it) for g in cg.GALLERIES for v in names for z in ("andrey", "service")
          for r in range(cg.ROTATIONS)}
    variants = ("DBA", cg.FINAL)
    pub_path = a.out_dir / f"closed_gallery_{a.tag}.json"
    pub = json.loads(pub_path.read_text(encoding="utf-8")) if pub_path.exists() else None
    check = {v: {g: res["presets"][v]["per_gallery"][g]["Q15"] for g in cg.MAIN} for v in variants}
    same = bool(pub) and all(abs(check[v][g] - pub["presets"][v]["per_gallery"][g]["Q15"]) < 1e-12
                             for v in variants for g in cg.MAIN)
    for key, x in by.items():                                  # трактовка a — ровно записи closed_gallery.py
        if key[2] == "service" and key[1] in variants:
            r_a, r_cg = records(x, "a"), rot(*x)
            assert all(np.array_equal(r_a[f], r_cg[f]) for f in ("score_raw", "has", "correct", "top1")), key

    def preset(rg):
        thr, strict = {}, {}
        for r_i in range(cg.ROTATIONS):
            for f in (0, 1):
                tune = qfold != f
                parts = [(rg[g][r_i]["score_raw"][tune], rg[g][r_i]["has"][tune], rg[g][r_i]["correct"][tune])
                         for g in cg.MAIN]
                thr[(r_i, f)] = cg.choose_joint(parts)
                strict[(r_i, f)] = cg.choose_joint(parts, cg.STRICT_TNR, True)
        return float(np.median(list(thr.values()))), float(np.median(list(strict.values())))

    def at(rg, t, ts):
        fx = {**{(r_i, f): t for r_i in range(cg.ROTATIONS) for f in (0, 1)},
              **{("strict", r_i, f): ts for r_i in range(cg.ROTATIONS) for f in (0, 1)}}
        return {g: {k: ev(rg[g], qfold, fx, "score_raw")[k] for k in ("Q15", "F1", "TNR", "recall")} for g in cg.MAIN}

    out = {"tag": a.tag, "protocol": "q17", "modes": MODES, "variants": {}, "counts": {},
           "code_hash": se.code_hash(), "reproduces_closed_gallery": same,
           "note": "a — так считают closed_gallery.py и research/official_evaluate.py; b — решение по оценке сервиса, "
                   "верность по первому не-junk из top-10; c — junk на первом месте считается ошибкой"}
    for v in variants:
        p = res["presets"][v]
        out["variants"][v] = {}
        for mode in MODES:
            rg = {g: [records(by[(g, v, "service", r)], mode) for r in range(cg.ROTATIONS)] for g in cg.MAIN}
            t, ts = preset(rg)
            out["variants"][v][mode] = {"threshold": t, "strict": ts, "per_gallery": at(rg, t, ts),
                                        "at_preset": at(rg, p["threshold"], p["strict"])}
            if mode == "a":
                out["counts"][v] = {g: {
                    "queries": int(len(qfold)),
                    "with_pair": float(np.mean([x["has"].sum() for x in rg[g]])),
                    "junk_top1": float(np.mean([x["junk_top1"].sum() for x in rg[g]])),
                    "junk_top1_without_pair": float(np.mean([(x["junk_top1"] & ~x["has"]).sum() for x in rg[g]])),
                    "junk_top1_share_of_with_pair": float(np.mean([x["junk_top1"][x["has"]].mean() for x in rg[g]])),
                    "junk_top1_accepted_at_preset": float(np.mean([(x["junk_top1"] & (x["score_raw"] >= p["threshold"])).sum()
                                                                   for x in rg[g]]))} for g in cg.MAIN}
            if mode == "b":
                for g in cg.MAIN:
                    out["counts"][v][g]["b_no_other_in_top10"] = float(np.mean([x["none_left"].sum() for x in rg[g]]))
    path = a.out_dir / f"refusal_topcam_{a.tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"сверка с {pub_path.name}: {'совпало' if same else 'НЕ совпало'}\n-> {path}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
