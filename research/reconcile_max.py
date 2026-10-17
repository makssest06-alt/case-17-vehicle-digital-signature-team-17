#!/usr/bin/env python3
"""Сверка с research/FINAL_WORK_REPORT.md (ветка team17/finals-student-gpu-handoff, 0fff030).

Воспроизводит числа отчёта для clean a10 нашими модулями и раскладывает
разницу с honest_eval по факторам:

  * малый clean holdout: 550 запросов / 357 кадров галереи (сплит make_split.py
    без 5 пар побайтных дублей holdout ↔ train), один прогон без фолдов, все
    машины запросов есть в галерее; junk = та же машина с той же камеры
    (эталон); сырой косинус;
  * то же с дублями (исторический сплит 551 / 361);
  * лестница honest_eval по фолдам (25 % машин фолда без пары) от 343 до 10 847
    кадров;
  * 8996: 550 / 357 + 8639 кадров обучающих машин; raw, qq, csls, rerank
    (qq + k-reciprocal = «joint k-reciprocal» отчёта), full; режимы эталона
    (junk вычищается из присланных 10 позиций) и «до top-k» (как
    ablate_postprocess.py: junk убран из всей матрицы до выбора 10);
  * отказ: research/calibrate_refusal.py (тот же файл, что у Максима) на малом
    holdout с 50 % «неизвестных» машин, seed 42, цель recall 0.70 — и наши пороги
    в его протоколе и его порог в нашем.

    python research/reconcile_max.py --data-dir <данные> --env REID_WEIGHTS=<веса a10> [--source frames --zip dataset.zip]
    python research/reconcile_max.py --data-dir <данные> --report   # reconcile_max_report.md из JSON
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate_refusal as cr  # noqa: E402  (тот же файл, что в 0fff030)
import calibrate_refusal_folds as crf  # noqa: E402
import honest_eval as he  # noqa: E402
import service_embeddings as se  # noqa: E402

VARIANTS = ("raw", "qq", "csls", "rerank", "kr", "full")
MAX = {"small_clean": {"mAP@10": 0.866609, "Rank-1": 0.830571, "Rank-5": 0.946593},
       "8996_raw_compare": 0.556907, "8996_ablation": {"raw": 0.558472, "qq": 0.681877, "csls": 0.598525,
                                                         "rerank": 0.737479, "full": 0.694322},
       "refusal": {"cross-camera": {"threshold": 0.669864, "recall": 0.701107, "TNR": 0.894161, "F1": 0.775510},
                   "no-camera-filter": {"threshold": 0.874347, "recall": 0.875458, "TNR": 1.0, "F1": 0.933594}}}


def ranking(D, q_rows, g_rows, variants, params, modes=("official", "nodup")) -> dict:
    Q, G = he.frame(q_rows), he.frame(g_rows)
    qv, gv = D.vectors(list(Q.index)), D.vectors(list(G.index))
    raw = qv @ gv.T
    out = {}
    for v in variants:
        for m in modes:
            r = he.evaluate(he.variant_scores(qv, gv, v, params, Q, G, m), raw, Q, G, m, 0.7, refusal=False)
            out[f"{v}/{m}"] = {k: r[k] for k in ("mAP@10", "Rank-1", "Rank-5", "n_scored", "n_openset")}
    return out


def max_protocol(D, q_rows, g_rows, seed: int, frac: float, target: float, cross_camera: bool, extra=()) -> dict:
    """calibrate_refusal.main без файлов: те же unknown_ids, top1_scores и choose_threshold."""
    qv, gv = D.vectors([r["image_id"] for r in q_rows]), D.vectors([r["image_id"] for r in g_rows])
    vids = np.array(sorted({r["vehicle_id"] for r in q_rows}))
    rng = np.random.default_rng(seed)
    unknown = set(rng.choice(vids, size=max(1, int(len(vids) * frac)), replace=False).tolist())
    known_s, unknown_s = cr.top1_scores(qv, gv, q_rows, g_rows, unknown, cross_camera)
    chosen = cr.choose_threshold(known_s, unknown_s, target)
    at = {f"{t:g}": cr.metrics(known_s, unknown_s, t) for t in (0.8025, 0.70, 0.652, *extra)}
    # те же запросы и галереи, но TP — только если top-1 верный (семантика эталона)
    sim = qv @ gv.T
    g_vid = np.array([r["vehicle_id"] for r in g_rows])
    g_cam = np.array([r["camera_id"] for r in g_rows])
    conf, has, ok = [], [], []
    for qi, row in enumerate(q_rows):
        keep = np.ones(len(g_rows), bool)
        if cross_camera:
            keep &= g_cam != row["camera_id"]
        if row["vehicle_id"] in unknown:
            keep &= g_vid != row["vehicle_id"]
        elif not (keep & (g_vid == row["vehicle_id"])).any():
            continue                                          # как top1_scores: известный без пары пропускается
        s = np.where(keep, sim[qi], -np.inf)
        j = int(s.argmax())
        conf.append(float(s[j]))
        has.append(row["vehicle_id"] not in unknown)
        ok.append(g_vid[j] == row["vehicle_id"])
    rec = {"conf": np.array(conf), "has_match": np.array(has), "correct": np.array(ok)}
    official = {f"{t:g}": crf.metrics(rec, t) for t in (chosen["threshold"], 0.8025, 0.70, 0.652, *extra)}
    return {"unknown_vehicles": len(unknown), "known": len(known_s), "unknown": len(unknown_s),
            "chosen": chosen, "at": at, "official_semantics": official}


def report_md(res: Path) -> str:
    """reconcile_max_report.md из reconcile_max_a10_{frames,crops}.json и refusal_score_a10.json."""
    fr = json.loads((res / "reconcile_max_a10_frames.json").read_text(encoding="utf-8"))
    cr_ = json.loads((res / "reconcile_max_a10_crops.json").read_text(encoding="utf-8"))
    rs = json.loads((res / "refusal_score_a10.json").read_text(encoding="utf-8"))["rules"]
    M = fr["max_reference"]
    f4, f3 = (lambda x: f"{x:.4f}"), (lambda x: f"{x:.3f}")
    both = lambda key, v, m="mAP@10": f"{f4(fr[key][v][m])} / {f4(cr_[key][v][m])}"
    lad = lambda d, k: d["ladder"][k]["official"]["mean"]
    xc, nc = fr["refusal_max_protocol"]["cross-camera"], fr["refusal_max_protocol"]["no-camera-filter"]
    t67 = f"{M['refusal']['cross-camera']['threshold']:g}"
    our = lambda name, k, m: rs[name]["galleries"][k][m]["mean"]
    ours_at = fr["max_threshold_in_honest_eval"]
    sm, kq = fr["sizes"]["small_clean"], xc["known"]
    rows = [
        ("mAP@10, малый clean holdout, raw", f4(M["small_clean"]["mAP@10"]), both("small_clean", "raw/official"),
         f"{sm[1]} (запросов {sm[0]}, в зачёте 543)", "одна галерея, все машины запросов в ней; junk — та же машина "
         "с той же камеры (эталон); 5 пар дублей убраны с обеих сторон", "FINAL_WORK_REPORT §2, §4"),
        ("Rank-1 / Rank-5, там же", f"{f4(M['small_clean']['Rank-1'])} / {f4(M['small_clean']['Rank-5'])}",
         f"{f4(fr['small_clean']['raw/official']['Rank-1'])} / {f4(fr['small_clean']['raw/official']['Rank-5'])} (кадры)",
         f"{sm[1]}", "то же", "FINAL_WORK_REPORT §4"),
        ("mAP@10, тот же holdout с дублями", "—", both("small_with_dups", "raw/official"),
         f"{fr['sizes']['small_with_dups'][1]} (запросов {fr['sizes']['small_with_dups'][0]})",
         "исторический сплит make_split.py", "наш прогон"),
        ("mAP@10, honest_eval: только holdout", "—", f"{f4(lad(fr, 'none'))} / {f4(lad(cr_, 'none'))}",
         f"{fr['ladder']['none']['n_gallery']}", "5 фолдов, 25 % машин фолда без пары, эталон", "наш прогон"),
        ("mAP@10, honest_eval: 2.2k", "—", f"{f4(lad(fr, 'test_all'))} / {f4(lad(cr_, 'test_all'))}",
         f"{fr['ladder']['test_all']['n_gallery']}", "то же + весь тест чужими машинами", "наш прогон, prod_a10.md"),
        ("mAP@10, 8996 raw (эталон: junk из присланных 10)", f4(M["8996_raw_compare"]), both("team8996", "raw/official"),
         "8996", "550 / 357 + 8639 кадров обучающих машин", "FINAL_WORK_REPORT §5, сравнительный прогон"),
        ("mAP@10, 8996 raw (junk до выбора top-k)", f4(M["8996_ablation"]["raw"]), both("team8996", "raw/nodup"),
         "8996", "то же, junk убран из всей матрицы (ablate_postprocess.py)", "FINAL_WORK_REPORT §5, абляция"),
        ("mAP@10, 8996 qq + k-reciprocal", f4(M["8996_ablation"]["rerank"]), both("team8996", "rerank/nodup"),
         "8996", "трансдуктивно: расширение запроса по другим запросам", "FINAL_WORK_REPORT §5, абляция"),
        ("mAP@10, 8996 full (qq → k-reciprocal → CSLS)", f4(M["8996_ablation"]["full"]), both("team8996", "full/nodup"),
         "8996", "то же", "FINAL_WORK_REPORT §5, абляция"),
        ("mAP@10, honest_eval: 10.8k", "—",
         f"{f4(lad(fr, 'test_all+train_seen'))} / {f4(lad(cr_, 'test_all+train_seen'))}",
         f"{fr['ladder']['test_all+train_seen']['n_gallery']}", "5 фолдов + тест и обучающие", "наш прогон, prod_a10.md"),
        ("Порог отказа, цель recall 0.70", t67, f"{f4(xc['chosen']['threshold'])} / "
         f"{f4(cr_['refusal_max_protocol']['cross-camera']['chosen']['threshold'])}",
         f"{sm[1]}", "calibrate_refusal.py: 50 % машин «неизвестны», seed 42, без кадров той же камеры, подбор в "
         "выборке; TP — принят известный запрос (top-1 может быть неверным)", "FINAL_WORK_REPORT §7"),
        ("Recall / TNR / F1 при этом пороге", f"{f3(M['refusal']['cross-camera']['recall'])} / "
         f"{f3(M['refusal']['cross-camera']['TNR'])} / {f3(M['refusal']['cross-camera']['F1'])}",
         f"{f3(xc['chosen']['recall'])} / {f3(xc['chosen']['TNR'])} / {f3(xc['chosen']['f1'])} (кадры)",
         f"{sm[1]}", f"то же; известных запросов {kq}, неизвестных {xc['unknown']}", "FINAL_WORK_REPORT §7"),
        ("То же, семантика эталона (TP — top-1 верный)", "—",
         f"{f3(xc['official_semantics'][t67]['recall'])} / {f3(xc['official_semantics'][t67]['TNR'])} / "
         f"{f3(xc['official_semantics'][t67]['F1'])} (кадры)", f"{sm[1]}", "те же запросы и галерея", "наш прогон"),
        ("Порог «без фильтра камер»", f"{M['refusal']['no-camera-filter']['threshold']:g}",
         f"{f4(nc['chosen']['threshold'])} (кадры)", f"{sm[1]}",
         "кадры той же машины с той же камеры остаются в галерее", "FINAL_WORK_REPORT §7"),
        (f"Recall / TNR / F1 при {t67} в нашем протоколе", "—",
         " / ".join(f"{f3(d['max_threshold_in_honest_eval'][t67]['recall']['mean'])} · "
                    f"{f3(d['max_threshold_in_honest_eval'][t67]['TNR']['mean'])} · "
                    f"{f3(d['max_threshold_in_honest_eval'][t67]['F1']['mean'])}" for d in (fr, cr_)),
         "2203", "5 фолдов, без почти-дублей, семантика эталона", "наш прогон"),
        ("Рабочая точка 0.70: F1 / TNR / Recall", "—",
         f"{f3(our('raw@0.7', 'test_all', 'F1'))} / {f3(our('raw@0.7', 'test_all', 'TNR'))} / "
         f"{f3(our('raw@0.7', 'test_all', 'recall'))} (кропы)", "2203", "то же; порог выбран вне фолда (цель recall ≥ 0.65)",
         "prod_a10.md, refusal_score_a10.md"),
        ("F1-оптимальный порог вне фолда", "—", f"{rs['raw@f1']['threshold']['mean']:.3f} ± "
         f"{rs['raw@f1']['threshold']['std']:.3f}: F1 {f3(our('raw@f1', 'test_all', 'F1'))}, TNR "
         f"{f3(our('raw@f1', 'test_all', 'TNR'))} (кропы)", "2203", "то же", "refusal_score_a10.md"),
    ]
    out = ["# Сверка с отчётом Максима (FINAL_WORK_REPORT.md, 0fff030)", "",
           "Отчёт прочитан через `git show 0fff030:research/FINAL_WORK_REPORT.md`, код — `calibrate_refusal.py` того "
           "же коммита (побайтно совпадает с файлом в этой ветке). Чужие ветки не менялись. Модель — "
           "`clean_student_a10_ep28.pt` (sha256 сверен с манифестом), сырой косинус, если не сказано иначе. "
           "Наши числа — «полные кадры / готовые кропы»: полный кадр, обрезанный кодом сервиса, как у команды; "
           "кропы `crops/` — как в остальных наших отчётах. Воспроизвести: `python research/reconcile_max.py "
           "--data-dir <данные> --env REID_WEIGHTS=<веса> [--source frames --zip dataset.zip]`, затем `--report`.", "",
           "| Метрика | Максим | Наше: кадры / кропы | Размер галереи | Протокол | Источник |",
           "|---|---:|---:|---:|---|---|"]
    out += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in rows]
    d_small = fr["small_clean"]["raw/official"]["mAP@10"] - M["small_clean"]["mAP@10"]
    d_dup = fr["small_with_dups"]["raw/official"]["mAP@10"] - fr["small_clean"]["raw/official"]["mAP@10"]
    d_junk = fr["team8996"]["raw/nodup"]["mAP@10"] - fr["team8996"]["raw/official"]["mAP@10"]
    d_src = max(abs(lad(fr, k) - lad(cr_, k)) for k in fr["ladder"])
    d_pp = fr["team8996"]["rerank/nodup"]["mAP@10"] - fr["team8996"]["raw/nodup"]["mAP@10"]
    out += ["", "## Откуда разница 0.867 → 0.687 → 0.549", "",
            f"- **Воспроизведение.** На полных кадрах {f4(fr['small_clean']['raw/official']['mAP@10'])} против "
            f"{f4(M['small_clean']['mAP@10'])} ({d_small:+.4f}); Rank-1 и Rank-5 расходятся на один запрос из 543. "
            "Протокол Максима понят правильно: одна галерея 357 кадров, все машины запросов в ней, 7 запросов без пары "
            "с другой камеры исключены, junk — та же машина с той же камеры, дубли убраны, сырой косинус.",
            f"- **Дубли** почти ничего не дают: с 5 парами {f4(fr['small_with_dups']['raw/official']['mAP@10'])} "
            f"({d_dup:+.4f}).",
            f"- **Камеры** (junk из присланных 10 против junk до выбора top-k) — {d_junk:+.4f} на 8996. Это и есть две "
            f"raw-цифры отчёта: {M['8996_raw_compare']} (эталон) и {M['8996_ablation']['raw']} (абляция).",
            f"- **Источник кадра** (кадр против кропа) — до {d_src:.4f}.",
            f"- **Фолды и open-set** при той же галерее не снижают: «только holdout» по фолдам "
            f"{f4(lad(fr, 'none'))} на {fr['ladder']['none']['n_gallery']} кадрах.",
            f"- **Размер галереи — главный фактор**: {f4(lad(fr, 'none'))} → {f4(lad(fr, 'test_gallery'))} → "
            f"{f4(lad(fr, 'test_all'))} → {f4(lad(fr, 'train_seen'))} → {f4(lad(fr, 'test_all+train_seen'))} при "
            + " → ".join(str(fr["ladder"][k]["n_gallery"]) for k in fr["ladder"]) + " кадрах.",
            f"- **Постобработка**: qq + k-reciprocal на 8996 даёт {d_pp:+.4f} к raw, но расширяет запрос по другим "
            "запросам — сервис по одному кадру так не ответит, отчёт Максима это тоже оговаривает.", "",
            "## Порог отказа", "",
            f"- Порог Максима {t67} воспроизведён до знака, но подобран на малой галерее ({sm[1]} кадров), в выборке, "
            "с 50 % неизвестных машин, без кадров той же камеры, и TP у него — любой принятый известный запрос.",
            f"- Числа порогов близки случайно: и он, и мы ищем recall ≈ 0.65–0.70 на сыром косинусе (наши вне фолда на "
            f"2.2k: F1-оптимум {rs['raw@f1']['threshold']['mean']:.3f}, рабочая точка 0.70, цель recall ≥ 0.70 — "
            "0.691). Совпадают пороги, не метрики при них.",
            f"- Его F1 {f3(M['refusal']['cross-camera']['F1'])} и TNR {f3(M['refusal']['cross-camera']['TNR'])} — "
            f"свойство малой галереи и его семантики: на тех же данных с TP «top-1 верный» F1 "
            f"{f3(xc['official_semantics'][t67]['F1'])}; тот же порог в нашем протоколе на 2.2k — F1 "
            f"{f3(ours_at[t67]['F1']['mean'])}, TNR {f3(ours_at[t67]['TNR']['mean'])}.",
            f"- Порог «без фильтра камер» {M['refusal']['no-camera-filter']['threshold']:g} опирается на соседние кадры "
            f"одного проезда: в нашем протоколе recall {f3(ours_at['0.874347']['recall']['mean'])}, F1 "
            f"{f3(ours_at['0.874347']['F1']['mean'])}.", "",
            "## Что показываем на питче", "",
            f"**mAP@10 a10 по протоколу организаторов, сырой косинус: {cr_['small_clean']['raw/official']['mAP@10']:.3f} "
            f"на малой отложенной галерее ({sm[1]} кадров; у Максима {M['small_clean']['mAP@10']:.3f}), "
            f"{lad(cr_, 'test_all'):.3f} на 2.2k и {lad(cr_, 'test_all+train_seen'):.3f} на 10.8k кадров; отказ на пороге 0.70 — F1 {our('raw@0.7', 'test_all', 'F1'):.2f}, TNR "
            f"{our('raw@0.7', 'test_all', 'TNR'):.2f} (2.2k, вне фолда). Каждое число — с размером галереи; "
            f"{M['8996_ablation']['rerank']:.3f} с qq + k-reciprocal и F1 {M['refusal']['cross-camera']['F1']:.3f} "
            "без оговорок не показываем.**", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--tag", default="a10")
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
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", action="store_true", help="только собрать reconcile_max_report.md из готовых JSON")
    a = ap.parse_args()
    if a.report:
        res = HERE / "honest_eval_results"
        (res / "reconcile_max_report.md").write_text(report_md(res), encoding="utf-8")
        print(f"-> {res / 'reconcile_max_report.md'}")
        return 0

    D = he.Data(a)
    params = he.pp_params(a.pp_params)
    tq, tg = he.team8996(D)
    tq, tg = D.valid(tq), D.valid(tg)
    small = [r for r in tg if r["vehicle_id"] in D.hold]
    old_q = D.valid([r for r in D.split_old if r["role"] == "query"])
    old_g = D.valid([r for r in D.split_old if r["role"] == "gallery"])
    out = {"source": a.source, "params": params, "sizes": {"small_clean": [len(tq), len(small)],
                                                         "small_with_dups": [len(old_q), len(old_g)],
                                                         "team8996": [len(tq), len(tg)]}}
    out["small_clean"] = ranking(D, tq, small, ("raw", "rerank", "kr"), params)
    out["small_with_dups"] = ranking(D, old_q, old_g, ("raw",), params)
    out["team8996"] = ranking(D, tq, tg, VARIANTS, params)
    ladder = {}
    for kind in ("none", "test_gallery", "test_all", "train_seen", "test_all+train_seen"):
        s, pf, _ = he.run_folds(D, kind, ["raw"], params, 0.7, galleries=("big",))
        ladder[kind] = {"n_gallery": int(round(np.mean([r["n_gallery"] for r in pf]))),
                        "official": s["big/raw/official"]["mAP@10"], "nodup": s["big/raw/nodup"]["mAP@10"],
                        "Rank-1": s["big/raw/official"]["Rank-1"]["mean"]}
    out["ladder"] = ladder
    out["refusal_max_protocol"] = {
        "cross-camera": max_protocol(D, tq, small, 42, 0.5, 0.70, True, extra=(0.669864,)),
        "no-camera-filter": max_protocol(D, tq, small, 42, 0.5, 0.70, False, extra=(0.874347,))}
    # порог Максима в нашем протоколе (2.2k, без почти-дублей, среднее по фолдам)
    recs = crf.fold_records(D, "test_all", "nodup")
    out["max_threshold_in_honest_eval"] = {
        f"{t:g}": {k: crf.stats([crf.metrics(r, t)[k] for r in recs]) for k in ("recall", "TNR", "F1", "precision")}
        for t in (0.669864, 0.874347)}
    out["max_reference"] = MAX
    path = a.out or (HERE / "honest_eval_results" / f"reconcile_max_{a.tag}_{a.source}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    sc = out["small_clean"]["raw/official"]
    print(f"малый clean holdout {len(tq)}/{len(small)}: mAP@10 {sc['mAP@10']:.6f} R1 {sc['Rank-1']:.6f} "
          f"R5 {sc['Rank-5']:.6f} (в зачёте {sc['n_scored']}, без пары {sc['n_openset']}); у Максима "
          f"{MAX['small_clean']['mAP@10']} / {MAX['small_clean']['Rank-1']} / {MAX['small_clean']['Rank-5']}")
    for v in VARIANTS:
        t = out["team8996"]
        print(f"  8996 {v:7s} эталон {t[f'{v}/official']['mAP@10']:.6f}  до top-k {t[f'{v}/nodup']['mAP@10']:.6f}  "
              f"у Максима {MAX['8996_ablation'].get(v, '—')}")
    for proto, r in out["refusal_max_protocol"].items():
        c = r["chosen"]
        print(f"  отказ, {proto}: порог {c['threshold']:.6f} recall {c['recall']:.6f} TNR {c['TNR']:.6f} "
              f"F1 {c['f1']:.6f} (у Максима {MAX['refusal'][proto]})")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
