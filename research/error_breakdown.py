#!/usr/bin/env python3
"""Разбор ошибок по подгруппам: где студент ошибается и почему.

Модель — студент из main (.env.example), галерея 2.2k (все машины holdout +
весь тест), режим эталона, сырой косинус — как отвечает /v1/search. Фолды,
open-set и галерея те же, что в research/honest_eval.py. AP@10 каждого запроса —
research/official_evaluate.ranking_metrics на этом одном запросе; средние по
фолду сверяются с honest_eval.evaluate, при расхождении скрипт падает.

Признаки запроса и верного ответа (лучшего по рангу позитива):

  ночь                пара (машина, камера) размечена глазами — день, сумерки,
                      ночь, разное (honest_eval_results/vehicle_camera_labels.csv);
                      «ночь» = ночь или сумерки. Правило error_analysis
                      «яркость кропа < 70» печатается для сравнения: на этих
                      данных оно ловит дневные кадры в тени, а не ночь;
  блики               там же, глазами: засветка фар или солнца, сильное зеркальное
                      отражение на кузове хотя бы на одном из двух кадров пары.
                      Правило error_analysis «пикселей ярче 235 > 2 %» печатается
                      для сравнения: оно срабатывает на светлом фоне и белом кузове;
  мелкий bbox         тот же порог error_analysis (< 120 000), но по площади
                      рамки из train.csv в пикселях кадра: кропы crops/
                      уменьшены до 512 px, их площадь размер машины не отражает;
  смена ракурса       ракурс пары (машина, камера) размечен глазами по кропам —
                      honest_eval_results/vehicle_camera_labels.csv (перед, зад, бок;
                      mixed / unknown в смене ракурса не участвуют); «смена
                      ракурса» — в галерее у запроса нет позитива того же
                      ракурса (свойство запроса, а не выдачи модели). Для
                      сравнения — правило error_analysis по пропорциям рамки
                      (|Δ w/h| > 0.45, запрос против верного ответа);
  похожая машина      пары «запрос — неверный top-1» размечены глазами
  того же цвета       (honest_eval_results/error_pairs_labels.csv): тот же цвет
                      кузова, похожая модель или кузов. Ошибки без разметки
                      считаются отдельно, в категорию не попадают;
                      «сервис принял бы» — косинус top-1 >= порога 0.8025;
  цвет запроса        цвет машины holdout размечен глазами по её самому светлому
                      кадру (honest_eval_results/vehicle_colors.csv).

Правило error_analysis «похожая машина: сходство top-1 > 0.55» не берём: оно
подобрано под шкалу CLIP-головы 18.09, у студента выше 0.55 почти любой top-1.
Цвет по HSV кропа тоже не берём: на 48 размеченных кадрах ошибался в 12.

    python research/error_breakdown.py --data-dir /opt/case17/data
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import error_analysis as ea  # noqa: E402
import honest_eval as he  # noqa: E402
import official_evaluate as oe  # noqa: E402
import service_embeddings as se  # noqa: E402

THRESHOLD = 0.8025
VIEW_RU = {"front": "перед", "rear": "зад", "side": "бок"}
CATEGORIES = ("ночь", "блики", "смена ракурса", "мелкий bbox", "похожая машина того же цвета",
              "соседняя машина в рамке")


COLOR_GROUP = {"black": "чёрный", "white": "белый", "gray": "серый"}


def color_group(c: str) -> str:
    return COLOR_GROUP.get(c, "цветной")


def read_pair_labels(path: Path) -> dict:
    """(запрос, top-1) -> {same_color, similar_model}; файла нет — пусто."""
    if not path.exists():
        return {}
    return {(r["query"], r["top1"]): {"same_color": r["same_color"] == "1", "similar_model": r["similar_model"] == "1",
                                      "neighbor": r["neighbor_in_box"] == "1"}
            for r in csv.DictReader(path.open(encoding="utf-8"))}


def per_query(D: he.Data, kind: str) -> list:
    """Запросы фолдов с рангами и скорами; open-set (нет позитива) — отдельно."""
    extra = D.valid(D.distractors(kind))
    qry, gal = D.valid(D.qry), D.valid(D.gal)
    folds = he.folds_by_vehicle(sorted(D.hold), D.a.folds, D.a.seed)
    rng = np.random.default_rng(D.a.seed + 1)           # тот же open-set, что в honest_eval
    out = []
    for f, fvids in enumerate(folds):
        fset = set(fvids)
        n_open = int(round(len(fvids) * D.a.openset_frac))
        opened = set(rng.choice(sorted(fvids, key=int), size=n_open, replace=False).tolist())
        q_rows = [r for r in qry if r["vehicle_id"] in fset]
        g_rows = [r for r in gal if r["vehicle_id"] not in opened] + extra
        Q, G = he.frame(q_rows), he.frame(g_rows)
        sim = D.vectors(list(Q.index)) @ D.vectors(list(G.index)).T
        ref = he.evaluate(sim, sim, Q, G, "official", THRESHOLD)
        g_ids, g_vid, g_cam = np.array(G.index), G.vehicle_id.to_numpy(), G.camera_id.to_numpy()
        order = np.argsort(-sim, axis=1, kind="stable")     # как honest_eval.evaluate
        aps, hits = [], []
        for i, qid in enumerate(Q.index):
            row = Q.iloc[i]
            m = oe.ranking_metrics(Q.iloc[[i]], G, {qid: [g_ids[j] for j in order[i, :10]]})
            junk = (g_vid == row.vehicle_id) & (g_cam == row.camera_id)
            clean = order[i][~junk[order[i]]]
            top = clean[0]
            rec = {"fold": f, "query": qid, "vehicle_id": row.vehicle_id, "camera_id": row.camera_id,
                   "top1": g_ids[top], "top1_vehicle": g_vid[top], "top1_sim": float(sim[i, top]),
                   "openset": not m["n_scored"]}
            if m["n_scored"]:
                first = int(np.flatnonzero(g_vid[clean] == row.vehicle_id)[0])
                c = clean[first]
                rec.update(ap10=m["mAP@10"], hit1=bool(m["Rank-1"]), correct=g_ids[c],
                           correct_camera=g_cam[c], correct_rank=first + 1, correct_sim=float(sim[i, c]),
                           positives=";".join(g_ids[j] for j in clean if g_vid[j] == row.vehicle_id))
                aps.append(m["mAP@10"])
                hits.append(m["Rank-1"])
            out.append(rec)
        if abs(np.mean(aps) - ref["mAP@10"]) > 1e-12 or abs(np.mean(hits) - ref["Rank-1"]) > 1e-12:
            raise AssertionError(f"фолд {f}: по запросам mAP@10 {np.mean(aps)} / R1 {np.mean(hits)}, "
                                 f"honest_eval.evaluate {ref['mAP@10']} / {ref['Rank-1']}")
        print(f"  фолд {f}: запросов {len(q_rows)}, mAP@10 {ref['mAP@10']:.4f}, R1 {ref['Rank-1']:.4f} "
              f"(сверено с honest_eval.evaluate)", flush=True)
    return out


def describe(D: he.Data, recs: list, views: dict, light: dict, glare: dict, colors: dict, pairs: dict) -> None:
    """Признаки и категории ошибок — в те же записи."""
    def area(i):
        b = D.boxes[i]
        return int(b[2]) * int(b[3])

    cam = {r["image_id"]: (r["vehicle_id"], r["camera_id"]) for r in D.train}
    for r in recs:
        q = r["query"]
        qs = {**ea.crop_stats(q), "area": area(q)}
        r.update(q_luma=round(qs["luma"], 1), q_bright=round(qs["bright_frac"], 4), q_area=qs["area"],
                 q_aspect=round(qs["aspect"], 3), q_view=views.get(cam[q], "unknown"),
                 q_light=light.get(cam[q], "unknown"), q_glare=glare.get(cam[q], False),
                 q_color=colors.get(r["vehicle_id"], "unknown"))
        if r["openset"]:
            continue
        cs = {**ea.crop_stats(r["correct"]), "area": area(r["correct"])}
        r["correct_view"] = views.get(cam[r["correct"]], "unknown")
        # Свойство запроса, а не выдачи: есть ли в галерее позитив того же ракурса.
        pv = {views.get(cam[p], "unknown") for p in r["positives"].split(";")}
        qv = r["q_view"]
        if qv in VIEW_RU and pv <= set(VIEW_RU):
            r["pos_same_view"] = qv in pv
            if not r["pos_same_view"]:
                r["view_change"] = ("перед↔зад" if qv != "side" and ({"front", "rear"} - {qv}) & pv
                                    else "перед/зад↔бок")
        if r["hit1"]:
            r["categories"] = ""
            continue
        base = ea.categorize(qs, cs, r["top1_sim"], r["correct_rank"])
        r["luma_rule_night"] = "ночь" in base
        cats = ["ночь"] if r["q_light"] in ("night", "dusk") else []
        r["bright_rule_glare"] = "блики" in base
        cats += ["блики"] if r["q_glare"] else []
        cats += ["мелкий bbox"] if "мелкий bbox" in base else []
        if "сильная смена ракурса" in base:
            r["aspect_rule"] = True
        if r.get("view_change"):
            cats.append("смена ракурса")
        lab = pairs.get((q, r["top1"]))
        r["pair_labeled"] = lab is not None
        if lab:
            r["same_color"], r["similar_model"], r["neighbor"] = lab["same_color"], lab["similar_model"], lab["neighbor"]
            if lab["same_color"]:
                cats.append("похожая машина того же цвета")
            if lab["neighbor"]:
                cats.append("соседняя машина в рамке")
        r["categories"] = ";".join(cats) or "прочее"


def summarize(recs: list) -> dict:
    scored = [r for r in recs if not r["openset"]]
    fails = [r for r in scored if not r["hit1"]]
    n_s, n_f = len(scored), len(fails)

    def group(name, pred):
        rows = [r for r in scored if pred(r)]
        if not rows:
            return {"group": name, "n": 0}
        fr = [r for r in rows if not r["hit1"]]
        return {"group": name, "n": len(rows), "share_queries": round(len(rows) / n_s, 4),
                "mAP@10": round(float(np.mean([r["ap10"] for r in rows])), 4),
                "Rank-1": round(float(np.mean([r["hit1"] for r in rows])), 4),
                "errors": len(fr), "share_errors": round(len(fr) / n_f, 4)}

    def has(cat):
        return lambda r: cat in r.get("categories", "").split(";")

    cats = {c: sum(has(c)(r) for r in fails) for c in CATEGORIES}
    vc = Counter(r.get("view_change") for r in fails if r.get("view_change"))
    return {
        "scored_queries": n_s, "openset_queries": len(recs) - n_s, "errors": n_f,
        "mAP@10": round(float(np.mean([r["ap10"] for r in scored])), 4),
        "Rank-1": round(float(np.mean([r["hit1"] for r in scored])), 4),
        "errors_by_category": {c: {"n": n, "share_errors": round(n / n_f, 4)} for c, n in cats.items()},
        "view_change_kinds": {k: {"n": n, "share_errors": round(n / n_f, 4)} for k, n in vc.items()},
        "aspect_rule_errors": {"n": sum(bool(r.get("aspect_rule")) for r in fails),
                               "share_errors": round(sum(bool(r.get("aspect_rule")) for r in fails) / n_f, 4)},
        "other_errors": {"n": sum(r["categories"] == "прочее" for r in fails),
                         "share_errors": round(sum(r["categories"] == "прочее" for r in fails) / n_f, 4)},
        "pair_labels": {"labeled": sum(r["pair_labeled"] for r in fails),
                        "same_color": sum(bool(r.get("same_color")) for r in fails),
                        "same_color_and_similar_model": sum(bool(r.get("same_color") and r.get("similar_model"))
                                                            for r in fails),
                        "similar_model": sum(bool(r.get("similar_model")) for r in fails)},
        "confident_wrong_same_color": sum(has("похожая машина того же цвета")(r) and r["top1_sim"] >= THRESHOLD
                                          for r in fails),
        "confident_wrong_any": sum(r["top1_sim"] >= THRESHOLD for r in fails),
        "correct_rank_on_error": {"median": int(np.median([r["correct_rank"] for r in fails])),
                                  "in_top10": sum(r["correct_rank"] <= 10 for r in fails),
                                  "beyond_50": sum(r["correct_rank"] > 50 for r in fails)},
        "subgroups": [
            group("ночь или сумерки (глазами)", lambda r: r["q_light"] in ("night", "dusk")),
            group("день (глазами)", lambda r: r["q_light"] == "day"),
            group("для сравнения: яркость кропа < 70", lambda r: r["q_luma"] < 70),
            group("блики (глазами)", lambda r: r["q_glare"]),
            group("без бликов (глазами)", lambda r: not r["q_glare"]),
            group("для сравнения: пикселей ярче 235 > 2 %", lambda r: r["q_bright"] > 0.02),
            group("мелкий bbox (< 120 000 пикс.)", lambda r: r["q_area"] < 120_000),
            group("обычный bbox", lambda r: r["q_area"] >= 120_000),
            *[group(f"ракурс запроса: {ru}", lambda r, v=v: r["q_view"] == v) for v, ru in VIEW_RU.items()],
            group("в галерее есть позитив того же ракурса", lambda r: r.get("pos_same_view") is True),
            group("позитивы только другого ракурса", lambda r: r.get("pos_same_view") is False),
            group("— только другого: перед↔зад", lambda r: r.get("view_change") == "перед↔зад"),
            group("— только другого: перед/зад↔бок", lambda r: r.get("view_change") == "перед/зад↔бок"),
            *[group(f"цвет запроса: {c}", lambda r, c=c: color_group(r["q_color"]) == c)
              for c in ("чёрный", "белый", "серый", "цветной")],
        ],
    }


def markdown(s: dict, env: dict) -> str:
    n_f = s["errors"]
    lines = [
        "# Разбор ошибок по подгруппам (шаг 7)", "",
        f"Студент из `main` (`{env.get('REID_WEIGHTS', '?')}`, `REID_TF_MODE={env.get('REID_TF_MODE', '')}`, "
        f"`EMBED_TTA_FLIP={env.get('EMBED_TTA_FLIP', '')}`), галерея 2.2k (машины holdout + весь тест), "
        "режим эталона, сырой косинус — как отвечает `/v1/search`; 5 фолдов вместе.",
        f"Запросов с парой {s['scored_queries']}, без пары {s['openset_queries']} (в mAP не входят). "
        f"mAP@10 {s['mAP@10']:.4f}, Rank-1 {s['Rank-1']:.4f} — среднее по всем запросам вместе; среднее пяти "
        "фолдов в `honest_eval_student_rk_suite.md` — 0.704 / 0.678. По каждому фолду средний AP запросов сверен "
        f"с `honest_eval.evaluate`. Ошибок top-1 — {n_f}.", "",
        "Воспроизвести: `python research/error_breakdown.py --data-dir <данные>`; картинки — "
        "`python research/error_examples.py --data-dir <данные>`.", "",
        "## Как размечено", "",
        "- Мелкий bbox — правило `research/error_analysis.py` (площадь рамки < 120 000 пикселей кадра; площадь "
        "берётся из `train.csv`, потому что кропы уменьшены до 512 px).",
        "- Блики — глазами, на пару (машина, камера): засветка фар или солнца, сильное отражение на кузове; 38 пар "
        "из 361. Правило `error_analysis.py` «пикселей ярче 235 больше 2 %» отмечает 56 % запросов, в том числе "
        "солнечную плитку и белый кузов, поэтому только для сравнения.",
        "- Ночь — глазами, на пару (машина, камера), по первому и последнему кадру: ночь 118 кадров, сумерки 32, "
        "разное в паре 9 пар. Правило `error_analysis.py` «яркость кропа < 70» здесь не годится: из 72 ночных "
        "запросов оно отметило 18, а ночью назвало 53 дневных кадра в тени (камеры с автоэкспозицией; яркость "
        "полного кадра тоже не разделяет: медиана 111 у ночи против 126 у дня).",
        "- Ракурс — глазами по кропам, на пару (машина, камера): `vehicle_camera_labels.csv`, 361 пара (перед 132, "
        "зад 196, бок 20, разный ракурс в паре 5, не разобрать 8). Камеры парковочные, машины стоят в обе стороны, "
        "поэтому разметка по камере не годится. «Смена ракурса» — в галерее у запроса нет позитива того же ракурса. "
        "Повторный проход по всем 361 паре нашёл 1 ошибку (пара машины 314 с камеры 71: «перед» вместо «зад»), "
        "она исправлена.",
        "- Цвет запроса — глазами, на машину: `vehicle_colors.csv`, 150 машин. Цвет по HSV кропа ошибался в 12 "
        "кадрах из 48, его не берём.",
        "- Пары «запрос — неверный top-1» — глазами: `error_pairs_labels.csv`, все ошибки.",
        "- Разметка — глазами модели (Claude) 17.10, один размечающий: это оценка, а не эталон. Второй проход "
        "тем же размечающим: ракурс — 1 ошибка из 361 пары, пары ошибок — 2 из 129 (две «соседние машины» "
        "оказались просто другой машиной), исправлены; ночь, блики и цвет — один проход. Правило "
        "`error_analysis.py` «похожая машина: сходство > 0.55» подобрано под шкалу CLIP-головы, у студента почти "
        "любой top-1 выше 0.55, поэтому не используется.", "",
        "## Ошибки top-1 по категориям", "",
        "Одна ошибка может попасть в несколько категорий, сумма больше 100 %.", "",
        "| Категория | Ошибок | Доля ошибок |", "|---|---:|---:|",
    ]
    for c, v in s["errors_by_category"].items():
        lines.append(f"| {c} | {v['n']} | {v['share_errors']:.0%} |")
        if c == "смена ракурса":
            for k, kv in sorted(s["view_change_kinds"].items()):
                lines.append(f"| — {k} | {kv['n']} | {kv['share_errors']:.0%} |")
        if c == "похожая машина того же цвета":
            pl = s["pair_labels"]
            lines.append(f"| — та же модель или тот же класс кузова | {pl['same_color_and_similar_model']} | "
                         f"{pl['same_color_and_similar_model'] / n_f:.0%} |")
    lines += [f"| прочее (ни одной категории) | {s['other_errors']['n']} | {s['other_errors']['share_errors']:.0%} |",
              f"| для сравнения: правило `error_analysis.py` по пропорциям рамки | {s['aspect_rule_errors']['n']} | "
              f"{s['aspect_rule_errors']['share_errors']:.0%} |", "",
              f"Пары «запрос — неверный top-1» размечены глазами: {pl['labeled']} из {n_f}. «Похожая модель» — та же "
              f"модель или тот же класс кузова близкого облика (например, два чёрных больших SUV); таких пар любого "
              f"цвета — {pl['similar_model']}. «Соседняя машина в рамке» — кропы совпали по другой машине, "
              "попавшей в обе рамки.", "",
              f"Неверный top-1 со сходством ≥ {THRESHOLD} (сервис выдал бы его кандидатом): "
              f"{s['confident_wrong_any']} из {n_f}, из них того же цвета — {s['confident_wrong_same_color']}. "
              f"Верный ответ при ошибке: медиана места {s['correct_rank_on_error']['median']}, в топ-10 — "
              f"{s['correct_rank_on_error']['in_top10']}, дальше 50-го — {s['correct_rank_on_error']['beyond_50']}.", "",
              "## mAP по подгруппам запросов", "",
              "| Подгруппа | Запросов | Доля запросов | mAP@10 | Rank-1 | Доля ошибок |",
              "|---|---:|---:|---:|---:|---:|"]
    lines.insert(lines.index("## Ошибки top-1 по категориям"), "")
    lines.insert(lines.index("## Ошибки top-1 по категориям"),
                 "Картинки (номера закрыты вручную по `error_examples/plate_masks.json`): "
                 + ", ".join(f"`error_examples/{p.name}`" for p in sorted((HERE / "honest_eval_results" /
                             "error_examples").glob("*.png"))) + ".")
    for g in s["subgroups"]:
        if not g["n"]:
            lines.append(f"| {g['group']} | 0 | — | — | — | — |")
            continue
        lines.append(f"| {g['group']} | {g['n']} | {g['share_queries']:.0%} | {g['mAP@10']:.3f} | "
                     f"{g['Rank-1']:.3f} | {g['share_errors']:.0%} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--distractors", default="test_all", choices=tuple(he.DISTRACTORS))
    ap.add_argument("--pair-labels", type=Path, default=HERE / "honest_eval_results" / "vehicle_camera_labels.csv")
    ap.add_argument("--colors", type=Path, default=HERE / "honest_eval_results" / "vehicle_colors.csv")
    ap.add_argument("--pairs", type=Path, default=HERE / "honest_eval_results" / "error_pairs_labels.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "honest_eval_results")
    # те же параметры данных и модели, что у honest_eval
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
    a = ap.parse_args()

    # error_analysis.crop_stats читает CROPS/train/<id>.jpg; запросы и верные
    # ответы — кадры holdout, они в crops/train.
    ea.CROPS = a.data_dir / "crops"
    D = he.Data(a)
    labels = list(csv.DictReader(a.pair_labels.open(encoding="utf-8")))
    views = {(r["vehicle_id"], r["camera_id"]): r["viewpoint"] for r in labels}
    light = {(r["vehicle_id"], r["camera_id"]): r["light"] for r in labels}
    glare = {(r["vehicle_id"], r["camera_id"]): r["glare"] == "1" for r in labels}
    colors = {r["vehicle_id"]: r["color"] for r in csv.DictReader(a.colors.open())}
    pairs = read_pair_labels(a.pairs)
    if not pairs:
        print(f"ВНИМАНИЕ: нет разметки пар {a.pairs} — «похожая машина того же цвета» не считается",
              file=sys.stderr, flush=True)

    recs = per_query(D, a.distractors)
    describe(D, recs, views, light, glare, colors, pairs)
    s = summarize(recs)
    env = {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~") for k, v in D.env.items()}
    report = {"model_env": env, "distractors": a.distractors, "threshold": THRESHOLD, **s}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "error_breakdown.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = ["fold", "query", "vehicle_id", "camera_id", "openset", "ap10", "hit1", "top1", "top1_vehicle",
            "top1_sim", "correct", "correct_camera", "correct_rank", "correct_sim", "q_luma", "q_bright", "q_area",
            "q_aspect", "q_light", "luma_rule_night", "q_glare", "bright_rule_glare", "q_view", "correct_view", "pos_same_view", "q_color", "view_change", "aspect_rule",
            "pair_labeled", "same_color", "similar_model", "neighbor", "categories"]
    with (a.out_dir / "error_breakdown_queries.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    (a.out_dir / "error_breakdown.md").write_text(markdown(s, env), encoding="utf-8")
    print(markdown(s, env))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
