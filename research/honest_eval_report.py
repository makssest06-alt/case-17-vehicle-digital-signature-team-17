#!/usr/bin/env python3
"""Сводные таблицы из прогонов research/honest_eval.py — без ручного переписывания чисел.

    python research/honest_eval_report.py --tag student \\
        --ladder ladder_none ladder_test_gallery student ladder_train_seen ladder_test_all_train_seen \\
        --out research/honest_eval_results/honest_eval_summary.md

Читает research/honest_eval_results/honest_eval_<tag>.json.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "honest_eval_results"
LADDER_NAMES = {"none": "только holdout", "test_gallery": "+ test gallery (750)",
                "test_all": "+ весь тест (1860)", "train_seen": "+ обучающие машины (8644)",
                "test_all+train_seen": "+ тест и обучающие (10 504)"}


def load(tag: str) -> dict:
    return json.loads((RES / f"honest_eval_{tag}.json").read_text(encoding="utf-8"))


def ms(s: dict, key: str, nd: int = 3) -> str:
    return f"{s[key]['mean']:.{nd}f} ± {s[key]['std']:.{nd}f}"


def main_table(r: dict) -> list:
    out = ["| Галерея | Постобработка | Режим | mAP@10 | Rank-1 | Rank-5 | mAP | mINP | F1 | Recall | TNR | PR-AUC без порога |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, s in r["summary"].items():
        g, v, m = key.split("/")
        out.append(f"| {g} | {v} | {m} | {ms(s, 'mAP@10')} | {s['Rank-1']['mean']:.3f} | {s['Rank-5']['mean']:.3f} | "
                   f"{s['mAP_full']['mean']:.3f} | {s['mINP']['mean']:.3f} | {s['F1']['mean']:.3f} | "
                   f"{s['Recall']['mean']:.3f} | {s['TNR']['mean']:.3f} | {s['PR-AUC_all']['mean']:.3f} |")
    return out


def old_table(r: dict) -> list:
    out = ["| Постобработка | mAP@10 | Rank-1 | Rank-5 | mAP | mINP |", "|---|---:|---:|---:|---:|---:|"]
    for v, m in r["old_protocol"].items():
        out.append(f"| {v} | {m['mAP@10']:.4f} | {m['Rank-1']:.4f} | {m['Rank-5']:.4f} | {m['mAP_full']:.4f} | {m['mINP']:.4f} |")
    return out


def ladder_table(tags: list, variant: str = "raw") -> list:
    out = ["| Дистракторы | Кадров в big-галерее (среднее по фолдам) | mAP@10 official | mAP@10 nodup | Rank-1 | mINP | Recall nodup | TNR |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag in tags:
        r = load(tag)
        s, n = r["summary"][f"big/{variant}/official"], r["summary"][f"big/{variant}/nodup"]
        # имя ступени восстанавливаем по числу дистракторов
        name = {0: LADDER_NAMES["none"], 750: LADDER_NAMES["test_gallery"], 1860: LADDER_NAMES["test_all"],
                8644: LADDER_NAMES["train_seen"], 10504: LADDER_NAMES["test_all+train_seen"]}.get(r["extra_distractors"], tag)
        out.append(f"| {name} | {int(s['n_gallery']['mean'])} | {ms(s, 'mAP@10')} | {n['mAP@10']['mean']:.3f} | "
                   f"{s['Rank-1']['mean']:.3f} | {s['mINP']['mean']:.3f} | {n['Recall']['mean']:.3f} | {s['TNR']['mean']:.3f} |")
    return out


TEAM_DOC = HERE / "GENERALIZATION_RESULTS.md"
TEAM_ROWS = {"raw cosine": "raw", "query expansion": "qq", "gallery-side CSLS": "csls",
             "k-reciprocal": "rerank", "old full pipeline": "full"}


def team_tables() -> dict:
    """Таблицы абляции из research/GENERALIZATION_RESULTS.md: {раздел: {вариант: (mAP@10, R1, R5)}}."""
    out, section = {}, None
    for line in TEAM_DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = "small" if "Small gallery" in line else ("8996" if "Clean-like" in line else None)
            continue
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if section and cells and cells[0] in TEAM_ROWS and len(cells) == 4:
            out.setdefault(section, {})[TEAM_ROWS[cells[0]]] = tuple(float(c) for c in cells[1:])
    return out


def team_compare(tags: dict) -> list:
    """Сверка с командой один в один: те же галереи, кропы из crops/ и полные кадры."""
    team = team_tables()
    runs = {src: load(tag) for src, tag in tags.items()}
    out = ["## Сверка с research/GENERALIZATION_RESULTS.md", "",
           "Те же галереи: маленькая — сплит make_split.py (551 / 361), clean-like — 550 запросов, "
           "8996 кадров (357 holdout + 8639 обучающих, 7 запросов без пары). Числа команды — из таблиц "
           "документа; Δ = мы − команда. «кропы» — готовые crops/ (как make_submission.py), "
           "«кадры» — полный кадр, кроп кодом сервиса (как /v1/embed и как extract_emb.py команды).", "",
           "| Галерея | Вариант | Команда mAP@10 / R1 / R5 | Кропы mAP@10 (Δ) | Кадры mAP@10 (Δ) | Кадры R1 / R5 |",
           "|---|---|---|---:|---:|---|"]
    for gal, key in (("small", "old_protocol"), ("8996", "team_clean_like")):
        for v, (tm, tr1, tr5) in team.get(gal, {}).items():
            cells = []
            for src in ("crops", "frames"):
                r = runs[src][key]
                m = r[v] if key == "old_protocol" else r[f"{v}/official"]
                cells.append(m)
            c, f = cells
            out.append(f"| {gal} | {v} | {tm:.4f} / {tr1:.4f} / {tr5:.4f} | {c['mAP@10']:.4f} ({c['mAP@10'] - tm:+.4f}) | "
                       f"{f['mAP@10']:.4f} ({f['mAP@10'] - tm:+.4f}) | {f['Rank-1']:.4f} / {f['Rank-5']:.4f} |")
    return out


def compare_suites(tags: list) -> list:
    """Модели рядом по прогонам --suite: mAP@10 и отказ по сырому косинусу."""
    suites = {t: json.loads((RES / f"honest_eval_{t}_suite.json").read_text(encoding="utf-8")) for t in tags}
    galleries = list(next(iter(suites.values()))["suite"])
    head = ["| Галерея | " + " | ".join(tags) + " |", "|---|" + "---:|" * len(tags)]
    rows_map, rows_ref = [], []
    for g in galleries:
        cm, cr = [], []
        for t in tags:
            b = suites[t]["suite"].get(g)
            if not b or "raw" not in b["metrics"]:
                cm.append("—"); cr.append("—")
                continue
            o = b["metrics"]["raw"]["official"]
            if b["kind"] == "folds":
                n = b["metrics"]["raw"]["nodup"]
                cm.append(f"{o['mAP@10']['mean']:.3f} ± {o['mAP@10']['std']:.3f}")
                cr.append(f"{n['Recall']['mean']:.3f} / {n['TNR']['mean']:.3f}")
            else:
                cm.append(f"{o['mAP@10']:.3f}")
                cr.append("—")
        rows_map.append(f"| {g} | " + " | ".join(cm) + " |")
        rows_ref.append(f"| {g} | " + " | ".join(cr) + " |")
    thr = " | ".join(f"{t}: {suites[t]['threshold']}" for t in tags)
    return (["### mAP@10 (сырой косинус, режим эталона)", "", *head, *rows_map, "",
             f"### Отказ без почти-дублей: Recall / TNR (порог: {thr})", "", *head, *rows_ref])


def models_table(models: list, variants: list) -> list:
    """Модели рядом. models — «тег:подпись»; читает honest_eval_<тег>_team8996.json,
    honest_eval_<тег>.json (5 фолдов, big = 2.2k) и refusal_folds_<тег>.json."""
    tags = [m.split(":", 1) for m in models]
    head = "| | " + " | ".join(lbl for _, lbl in tags) + " |"
    sep = "|---|" + "---:|" * len(tags)
    out = ["### Ранжирование: mAP@10 / Rank-1 (режим эталона)", "", head, sep]
    for v in variants:
        for gal in ("8996", "2.2k"):
            cells = []
            for tag, _ in tags:
                try:
                    if gal == "8996":
                        m = load(f"{tag}_team8996")["team_clean_like"][f"{v}/official"]
                        cells.append(f"{m['mAP@10']:.4f} / {m['Rank-1']:.4f}")
                    else:
                        s = load(tag)["summary"][f"big/{v}/official"]
                        cells.append(f"{s['mAP@10']['mean']:.4f} ± {s['mAP@10']['std']:.4f} / {s['Rank-1']['mean']:.4f}")
                except (FileNotFoundError, KeyError):
                    cells.append("—")
            out.append(f"| {gal}, {v} | " + " | ".join(cells) + " |")
    out += ["", "### Отказ, 2.2k, без почти-дублей: Recall / TNR / F1 (сырой косинус, метрики эталона, 5 фолдов)",
            "", head, sep]
    refusal = {}
    for tag, _ in tags:
        p = RES / f"refusal_folds_{tag}.json"
        refusal[tag] = json.loads(p.read_text(encoding="utf-8"))["modes"]["nodup"] if p.exists() else None
    names = []
    for r in refusal.values():
        if r:
            names += [n for n in r["current"] if n not in names]
    for name in names:
        cells = []
        for tag, _ in tags:
            c = (refusal[tag] or {}).get("current", {}).get(name)
            cells.append("—" if not c else f"{c['recall']['mean']:.3f} / {c['TNR']['mean']:.3f} / {c['F1']['mean']:.3f}")
        out.append(f"| порог {name} | " + " | ".join(cells) + " |")
    for target in ("recall>=0.6", "recall>=0.7"):
        cells = []
        for tag, _ in tags:
            t = (refusal[tag] or {}).get("targets", {}).get(target)
            cells.append("—" if not t else f"порог {t['threshold']['mean']:.3f}: {t['recall']['mean']:.3f} / "
                                           f"{t['TNR']['mean']:.3f} / {t['F1']['mean']:.3f}")
        out.append(f"| вне фолда, {target} | " + " | ".join(cells) + " |")
    return out


def prod_table(models: list) -> list:
    """Таблица для порога в проде. models — «тег:подпись», первый — прод-модель.
    Читает honest_eval_<тег>_suite.json (8996 и 10.8k, raw и kr) и
    refusal_folds_<тег>.json (2.2k, без почти-дублей, вне фолда)."""
    tags = [m.split(":", 1) for m in models]
    head = "| | " + " | ".join(lbl for _, lbl in tags) + " |"
    sep = "|---|" + "---:|" * len(tags)
    suites = {}
    for tag, _ in tags:
        p = RES / f"honest_eval_{tag}_suite.json"
        suites[tag] = json.loads(p.read_text(encoding="utf-8"))["suite"] if p.exists() else {}
    weights = []
    for tag, lbl in tags:
        p = RES / f"honest_eval_{tag}_suite.json"
        w = json.loads(p.read_text(encoding="utf-8"))["env"].get("REID_WEIGHTS", "?") if p.exists() else "нет прогона"
        weights.append(f"{lbl} — `{w}`")
    out = ["# Сравнение для порога в проде: " + ", ".join(lbl for _, lbl in tags), "",
           "Веса: " + "; ".join(weights) + ". Протокол research/honest_eval.py: 5 фолдов по 150 машинам holdout, "
           "25 % машин фолда без пары, эталон организаторов; kr — k-reciprocal без расширения запроса "
           "(трансдуктивен, сервису недоступен). Отказ — research/calibrate_refusal_folds.py: сырой косинус, "
           "«вне фолда» — порог выбран на одном фолде (максимум TNR при recall ≥ цели) и проверен на остальных "
           "четырёх, ± — разброс выбранного порога. Таблицы собирает "
           f"`python research/honest_eval_report.py --prod {' '.join(repr(m) for m in models)}`.", "",
           "### Ранжирование: mAP@10 / Rank-1, режим эталона", "", head, sep]
    galleries = list(next((sv for sv in suites.values() if sv), {}))
    for key in galleries:
        gal = key.split(" (")[0]
        for v in ("raw", "kr"):
            cells = []
            for tag, _ in tags:
                m = suites[tag].get(key, {}).get("metrics", {}).get(v, {}).get("official")
                if not m:
                    cells.append("—")
                elif isinstance(m["mAP@10"], dict):
                    cells.append(f"{m['mAP@10']['mean']:.3f} ± {m['mAP@10']['std']:.3f} / {m['Rank-1']['mean']:.3f}")
                else:
                    cells.append(f"{m['mAP@10']:.4f} / {m['Rank-1']:.3f}")
            out.append(f"| {gal}, {v} | " + " | ".join(cells) + " |")
    out += ["", "### Отказ, 2.2k, без почти-дублей, 5 фолдов: Recall / TNR / F1", "", head, sep]
    refusal = {}
    for tag, _ in tags:
        p = RES / f"refusal_folds_{tag}.json"
        refusal[tag] = json.loads(p.read_text(encoding="utf-8"))["modes"]["nodup"] if p.exists() else {}
    fixed = [("сбалансированный 0.8025", "порог 0.8025 (сейчас в проде)")]
    fixed += [(n, n) for n in refusal[tags[0][0]].get("current", {}) if n.startswith(("рабочая точка", "точка старого"))]
    for name, label in fixed:
        cells = []
        for tag, _ in tags:
            c = refusal[tag].get("current", {}).get(name)
            cells.append("—" if not c else f"{c['recall']['mean']:.3f} / {c['TNR']['mean']:.3f} / {c['F1']['mean']:.3f}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    targets = list(refusal[tags[0][0]].get("targets", {})) or ["recall>=0.6", "recall>=0.7"]
    for target in targets:
        cells = []
        for tag, _ in tags:
            t = refusal[tag].get("targets", {}).get(target)
            cells.append("—" if not t else
                         f"порог {t['threshold']['mean']:.3f} ± {t['threshold']['std']:.3f}: {t['recall']['mean']:.3f} / "
                         f"{t['TNR']['mean']:.3f} / {t['F1']['mean']:.3f}")
        out.append(f"| вне фолда, {target.replace('>=', ' ≥ ')} | " + " | ".join(cells) + " |")
    first = refusal[tags[0][0]].get("targets", {})
    band = {t: v for t, v in first.items() if BAND[0] <= v["recall"]["mean"] <= BAND[1]}
    if band:
        t, v = max(band.items(), key=lambda kv: kv[1]["TNR"]["mean"])
        out += ["", f"Рабочая точка {tags[0][1]}: среди целей вне фолда с recall на остальных фолдах в "
                f"{BAND[0]}–{BAND[1]} ({', '.join(band)}) выше TNR у {t}: порог {v['threshold']['mean']:.3f} ± "
                f"{v['threshold']['std']:.3f}, recall {v['recall']['mean']:.3f}, TNR {v['TNR']['mean']:.3f}."]
    return out


BAND = (0.65, 0.70)          # полоса recall для рабочей точки отказа


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="«тег:подпись» ...: таблица моделей рядом (8996, 2.2k, отказ) и выход")
    ap.add_argument("--prod", nargs="*", default=None,
                    help="«тег:подпись» ..., первый — прод-модель: 8996 и 10.8k (raw, kr) и отказ на 2.2k для порога")
    ap.add_argument("--variants", default="raw,rerank,kr")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="теги прогонов --suite: печатает таблицу моделей рядом и выходит")
    ap.add_argument("--tag", default="student")
    ap.add_argument("--ladder", nargs="*", default=[])
    ap.add_argument("--team-crops", default=None, help="тег прогона --team-clean-like на кропах")
    ap.add_argument("--team-frames", default=None, help="тег прогона --team-clean-like на полных кадрах")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.prod:
        text = "\n".join(prod_table(a.prod))
        print(text)
        if a.out:
            a.out.write_text(text + "\n", encoding="utf-8")
        return 0
    if a.models:
        text = "\n".join(models_table(a.models, a.variants.split(",")))
        print(text)
        if a.out:
            a.out.write_text(text + "\n", encoding="utf-8")
        return 0
    if a.compare:
        text = "\n".join(compare_suites(a.compare))
        print(text)
        if a.out:
            a.out.write_text(text + "\n", encoding="utf-8")
        return 0
    r = load(a.tag)
    lines = [f"# Честная валидация: {a.tag}", "",
             f"Порог отказа {r['threshold']} (сырой косинус), {r['folds']} фолдов по машинам, "
             f"open-set {r['openset_frac']:.0%}, holdout {r['holdout_vehicles']} машин "
             f"({r['gallery_frames']} gallery, {r['query_frames']} query, исключено кадров {r['excluded_frames']}), "
             f"дистракторов в big-галерее {r['extra_distractors']}.", "",
             "## Старый протокол (сплит make_split.py, одна галерея, эталон)", "", *old_table(r), "",
             "## Честный протокол (среднее ± разброс по фолдам)", "", *main_table(r), ""]
    if a.ladder:
        lines += ["## Размер галереи (сырой косинус)", "", *ladder_table(a.ladder), ""]
    if a.team_crops and a.team_frames:
        lines += [*team_compare({"crops": a.team_crops, "frames": a.team_frames}), ""]
        notes = RES / "team_parity_notes.md"          # выводы пишутся руками, таблица — из JSON
        if notes.exists():
            lines += [notes.read_text(encoding="utf-8").rstrip(), ""]
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
