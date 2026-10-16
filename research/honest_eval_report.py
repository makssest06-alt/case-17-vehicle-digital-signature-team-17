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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="student")
    ap.add_argument("--ladder", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
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
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
