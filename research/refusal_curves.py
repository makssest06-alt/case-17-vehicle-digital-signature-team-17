#!/usr/bin/env python3
"""PR и ROC порога отказа для слайда: режим «без почти-дублей», 2.2k, 5 фолдов вместе.

Записи и метрики — research/calibrate_refusal_folds.py (fold_records, metrics):
семантика эталона, TP = принят И верхний кандидат верный, TNR — по запросам
без пары. ROC: по горизонтали доля принятых запросов без пары (1 − TNR), по
вертикали recall. Кривые и точки посчитаны по всем фолдам сразу; средние по
фолдам — в honest_eval_results/refusal_folds.json.

    python research/refusal_curves.py --data-dir /opt/case17/data
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate_refusal_folds as crf  # noqa: E402
import honest_eval as he  # noqa: E402
import service_embeddings as se  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CURVE, NOW = "#2a78d6", "#eb6834"
FONT = "PT Sans"
# порог, подпись, цвет, маркер, сдвиг подписи на PR и на ROC (точки экрана)
POINTS = ((0.8025, "порог 0.8025 (сервис)", NOW, "o", (-40, -22), (30, 8)),
          (0.71, "порог 0.71", INK, "s", (-20, 26), (14, -10)),
          (0.874, "порог 0.874 (строгий)", MUTED, "D", (12, 30), (14, 4)))


def curve(rec: dict) -> list:
    values = np.unique(rec["conf"][np.isfinite(rec["conf"])])
    return [crf.metrics(rec, float(t)) for t in np.concatenate(([np.inf], values[::-1], [-np.inf]))]


def area(x, y) -> float:
    order = np.argsort(x, kind="stable")
    return float(np.trapezoid(np.asarray(y)[order], np.asarray(x)[order]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--distractors", default="test_all", choices=tuple(he.DISTRACTORS))
    ap.add_argument("--out", type=Path, default=HERE / "honest_eval_results" / "refusal_pr_roc.png")
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

    D = he.Data(a)
    recs = crf.fold_records(D, a.distractors, "nodup")
    crf.self_check(recs, 0.8025)
    rec = crf.merge(recs)
    pts = curve(rec)
    rec_, prec, tnr = (np.array([p[k] for p in pts], dtype="float64") for k in ("recall", "precision", "TNR"))
    fpr = 1 - tnr
    roc_auc = area(fpr, rec_)
    pr_auc = area(rec_[1:], prec[1:])           # без точки «ничего не принято»: precision там не определён
    marks = {t: crf.metrics(rec, t) for t, *_ in POINTS}
    n_pos, n_neg = int(rec["has_match"].sum()), int((~rec["has_match"]).sum())

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    fig.text(0.04, 0.94, "Порог отказа: что даёт каждая точка", fontsize=32, color=INK, family=FONT,
             weight="bold", va="top")
    fig.text(0.04, 0.865, f"Студент из main, галерея 2.2k, без почти-дублей, 5 фолдов вместе: {n_pos} запросов "
             f"с парой, {n_neg} без пары.\nRecall: запрос принят и первый кандидат верный, как в эталоне "
             "организаторов.", fontsize=17, color=INK2, family=FONT, va="top", linespacing=1.4)
    panels = (("PR: точность против полноты", rec_[1:], prec[1:], "Recall", "Precision", f"площадь {pr_auc:.2f}"),
              ("ROC: полнота против ложных принятий", fpr, rec_, "Принято запросов без пары (1 − TNR)",
               "Recall", f"площадь {roc_auc:.2f}"))
    for k, (title, x, y, xl, yl, auc) in enumerate(panels):
        ax = fig.add_axes([0.07 + k * 0.48, 0.12, 0.4, 0.57])
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=16, length=0)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_family(FONT)
        if k == 1:
            ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls=(0, (4, 4)))
        ax.plot(x, y, color=CURVE, lw=3.5, drawstyle="steps-post" if k == 1 else "default")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xl, fontsize=19, color=INK2, family=FONT, labelpad=10)
        ax.set_ylabel(yl, fontsize=19, color=INK2, family=FONT, labelpad=10)
        ax.set_title(title, fontsize=22, color=INK, family=FONT, weight="bold", loc="left", pad=14)
        ax.text(0.98, 0.04, auc, transform=ax.transAxes, fontsize=17, color=CURVE, family=FONT,
                ha="right", va="bottom")
        for t, label, color, marker, off_pr, off_roc in POINTS:
            m = marks[t]
            px, py = (m["recall"], m["precision"]) if k == 0 else (1 - m["TNR"], m["recall"])
            ax.scatter([px], [py], s=230, color=color, marker=marker, zorder=5, edgecolor=SURFACE, linewidth=2)
            second = f"precision {m['precision']:.2f}" if k == 0 else f"TNR {m['TNR']:.2f}"
            dx, dy = off_pr if k == 0 else off_roc
            ax.annotate(f"{label}\nrecall {m['recall']:.2f}, {second}", (px, py), xytext=(dx, dy),
                        textcoords="offset points", fontsize=15, color=INK if color != MUTED else INK2,
                        family=FONT, va="top" if dy < 0 else "bottom", linespacing=1.3, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.92))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    summary = {"mode": "nodup", "distractors": a.distractors, "pooled_folds": len(recs), "queries_with_pair": n_pos,
               "queries_without_pair": n_neg, "PR_area": round(pr_auc, 4), "ROC_area": round(roc_auc, 4),
               "points": {str(t): {k: (round(v, 4) if isinstance(v, float) else v) for k, v in marks[t].items()}
                          for t, *_ in POINTS}}
    a.out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {a.out} ({a.out.stat().st_size / 1e6:.2f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
