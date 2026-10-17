#!/usr/bin/env python3
"""Оценка уверенности для отказа: какая устойчива к размеру галереи.

Кандидат всегда один и тот же — top-1 по сырому косинусу (как /v1/search);
меняется только число, с которым сравнивается порог:

  raw      s1 — сырой косинус top-1 (сейчас в сервисе);
  gap      s1 − s2 — отрыв от второго кадра;
  z50      (s1 − mean(s2..s50)) / std(s2..s50) — top-1 в единицах разброса
           «фона» этого же запроса;
  gap_vid  s1 − v2 — отрыв от лучшей ДРУГОЙ машины: v_k — максимум косинуса
           по кадрам k-й машины галереи (кадры той же машины с других камер
           фоном не считаются);
  z50_vid  (s1 − mean(v2..v50)) / std(v2..v50) — z по другим машинам.

Всё считается по одному запросу и галерее (vehicle_id галереи сервис знает),
то есть годится для онлайна.

Протокол — research/honest_eval.py: 5 фолдов по 150 машинам holdout, 25 % машин
фолда без пары, режим «без почти-дублей» (кадры той же машины с той же камеры
из галереи убраны). Метрики — research/official_evaluate.candidate_metrics:
TP — у запроса есть пара, ответ дан, top-1 верный; FP — ответ дан, но top-1
неверен или пары нет; FN — пара есть, а отказ; TN — пары нет, и отказ;
F1 = 2PR/(P+R), TNR = TN/(TN + FP без пары). PR-AUC — эталонный pr_auc по
метке «у запроса есть пара»: «эталон» — со скором −inf у отказанных (так
посчитает эталон по candidates.csv), «без порога» — по всем top-1.

Порог: на 2.2k-записях фолда f выбирается F1-оптимальный, проверяется на
остальных четырёх фолдах — на 2.2k и без изменений на других галереях; так
для всех 5 фолдов. Пороги 0.8025 и 0.70 (только raw) проверяются на тех же
наборах. 8996 — одна галерея состава GENERALIZATION_RESULTS.md: оценивается на
запросах машин не из фолда f, пар нет только у 7 запросов (TNR очень шумный).

    python research/refusal_score.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate_refusal_folds as crf  # noqa: E402
import honest_eval as he  # noqa: E402
import official_evaluate as oe  # noqa: E402
import service_embeddings as se  # noqa: E402

SCORES = {"raw": "s1 — сырой косинус top-1",
          "gap": "s1 − s2 — отрыв от второго кадра",
          "z50": "(s1 − mean(s2..s50)) / std(s2..s50)",
          "gap_vid": "s1 − v2 — отрыв от лучшей другой машины",
          "z50_vid": "(s1 − mean(v2..v50)) / std(v2..v50) — z по другим машинам",
          "z20": "(s1 − mean(s2..s20)) / std(s2..s20) — проверка окна",
          "z100": "(s1 − mean(s2..s100)) / std(s2..s100) — проверка окна"}
LADDER = (("none", "только holdout"), ("test_gallery", "+ test gallery"), ("test_all", "2.2k"),
          ("train_seen", "+ обучающие"), ("test_all+train_seen", "10.8k"))
REPORT = (("2.2k", "test_all"), ("8996", "team8996"), ("10.8k", "test_all+train_seen"))
FIXED = (("raw", 0.8025), ("raw", 0.70))
TOP = 100
TNR_FLOOR = 0.70          # правило «максимум F1 при TNR ≥ 0.70 на фолде выбора»


def top_desc(x: np.ndarray, k: int) -> np.ndarray:
    """k наибольших по строке, по убыванию (−inf — для коротких строк)."""
    k = min(k, x.shape[1])
    part = -np.partition(-x, k - 1, axis=1)[:, :k]
    return -np.sort(-part, axis=1)


def z_of(top: np.ndarray) -> np.ndarray:
    """(s1 − mean(s2..)) / std(s2..) по конечным значениям строки."""
    bg = np.where(np.isfinite(top[:, 1:]), top[:, 1:], np.nan)
    mu, sd = np.nanmean(bg, axis=1), np.nanstd(bg, axis=1)
    return (top[:, 0] - mu) / np.maximum(sd, 1e-6)


def block(D: he.Data, q_rows: list, g_rows: list) -> dict:
    """Записи запросов против одной галереи, режим без почти-дублей."""
    Q, G = he.frame(q_rows), he.frame(g_rows)
    sim = D.vectors(list(Q.index)) @ D.vectors(list(G.index)).T
    mask = he.mode_mask(Q, G, "nodup")
    sim = np.where(mask, -np.inf, sim)
    q_vid = Q.vehicle_id.to_numpy()
    g_vid = G.vehicle_id.to_numpy()
    top1 = sim.argmax(1)
    top = top_desc(sim, TOP)                     # s1..s100
    codes, uniq = pd.factorize(G.vehicle_id)
    order = np.argsort(codes, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(codes[order]) != 0])
    vmax = np.maximum.reduceat(sim[:, order], starts, axis=1)      # запрос × машина галереи
    vtop = top_desc(vmax, TOP)                   # v1..v100
    has = ((g_vid[None, :] == q_vid[:, None]) & ~mask).any(1)
    conf = {"raw": top[:, 0], "gap": top[:, 0] - top[:, 1], "z50": z_of(top[:, :50]),
            "gap_vid": vtop[:, 0] - vtop[:, 1], "z50_vid": z_of(vtop[:, :50]),
            "z20": z_of(top[:, :20]), "z100": z_of(top[:, :100])}
    m = he.full_metrics(sim, Q, G)
    return {"conf": conf, "has_match": has, "correct": g_vid[top1] == q_vid, "vehicle": q_vid,
            "mINP": m["mINP"], "n_gallery": len(g_rows)}


def gallery_blocks(D: he.Data, kind: str, folds: list) -> list:
    """Блоки по фолдам: те же open-set и галерея, что honest_eval.run_folds (big)."""
    extra = D.valid(D.distractors(kind))
    qry, gal = D.valid(D.qry), D.valid(D.gal)
    rng = np.random.default_rng(D.a.seed + 1)
    out = []
    for fvids in folds:
        fset = set(fvids)
        n_open = int(round(len(fvids) * D.a.openset_frac))
        opened = set(rng.choice(sorted(fvids, key=int), size=n_open, replace=False).tolist())
        out.append(block(D, [r for r in qry if r["vehicle_id"] in fset],
                         [r for r in gal if r["vehicle_id"] not in opened] + extra))
    return out


def subset(b: dict, keep: np.ndarray, score: str) -> dict:
    return {"conf": b["conf"][score][keep], "has_match": b["has_match"][keep], "correct": b["correct"][keep]}


def merged(blocks: list, score: str, f: int, folds: list) -> dict:
    """Записи всех фолдов, кроме f. Для одной галереи (8996) — запросы машин не из фолда f."""
    if len(blocks) == 1:
        b = blocks[0]
        return subset(b, ~np.isin(b["vehicle"], list(folds[f])), score)
    parts = [subset(b, np.ones(len(b["correct"]), bool), score) for i, b in enumerate(blocks) if i != f]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("conf", "has_match", "correct")}


def choose_f1(rec: dict, tnr_floor: float = None) -> float:
    """F1-оптимальный порог; с tnr_floor — лучший F1 среди порогов с TNR ≥ tnr_floor."""
    values = np.unique(rec["conf"][np.isfinite(rec["conf"])])
    pts = [crf.metrics(rec, float(t)) for t in np.concatenate(([np.inf], values))]
    if tnr_floor is not None:
        pts = [m for m in pts if m["TNR"] == m["TNR"] and m["TNR"] >= tnr_floor] or pts[:1]
    best = max(pts, key=lambda m: (round(m["F1"], 12), m["TNR"] if m["TNR"] == m["TNR"] else -1, m["threshold"]))
    return best["threshold"]


def evaluate(rec: dict, t: float) -> dict:
    m = crf.metrics(rec, t)
    acc = np.isfinite(rec["conf"]) & (rec["conf"] >= t)
    y = rec["has_match"].astype(int)
    m["PR-AUC"] = oe.pr_auc(np.where(acc, rec["conf"], -np.inf), y)
    m["PR-AUC_all"] = oe.pr_auc(np.where(np.isfinite(rec["conf"]), rec["conf"], -np.inf), y)
    return m


def stats(vals: list) -> dict:
    v = np.array([x for x in vals if x == x], dtype="float64")
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
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
    blocks = {kind: gallery_blocks(D, kind, folds) for kind, _ in LADDER}
    tq, tg = he.team8996(D)
    blocks["team8996"] = [block(D, D.valid(tq), D.valid(tg))]

    # сверка: raw-записи 2.2k совпадают с calibrate_refusal_folds.fold_records
    ref = crf.fold_records(D, "test_all", "nodup")
    for b, r in zip(blocks["test_all"], ref):
        for k_mine, k_ref in (("has_match", "has_match"), ("correct", "correct")):
            if not np.array_equal(b[k_mine], r[k_ref]):
                raise AssertionError(f"{k_mine} не совпадает с calibrate_refusal_folds")
        if not np.allclose(b["conf"]["raw"], r["conf"]):
            raise AssertionError("raw top-1 не совпадает с calibrate_refusal_folds")

    rules = [(s, "f1") for s in SCORES] + [(s, "f1_tnr") for s in SCORES] + [(s, t) for s, t in FIXED]
    kinds = [k for k, _ in LADDER] + ["team8996"]
    res = {}
    for score, rule in rules:
        name = f"{score}@{rule if isinstance(rule, str) else f'{rule:g}'}"
        sel = lambda f: subset(blocks["test_all"][f], np.ones(len(blocks["test_all"][f]["correct"]), bool), score)
        thr = [choose_f1(sel(f)) if rule == "f1" else choose_f1(sel(f), TNR_FLOOR) if rule == "f1_tnr"
               else float(rule) for f in range(a.folds)]
        res[name] = {"score": score, "rule": rule, "threshold": stats(thr), "galleries": {}}
        for kind in kinds:
            per = [evaluate(merged(blocks[kind], score, f, folds), thr[f]) for f in range(a.folds)]
            res[name]["galleries"][kind] = {k: stats([p[k] for p in per])
                                            for k in ("F1", "TNR", "recall", "precision", "PR-AUC", "PR-AUC_all")}
            res[name]["galleries"][kind]["n_gallery"] = int(np.mean([b["n_gallery"] for b in blocks[kind]]))
            res[name]["galleries"][kind]["n_openset"] = int(np.mean(
                [(~merged(blocks[kind], score, f, folds)["has_match"]).sum() for f in range(a.folds)]))
    minp = {kind: stats([b["mINP"] for b in blocks[kind]]) for kind in kinds}

    w = {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~") for k, v in D.env.items()}
    report = {"tag": a.tag, "weights": w.get("REID_WEIGHTS", ""), "scores": SCORES, "mINP_nodup": minp, "rules": res}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"refusal_score_{a.tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                           encoding="utf-8")
    for name, r in res.items():
        g = r["galleries"]
        print(f"{name:16s} порог {r['threshold']['mean']:.4f}±{r['threshold']['std']:.4f}  " + "  ".join(
            f"{lbl}: F1 {g[k]['F1']['mean']:.3f} TNR {g[k]['TNR']['mean']:.3f} R {g[k]['recall']['mean']:.3f}"
            for lbl, k in REPORT))
    plot(report, a.out_dir / f"refusal_score_{a.tag}.png", a.tag)
    (a.out_dir / f"refusal_score_{a.tag}.md").write_text(markdown(report), encoding="utf-8")
    print(f"-> {a.out_dir / f'refusal_score_{a.tag}.md'}")
    return 0


# --------------------------------------------------------------------------
# отчёт и картинка
# --------------------------------------------------------------------------

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# имя правила, подпись, цвет, стиль, маркер, сдвиг подписи конца на панели F1 (None — без подписи)
LINES = (("raw@0.7", "сырой косинус, порог 0.70", "#eb6834", "-", "o", (10, 10)),
         ("z100@f1_tnr", "z100, F1 при TNR ≥ 0.70", "#2a78d6", "-", "D", (10, -12)),
         ("z100@f1", "z100, F1-оптимальный", "#86b6ef", (0, (5, 4)), "^", None))
FONT = "PT Sans"


def plot(report: dict, path: Path, tag: str) -> None:
    r = report["rules"]
    kinds = [k for k, _ in LADDER]
    xs = [r["raw@0.7"]["galleries"][k]["n_gallery"] for k in kinds]
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    fig.text(0.04, 0.94, "F1 и TNR отказа против размера галереи", fontsize=32, color=INK, family=FONT,
             weight="bold", va="top")
    fig.text(0.04, 0.865, f"Прод-модель {tag}, без почти-дублей, 5 фолдов. Порог выбран на 2.2k и без изменений "
             "проверен на других галереях.\nz100 = (s1 − mean(s2..s100)) / std(s2..s100): top-1 в единицах "
             "разброса ближайших соседей того же запроса.",
             fontsize=17, color=INK2, family=FONT, va="top", linespacing=1.4)
    for k, (metric, title) in enumerate((("F1", "F1, основная метрика ТЗ"), ("TNR", "TNR, отказы на запросах без пары"))):
        ax = fig.add_axes([0.07 + k * 0.48, 0.13, 0.4, 0.56])
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=15, length=0)
        for name, label, color, ls, marker, off in LINES:
            y = [r[name]["galleries"][kk][metric]["mean"] for kk in kinds]
            ax.plot(xs, y, color=color, lw=3.5, ls=ls, marker=marker, ms=10, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=label, zorder=3)
            if metric == "TNR" or off:
                ax.annotate(f"{y[-1]:.2f}", (xs[-1], y[-1]), xytext=off if metric == "F1" else (10, 0),
                            textcoords="offset points", fontsize=15, color=color, family=FONT, va="center",
                            weight="bold")
        ax.set_xscale("log")
        ax.set_xlim(250, 17000)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:,}".replace(",", " ") for x in xs], fontsize=14, family=FONT, color=INK2)
        ax.minorticks_off()
        ticks = ax.get_xticklabels()
        ticks[-2].set_ha("right")
        ticks[-1].set_ha("left")
        for lab in ax.get_yticklabels():
            lab.set_family(FONT)
        ax.set_xlabel("Кадров в галерее (логарифмическая шкала)", fontsize=17, color=INK2, family=FONT, labelpad=8)
        ax.set_title(title, fontsize=21, color=INK, family=FONT, weight="bold", loc="left", pad=12)
        if k == 1:
            ax.legend(loc="lower left", frameon=False, prop={"family": FONT, "size": 14}, labelcolor=INK2)
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def markdown(report: dict) -> str:
    r, mi = report["rules"], report["mINP_nodup"]
    g = lambda name, k, m: r[name]["galleries"][k][m]["mean"]
    sd = lambda name, k, m: r[name]["galleries"][k][m]["std"]
    thr = lambda name: f"{r[name]['threshold']['mean']:.3f} ± {r[name]['threshold']['std']:.3f}" \
        if r[name]["rule"] in ("f1", "f1_tnr") else f"{r[name]['threshold']['mean']:g}"
    cols = [lbl for lbl, _ in REPORT]
    head = "| Правило | Порог | " + " | ".join(f"{c}: F1 / TNR / Recall / Precision" for c in cols) + " |"
    sep = "|---|---:|" + "---:|" * len(cols)
    lines = [f"# Отказ для {report['tag']}: метрика ТЗ и размер галереи", "",
             f"Веса `{report['weights']}`. Воспроизвести: `python research/refusal_score.py --data-dir <данные> "
             f"--env REID_WEIGHTS=<веса> --tag {report['tag']}` (минута на кэшированных векторах).", "",
             "## Как считается F1 в режиме отказа", "",
             "Как в эталоне организаторов (`research/official_evaluate.candidate_metrics`): решение принимается "
             "на уровне запроса, кандидат — top-1 по сырому косинусу, ответ даётся, если его оценка уверенности ≥ "
             "порога.", "",
             "- **TP** — у запроса есть пара в галерее (кадр той же машины с другой камеры), ответ дан и top-1 верный;",
             "- **FP** — ответ дан, но top-1 неверный, или пары у запроса нет вовсе;",
             "- **FN** — пара есть, а сервис отказал; **TN** — пары нет, и сервис отказал.",
             "- Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2PR / (P + R); "
             "TNR = TN / (TN + FP без пары).",
             "- Позитив F1 — «ответ дан и он верный». Принятый неверный top-1 у запроса с парой — FP, но не FN: "
             "отказ на таком запросе превращает FP в FN.",
             "- PR-AUC — эталонный `pr_auc` по метке «у запроса есть пара»: «эталон» — отказанным ставится −inf (так "
             "эталон посчитает по `candidates.csv`), «без порога» — по всем top-1. mINP от порога не зависит.",
             "- Протокол: 5 фолдов по 150 машинам holdout, 25 % машин фолда без пары, режим «без почти-дублей». "
             "Порог выбирается на 2.2k-записях одного фолда и проверяется на остальных четырёх — на 2.2k и без "
             "изменений на других галереях; ± — разброс по 5 повторам. 8996 — одна галерея состава "
             f"GENERALIZATION_RESULTS.md: без пары в каждой проверке в среднем {r['raw@0.7']['galleries']['team8996']['n_openset']} "
             "запросов, TNR там почти не информативен.", "",
             "## 1. Пороги 0.8025, 0.70 и F1-оптимальный (сырой косинус)", "", head, sep]
    for name, label in (("raw@0.8025", "0.8025 (сейчас в сервисе)"), ("raw@0.7", "0.70 (рабочая точка)"),
                        ("raw@f1", "F1-оптимальный вне фолда"), ("raw@f1_tnr", "F1-оптимальный при TNR ≥ 0.70")):
        cells = [f"{g(name, k, 'F1'):.3f} / {g(name, k, 'TNR'):.3f} / {g(name, k, 'recall'):.3f} / "
                 f"{g(name, k, 'precision'):.3f}" for _, k in REPORT]
        lines.append(f"| {label} | {thr(name)} | " + " | ".join(cells) + " |")
    lines += ["", "| | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols),
              "| PR-AUC эталона при 0.70 | " + " | ".join(f"{g('raw@0.7', k, 'PR-AUC'):.3f}" for _, k in REPORT) + " |",
              "| PR-AUC без порога (сырой косинус) | " + " | ".join(f"{g('raw@0.7', k, 'PR-AUC_all'):.3f}"
                                                              for _, k in REPORT) + " |",
              "| mINP (без почти-дублей, от порога не зависит) | " + " | ".join(f"{mi[k]['mean']:.3f}"
                                                                         for _, k in REPORT) + " |", "",
              f"На 8996 без пары в среднем {r['raw@0.7']['galleries']['team8996']['n_openset']} запросов против "
              f"{r['raw@0.7']['galleries']['test_all']['n_openset']} на 2.2k и 10.8k: PR-AUC там ≈ 1 и TNR "
              "почти ничего не говорят, F1 по ~545 запросам надёжен.", ""]
    lines += ["## 2. Какая оценка уверенности держится при росте галереи", "",
              "Порог каждой оценки выбран на 2.2k (вне фолда) и без изменений применён к большим галереям. "
              "Δ — 10.8k минус 2.2k.", "",
              "| Оценка | Правило | Порог | 2.2k F1 / TNR | 8996 F1 / TNR | 10.8k F1 / TNR | ΔF1 | ΔTNR | "
              "PR-AUC без порога 2.2k → 10.8k |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for score in SCORES:
        for rule, rlabel in (("f1", "F1-оптимум"), ("f1_tnr", "F1 при TNR ≥ 0.70")):
            n = f"{score}@{rule}"
            lines.append(
                f"| {score} | {rlabel} | {thr(n)} | " + " | ".join(
                    f"{g(n, k, 'F1'):.3f} / {g(n, k, 'TNR'):.3f}" for _, k in REPORT) +
                f" | {g(n, 'test_all+train_seen', 'F1') - g(n, 'test_all', 'F1'):+.3f} | "
                f"{g(n, 'test_all+train_seen', 'TNR') - g(n, 'test_all', 'TNR'):+.3f} | "
                f"{g(n, 'test_all', 'PR-AUC_all'):.3f} → {g(n, 'test_all+train_seen', 'PR-AUC_all'):.3f} |")
    lines += ["", "Оценки: " + "; ".join(f"`{k}` — {v}" for k, v in SCORES.items()) + ".", "",
              "## 3. Лестница размеров для картинки (`refusal_score_" + report["tag"] + ".png`)", "",
              "| Правило | " + " | ".join(f"{r['raw@0.7']['galleries'][k]['n_gallery']} кадров" for k, _ in LADDER)
              + " |", "|---|" + "---:|" * len(LADDER)]
    for name, label, *_ in (LINES[0], ("raw@f1_tnr", "сырой косинус, F1 при TNR ≥ 0.70"), *LINES[1:]):
        lines.append(f"| {label} | " + " | ".join(f"{g(name, k, 'F1'):.3f} / {g(name, k, 'TNR'):.3f}"
                                                for k, _ in LADDER) + " |")
    n70, z = "raw@0.7", "z100@f1_tnr"
    b, t = "test_all", "test_all+train_seen"
    dF1 = [g(n, t, "F1") - g(n, b, "F1") for n in r if r[n]["rule"] in ("f1", "f1_tnr")] + [g(n70, t, "F1") - g(n70, b, "F1")]
    lines += ["", "Ячейка — F1 / TNR.", "", "## Выводы", "",
              f"- **F1 не держит ни одна оценка**: от 2.2k к 10.8k F1 падает на {-max(dF1):.2f}–{-min(dF1):.2f} у всех "
              "правил выбора, потому что падает "
              f"само ранжирование (mINP {mi[b]['mean']:.3f} → {mi[t]['mean']:.3f}); порог это не лечит.",
              f"- **TNR сырого косинуса проваливается**: при пороге 0.70 — {g(n70, b, 'TNR'):.3f} на 2.2k, "
              f"{g(n70, t, 'TNR'):.3f} на 10.8k. С ростом галереи максимум похожести у запросов без пары растёт "
              "(максимум из большего числа чужих машин), и фиксированный порог их пропускает.",
              f"- **z100 держит TNR начиная с 2.2k**: {g(z, b, 'TNR'):.3f} → {g(z, 'train_seen', 'TNR'):.3f} → "
              f"{g(z, t, 'TNR'):.3f} (правило «F1 при TNR ≥ 0.70»), окна s2..s20 и s2..s50 ведут себя так же. "
              f"Но на маленьких галереях TNR у z100 ниже: {g(z, 'none', 'TNR'):.3f} при "
              f"{r[z]['galleries']['none']['n_gallery']} кадрах против {g(n70, 'none', 'TNR'):.3f} у сырого "
              "косинуса — фона из ближайших соседей там мало.",
              f"- **Цена z100 — F1 на больших галереях**: на 10.8k {g(z, t, 'F1'):.3f} против {g(n70, t, 'F1'):.3f} у "
              f"0.70 (recall {g(z, t, 'recall'):.3f} против {g(n70, t, 'recall'):.3f}); на 2.2k F1 почти равны "
              f"({g(z, b, 'F1'):.3f} против {g(n70, b, 'F1'):.3f}). По PR-AUC без порога z100 лучше и падает "
              f"меньше: {g(z, b, 'PR-AUC_all'):.3f} → {g(z, t, 'PR-AUC_all'):.3f} против "
              f"{g(n70, b, 'PR-AUC_all'):.3f} → {g(n70, t, 'PR-AUC_all'):.3f}.",
              "- **Почему z стабильнее по TNR.** У запроса без пары s1 — максимум из N похожестей на чужие машины. "
              "С ростом N весь хвост распределения сдвигается вверх, поэтому сырой s1 растёт. Отрыв максимума от "
              "ближайших 100 соседей в единицах их же разброса для хвостов типа Гумбеля от N почти не зависит: "
              "растут и s1, и фон. У запроса с парой s1 — похожесть на свою машину, она от N не зависит, а фон "
              "растёт, поэтому recall z100 с ростом галереи падает. Это и есть цена.",
              "- **Рекомендация.** Решение сервиса оставить на сыром косинусе с порогом 0.70: на галереях "
              f"1–2 тыс. кадров (тест организаторов — 750 кадров галереи) он лучше по TNR при том же F1 "
              f"({g(n70, 'test_gallery', 'F1'):.3f} / {g(n70, 'test_gallery', 'TNR'):.3f} против "
              f"{g(z, 'test_gallery', 'F1'):.3f} / {g(z, 'test_gallery', 'TNR'):.3f} у z100 на "
              f"{r[n70]['galleries']['test_gallery']['n_gallery']} кадрах). z100 — запасная оценка для галерей от "
              f"~9 тыс. кадров: порог {thr(z)}, в `candidates.csv` confidence = z100 (больше — увереннее, отказ — "
              "пустой список), считается по одному запросу из его top-100 без других запросов.",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
