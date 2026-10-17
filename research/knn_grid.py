#!/usr/bin/env python3
"""Сетка knn (k × α) для одной модели на 2.2k и 10.8k — одной таблицей для Андрея.

Те же фолды, open-set и метрики, что research/honest_eval.py (run_folds),
вариант knn — honest_eval.knn_expand. Raw считается один раз и стоит строкой
сверху. Отказ — top-1 по варианту, решение по сырому косинусу против --thr,
режим «без почти-дублей». Параметры выбираются на тех же 150 машинах, что и
меряются: лучшая клетка — подбор в выборке, её надо проверить отдельно.

    python research/knn_grid.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10 \\
        --ks 2,3,5 --alphas 1,3 --thr 0.70
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import honest_eval as he  # noqa: E402
import service_embeddings as se  # noqa: E402

GALLERIES = (("2.2k", "test_all"), ("10.8k", "test_all+train_seen"))


def cell(s: dict) -> dict:
    o, n = s["official"], s["nodup"]
    return {"mAP@10": o["mAP@10"], "Rank-1": o["Rank-1"]["mean"], "Recall": n["Recall"]["mean"],
            "TNR": n["TNR"]["mean"], "n_gallery": int(round(o["n_gallery"]["mean"]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ks", default="2,3,5")
    ap.add_argument("--alphas", default="1,3")
    ap.add_argument("--thr", type=float, default=None, help="порог отказа по сырому косинусу; по умолчанию профиля")
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
    ap.add_argument("--out-dir", type=Path, default=HERE / "honest_eval_results")
    a = ap.parse_args()

    t0 = time.time()
    D = he.Data(a)
    thr = a.thr if a.thr is not None else float(D.env.get("SEARCH_THRESHOLD", "nan"))
    ks = [int(k) for k in a.ks.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    base = he.pp_params(a.pp_params)
    rows = {}
    for gname, kind in GALLERIES:
        s, _, _ = he.run_folds(D, kind, ["raw"], base, thr, galleries=("big",))
        rows.setdefault("raw", {})[gname] = cell({m: s[f"big/raw/{m}"] for m in ("official", "nodup")})
        for k in ks:
            for al in alphas:
                p = {**base, "knn_k": k, "knn_alpha": al}
                s, _, _ = he.run_folds(D, kind, ["knn"], p, thr, galleries=("big",))
                rows.setdefault(f"knn k={k} α={al:g}", {})[gname] = cell({m: s[f"big/knn/{m}"] for m in ("official", "nodup")})
    took = time.time() - t0

    w = {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~") for k, v in D.env.items()}
    report = {"tag": a.tag, "weights": w.get("REID_WEIGHTS", ""), "threshold": thr, "ks": ks, "alphas": alphas,
              "seconds": round(took), "rows": rows}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"knn_grid_{a.tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    g = [gname for gname, _ in GALLERIES]
    raw = rows["raw"]
    lines = [f"# knn для {a.tag}: сетка k × α", "",
             f"Веса `{report['weights']}`, 5 фолдов по 150 машинам holdout, режим эталона для mAP@10 / Rank-1; "
             f"Recall / TNR — без почти-дублей при пороге {thr} (кандидат — top-1 варианта, решение по сырому "
             "косинусу). knn — α-QE по k ближайшим кадрам галереи (`honest_eval.knn_expand`). Δ — к raw.", "",
             f"Воспроизвести: `python research/knn_grid.py --data-dir <данные> --env REID_WEIGHTS=<веса> --tag {a.tag} "
             f"--ks {a.ks} --alphas {a.alphas} --thr {thr}`; счёт занял {took / 60:.1f} мин.", "",
             "| Вариант | " + " | ".join(f"{x}: mAP@10 (Δ) | {x}: R1 | {x}: Recall / TNR" for x in g) + " |",
             "|---|" + "---:|---:|---:|" * len(g)]
    for name, by in rows.items():
        cells = []
        for x in g:
            c, r = by[x], raw[x]
            d = "" if name == "raw" else f" ({c['mAP@10']['mean'] - r['mAP@10']['mean']:+.3f})"
            cells += [f"{c['mAP@10']['mean']:.3f} ± {c['mAP@10']['std']:.3f}{d}", f"{c['Rank-1']:.3f}",
                      f"{c['Recall']:.3f} / {c['TNR']:.3f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    best = {x: max((n for n in rows if n != "raw"), key=lambda n: rows[n][x]["mAP@10"]["mean"]) for x in g}
    lines += ["", "Лучшая клетка по mAP@10 (подбор в выборке): "
              + "; ".join(f"{x} — {best[x]} ({rows[best[x]][x]['mAP@10']['mean']:.3f} против raw "
                          f"{raw[x]['mAP@10']['mean']:.3f})" for x in g) + "."]
    text = "\n".join(lines) + "\n"
    (a.out_dir / f"knn_grid_{a.tag}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
