#!/usr/bin/env python3
"""Разбор провалов модели на валидации с картинками.

    python error_analysis.py --backbone clip_vitb16 --head v1_arc512

Находит запросы, на которых модель ошиблась, раскладывает их по причинам
(ночь, блики, сильный ракурс, похожая машина, мелкий bbox) и рисует коллажи:
запрос, что модель выдала первым, и где на самом деле лежал правильный ответ.

Категории определяются по измеримым признакам кропа, а не на глаз, чтобы
разбор был воспроизводимым.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat

from head_model import ReidHead

HERE = Path(__file__).parent
FEAT = Path("/opt/case17/data/features")
CROPS = Path("/opt/case17/data/crops")
MODELS = Path("/opt/case17/models")


def crop_stats(image_id: str) -> dict:
    """Измеримые характеристики кропа, по которым раскладываем ошибки."""
    p = CROPS / "train" / f"{image_id}.jpg"
    im = Image.open(p).convert("RGB")
    g = im.convert("L")
    st = ImageStat.Stat(g)
    arr = np.asarray(g, dtype="float32")
    return {
        "w": im.width,
        "h": im.height,
        "area": im.width * im.height,
        "aspect": im.width / max(1, im.height),
        "luma": st.mean[0],
        "bright_frac": float((arr > 235).mean()),   # доля пересветов — блики
        "contrast": st.stddev[0],
    }


def categorize(q: dict, g: dict, sim_top1: float, correct_rank: int) -> list:
    """Одна ошибка может попасть в несколько категорий — это нормально."""
    cats = []
    if q["luma"] < 70:
        cats.append("ночь")
    if q["bright_frac"] > 0.02:
        cats.append("блики")
    if abs(q["aspect"] - g["aspect"]) > 0.45:
        cats.append("сильная смена ракурса")
    if q["area"] < 120_000:
        cats.append("мелкий bbox")
    if sim_top1 > 0.55:
        cats.append("похожая машина")
    if not cats:
        cats.append("прочее")
    return cats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip_vitb16")
    ap.add_argument("--head", default="v1_arc512")
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--outdir", default="errors")
    a = ap.parse_args()

    idx = list(csv.DictReader((FEAT / "train_index.csv").open()))
    orig = np.load(FEAT / f"train_{a.backbone}_orig.npy")
    flip = np.load(FEAT / f"train_{a.backbone}_flip.npy")
    val_ids = {l.strip() for l in (HERE / "val_ids.txt").read_text().split() if l.strip()}
    rows = [(i, r) for i, r in enumerate(idx) if r["vehicle_id"] in val_ids]

    head = None
    if a.head:
        ck = torch.load(MODELS / f"head_{a.head}.pt", map_location="cpu", weights_only=False)
        ar = ck["args"]
        head = ReidHead(ar.get("in_dim", orig.shape[1]), ar["hidden"], ar["out_dim"], ar["dropout"])
        head.load_state_dict(ck["head"]); head.eval()

    by_v = defaultdict(lambda: defaultdict(list))
    for i, r in rows:
        by_v[r["vehicle_id"]][r["camera_id"]].append((i, r))
    gal, qry = [], []
    for v in sorted(by_v):
        for cam in sorted(by_v[v], key=int):
            items = sorted(by_v[v][cam], key=lambda t: t[1]["image_id"])
            gal.append(items[0]); qry += items[1:]

    def emb(rs):
        ii = [i for i, _ in rs]
        a_, b_ = orig[ii], flip[ii]
        if head is not None:
            with torch.inference_mode():
                a_ = head(torch.from_numpy(a_))[1].numpy()
                b_ = head(torch.from_numpy(b_))[1].numpy()
        v = (a_ + b_) / 2
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    q_emb, g_emb = emb(qry), emb(gal)
    g_vid = np.array([r["vehicle_id"] for _, r in gal])
    g_cam = np.array([r["camera_id"] for _, r in gal])
    sim = q_emb @ g_emb.T

    failures, n_eval, n_ok = [], 0, 0
    for i, (_, qr) in enumerate(qry):
        keep = g_cam != qr["camera_id"]
        rel = g_vid[keep] == qr["vehicle_id"]
        if not keep.any() or not rel.any():
            continue
        n_eval += 1
        s = sim[i][keep]
        kept = [gal[j] for j in np.flatnonzero(keep)]
        order = np.argsort(-s)
        if rel[order[0]]:
            n_ok += 1
            continue
        correct_rank = int(np.flatnonzero(rel[order])[0]) + 1
        failures.append({
            "query": qr,
            "top1": kept[order[0]][1],
            "top1_sim": float(s[order[0]]),
            "correct": kept[order[np.flatnonzero(rel[order])[0]]][1],
            "correct_sim": float(s[order[np.flatnonzero(rel[order])[0]]]),
            "correct_rank": correct_rank,
        })

    print(f"запросов оценено: {n_eval}, Rank-1 верных: {n_ok} ({n_ok/n_eval*100:.1f}%), "
          f"ошибок: {len(failures)}")

    # раскладываем по категориям
    buckets = defaultdict(list)
    for f in failures:
        qs = crop_stats(f["query"]["image_id"])
        gs = crop_stats(f["correct"]["image_id"])
        f["qstats"], f["gstats"] = qs, gs
        for c in categorize(qs, gs, f["top1_sim"], f["correct_rank"]):
            buckets[c].append(f)

    print("\nошибки по категориям:")
    for c in sorted(buckets, key=lambda k: -len(buckets[k])):
        print(f"  {c:<26} {len(buckets[c]):>4}  ({len(buckets[c])/len(failures)*100:.0f}% ошибок)")

    outdir = HERE / a.outdir
    outdir.mkdir(exist_ok=True)
    made = {}
    for cat, items in buckets.items():
        items = sorted(items, key=lambda f: -f["correct_rank"])[: a.per_category]
        if not items:
            continue
        fig, axes = plt.subplots(len(items), 3, figsize=(10.5, 3.6 * len(items)), squeeze=False)
        for r, f in enumerate(items):
            panes = [
                (f["query"]["image_id"], f"ЗАПРОС\nмашина {f['query']['vehicle_id']}, камера {f['query']['camera_id']}"),
                (f["top1"]["image_id"], f"ВЫДАЛА ПЕРВЫМ (неверно)\nмашина {f['top1']['vehicle_id']}, сходство {f['top1_sim']:.3f}"),
                (f["correct"]["image_id"], f"ПРАВИЛЬНЫЙ ОТВЕТ\nбыл на месте {f['correct_rank']}, сходство {f['correct_sim']:.3f}"),
            ]
            for c, (iid, title) in enumerate(panes):
                ax = axes[r][c]
                ax.imshow(Image.open(CROPS / "train" / f"{iid}.jpg"))
                ax.set_title(title, fontsize=8)
                ax.axis("off")
        fig.suptitle(f"Категория ошибки: {cat}", fontsize=12)
        fig.tight_layout()
        name = f"error_{cat.replace(' ', '_')}.jpg"
        fig.savefig(outdir / name, dpi=110, pil_kwargs={"quality": 86, "optimize": True})
        plt.close(fig)
        made[cat] = name
        print(f"  коллаж: {outdir / name}")

    summary = {
        "backbone": a.backbone, "head": a.head,
        "queries_evaluated": n_eval, "rank1_correct": n_ok,
        "rank1_pct": round(n_ok / n_eval * 100, 2),
        "failures": len(failures),
        "by_category": {c: len(v) for c, v in sorted(buckets.items(), key=lambda kv: -len(kv[1]))},
        "collages": made,
        "median_correct_rank_on_failure": int(np.median([f["correct_rank"] for f in failures])),
        "failures_beyond_top10": sum(1 for f in failures if f["correct_rank"] > 10),
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "error_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
