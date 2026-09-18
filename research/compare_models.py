#!/usr/bin/env python3
"""Сводная таблица по всем вариантам: качество, отказ и отдельно ракурс.

    python compare_models.py
    python compare_models.py --variants siglip_base:s1_arc512 clip_vitb16:v1_arc512

Считает на одном и том же валидационном сплите:
  * mAP, Rank-1, Rank-5 — кросс-камерно, как у жюри;
  * AUC, F1, TNR режима отказа в сбалансированной точке;
  * **долю катастрофических провалов** — запросов, где правильный ответ
    улетел дальше 50 места. Это и есть числовое выражение проблемы ракурса,
    которую мы заявили в анализе ошибок: обещали починить — показываем, как.

Вариант вида `бэкбон:голова` или просто `бэкбон` для zero-shot.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from head_model import ReidHead

HERE = Path(__file__).parent
FEAT = Path("/opt/case17/data/features")
MODELS = Path("/opt/case17/models")


def load_split(backbone: str):
    idx = list(csv.DictReader((FEAT / "train_index.csv").open()))
    orig = np.load(FEAT / f"train_{backbone}_orig.npy")
    flip = np.load(FEAT / f"train_{backbone}_flip.npy")
    val_ids = {l.strip() for l in (HERE / "val_ids.txt").read_text().split() if l.strip()}
    rows = [(i, r) for i, r in enumerate(idx) if r["vehicle_id"] in val_ids]

    by = defaultdict(lambda: defaultdict(list))
    for i, r in rows:
        by[r["vehicle_id"]][r["camera_id"]].append((i, r))
    gal, qry = [], []
    for v in sorted(by):
        for cam in sorted(by[v], key=int):
            items = sorted(by[v][cam], key=lambda t: t[1]["image_id"])
            gal.append(items[0]); qry += items[1:]
    return orig, flip, qry, gal


def embed(orig, flip, rows, head):
    ii = [i for i, _ in rows]
    a, b = orig[ii], flip[ii]
    if head is not None:
        with torch.inference_mode():
            a = head(torch.from_numpy(a))[1].numpy()
            b = head(torch.from_numpy(b))[1].numpy()
    v = (a + b) / 2
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load_head(tag: str, in_dim: int):
    if not tag:
        return None
    ck = torch.load(MODELS / f"head_{tag}.pt", map_location="cpu", weights_only=False)
    ar = ck["args"]
    h = ReidHead(ar.get("in_dim", in_dim), ar["hidden"], ar["out_dim"], ar["dropout"])
    h.load_state_dict(ck["head"]); h.eval()
    return h


def evaluate(backbone: str, head_tag: str, seed: int = 42) -> dict:
    orig, flip, qry, gal = load_split(backbone)
    head = load_head(head_tag, orig.shape[1])
    q, g = embed(orig, flip, qry, head), embed(orig, flip, gal, head)
    g_vid = np.array([r["vehicle_id"] for _, r in gal])
    g_cam = np.array([r["camera_id"] for _, r in gal])
    sim = q @ g.T

    aps, r1, r5, ranks = [], 0, 0, []
    for i, (_, qr) in enumerate(qry):
        keep = g_cam != qr["camera_id"]
        if not keep.any():
            continue
        rel = g_vid[keep] == qr["vehicle_id"]
        if not rel.any():
            continue
        s = sim[i][keep]; order = np.argsort(-s); rs = rel[order]
        hits = np.flatnonzero(rs)
        aps.append(float(((np.arange(len(hits)) + 1) / (hits + 1)).mean()))
        r1 += int(rs[0]); r5 += int(rs[:5].any())
        ranks.append(int(hits[0]) + 1)
    ranks = np.array(ranks); used = len(ranks)

    # --- режим отказа ---
    rng = np.random.default_rng(seed)
    vids = sorted({r["vehicle_id"] for _, r in qry})
    unknown = set(rng.choice(vids, size=len(vids) // 2, replace=False).tolist())
    pos, neg = [], []
    for i, (_, qr) in enumerate(qry):
        v = qr["vehicle_id"]
        keep = g_cam != qr["camera_id"]
        if v in unknown:
            keep = keep & (g_vid != v)
        if not keep.any():
            continue
        (neg if v in unknown else pos).append(float(sim[i][keep].max()))
    pos, neg = np.array(pos), np.array(neg)
    allv = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(allv); rk = np.empty(len(allv)); rk[o] = np.arange(1, len(allv) + 1)
    auc = float((rk[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    best_j, best = -9.0, {}
    for t in np.arange(allv.min(), allv.max(), 0.002):
        tp, fp = int((pos >= t).sum()), int((neg >= t).sum())
        fn, tn = int((pos < t).sum()), int((neg < t).sum())
        if tp == 0:
            continue
        pr, rc, tnr = tp / (tp + fp), tp / (tp + fn), tn / (tn + fp)
        j = rc + tnr - 1
        if j > best_j:
            best_j = j
            best = {"threshold": round(float(t), 3), "f1": round(2 * pr * rc / (pr + rc), 3),
                    "recall": round(rc, 3), "TNR": round(tnr, 3)}

    return {
        "variant": f"{backbone}:{head_tag}" if head_tag else f"{backbone} (zero-shot)",
        "backbone": backbone, "head": head_tag or None, "dim": int(q.shape[1]),
        "mAP": round(float(np.mean(aps)) * 100, 2),
        "rank1": round(r1 / used * 100, 2),
        "rank5": round(r5 / used * 100, 2),
        "queries": used,
        "auc": round(auc, 3), **best, "youden": round(best_j, 3),
        # --- ракурс: чем меньше, тем лучше ---
        "top10_pct": round(float((ranks <= 10).mean()) * 100, 1),
        "catastrophic_pct": round(float((ranks > 50).mean()) * 100, 1),
        "median_rank_on_fail": int(np.median(ranks[ranks > 1])) if (ranks > 1).any() else 0,
        "worst_rank": int(ranks.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--out", default="results/comparison.json")
    ap.add_argument("--md", default="results/comparison.md")
    a = ap.parse_args()

    if a.variants:
        pairs = [(v.split(":")[0], v.split(":")[1] if ":" in v else "") for v in a.variants]
    else:  # авто: все бэкбоны с признаками x все головы для них
        pairs = []
        for f in sorted(FEAT.glob("train_*_orig.npy")):
            bb = f.name[len("train_"):-len("_orig.npy")]
            pairs.append((bb, ""))
            for ck in sorted(MODELS.glob("head_*.pt")):
                tag = ck.stem[len("head_"):]
                meta = torch.load(ck, map_location="cpu", weights_only=False)["args"]
                if meta.get("backbone") == bb:
                    pairs.append((bb, tag))

    rows = []
    for bb, tag in pairs:
        try:
            rows.append(evaluate(bb, tag))
            print(f"  посчитан: {rows[-1]['variant']}")
        except FileNotFoundError as e:
            print(f"  пропущен {bb}:{tag} — {e}")

    rows.sort(key=lambda r: -r["mAP"])

    hdr = (f"{'вариант':<34}{'mAP':>7}{'R-1':>7}{'R-5':>7}{'AUC':>7}"
           f"{'F1':>7}{'TNR':>7}{'топ10%':>8}{'катастр%':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:<34}{r['mAP']:>7.2f}{r['rank1']:>7.2f}{r['rank5']:>7.2f}"
              f"{r['auc']:>7.3f}{r['f1']:>7.3f}{r['TNR']:>7.3f}"
              f"{r['top10_pct']:>8.1f}{r['catastrophic_pct']:>10.1f}")
    print("-" * len(hdr))
    print("топ10% — доля запросов, где правильный ответ в первой десятке (больше = лучше)")
    print("катастр% — доля, где правильный ответ дальше 50 места (меньше = лучше,")
    print("           это и есть числовое выражение проблемы ракурса)")

    out = HERE / a.out; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    md = ["# Сводное сравнение вариантов", "",
          "Валидация: 150 машин (`seed=42`), кросс-камерный расчёт как у жюри.", "",
          "| Вариант | dim | mAP | Rank-1 | Rank-5 | AUC отказа | F1 | TNR | В топ-10 | Катастроф |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['dim']} | {r['mAP']} | {r['rank1']} | {r['rank5']} | "
                  f"{r['auc']} | {r['f1']} | {r['TNR']} | {r['top10_pct']}% | {r['catastrophic_pct']}% |")
    md += ["", "**В топ-10** — доля запросов, где правильный ответ попал в первую десятку "
               "(метрика сдачи требует именно топ-10). Больше — лучше.", "",
           "**Катастроф** — доля запросов, где правильный ответ дальше 50 места. "
           "Это числовое выражение проблемы смены ракурса из "
           "[ERROR_ANALYSIS.md](../ERROR_ANALYSIS.md): мы заявили её как главное "
           "ограничение, и по этой колонке видно, решена она или нет. Меньше — лучше."]
    (HERE / a.md).write_text("\n".join(md) + "\n")
    print(f"\nзаписано: {out} и {HERE / a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
