#!/usr/bin/env python3
"""Режим отказа: умеет ли модель понять, что этой машины в галерее вообще нет.

На val половине запросов искусственно убираем их машину из галереи —
получаем честные «запросы без пары». Дальше смотрим, отделяется ли
сходство top-1 у запросов с парой от запросов без пары.
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

from head_model import ReidHead

HERE = Path(__file__).parent
FEAT = Path("/opt/case17/data/features")
MODELS = Path("/opt/case17/models")


def build_val(orig, flip, head=None, seed=42):
    idx = list(csv.DictReader((FEAT / "train_index.csv").open()))
    val_ids = {l.strip() for l in (HERE / "val_ids.txt").read_text().split() if l.strip()}
    rows = [(i, r) for i, r in enumerate(idx) if r["vehicle_id"] in val_ids]

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
        a, b = orig[ii], flip[ii]
        if head is not None:
            with torch.inference_mode():
                a = head(torch.from_numpy(a))[1].numpy()
                b = head(torch.from_numpy(b))[1].numpy()
        v = (a + b) / 2
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    return emb(qry), emb(gal), qry, gal


def run(q, g, qry, gal, seed=42):
    """Половине машин убираем их саму из галереи -> запросы заведомо без пары."""
    rng = np.random.default_rng(seed)
    vids = sorted({r["vehicle_id"] for _, r in qry})
    unknown = set(rng.choice(vids, size=len(vids) // 2, replace=False).tolist())

    g_vid = np.array([r["vehicle_id"] for _, r in gal])
    g_cam = np.array([r["camera_id"] for _, r in gal])
    sim = q @ g.T

    known_top1, unknown_top1 = [], []
    for i, (_, qr) in enumerate(qry):
        v = qr["vehicle_id"]
        keep = g_cam != qr["camera_id"]
        if v in unknown:
            keep = keep & (g_vid != v)          # его машины в галерее нет вовсе
        if not keep.any():
            continue
        best = float(sim[i][keep].max())
        (unknown_top1 if v in unknown else known_top1).append(best)

    pos, neg = np.array(known_top1), np.array(unknown_top1)
    allv = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(allv); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    auc = float((ranks[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    # Две рабочие точки. Порог по F1 при сбалансированных классах вырождается
    # в "принимать всё" (полнота 1.0, TNR 0), поэтому отдельно считаем
    # сбалансированную точку по Юдену (TPR + TNR - 1) — она и нужна на проде.
    by_f1, by_j, best_j = None, None, -9.0
    for t in np.arange(allv.min(), allv.max(), 0.002):
        tp = int((pos >= t).sum()); fp = int((neg >= t).sum())
        fn = int((pos < t).sum()); tn = int((neg < t).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        tnr = tn / (tn + fp)
        point = {"threshold": round(float(t), 3), "f1": round(f1, 3),
                 "precision": round(pr, 3), "recall": round(rc, 3), "TNR": round(tnr, 3)}
        if by_f1 is None or f1 > by_f1["f1"]:
            by_f1 = point
        j = rc + tnr - 1
        if j > best_j:
            best_j, by_j = j, point
    balanced = {"bal_" + k: v for k, v in by_j.items()}
    balanced["bal_youden_j"] = round(best_j, 3)
    return {"auc": round(auc, 3), "n_known": len(pos), "n_unknown": len(neg),
            "known_median": round(float(np.median(pos)), 3),
            "unknown_median": round(float(np.median(neg)), 3),
            **by_f1, **balanced}, pos, neg


def plot(pairs, path):
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.4 * len(pairs), 4.0), squeeze=False)
    for ax, (title, pos, neg, st) in zip(axes[0], pairs):
        bins = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 40)
        ax.hist(neg, bins=bins, alpha=.6, label=f"нет пары (n={len(neg)})", color="#c44")
        ax.hist(pos, bins=bins, alpha=.6, label=f"пара есть (n={len(pos)})", color="#37a")
        ax.axvline(st["threshold"], color="#222", ls="--", lw=1.2, label=f"порог {st['threshold']}")
        ax.axvline(st["bal_threshold"], color="#0a0", ls=":", lw=1.4,
                   label=f"сбаланс. {st['bal_threshold']}")
        ax.set_title(f"{title}\nAUC {st['auc']} | по F1: F1 {st['f1']} TNR {st['TNR']}"
                     f"\nсбаланс.: F1 {st['bal_f1']} TNR {st['bal_TNR']}", fontsize=9)
        ax.set_xlabel("сходство top-1"); ax.set_ylabel("запросов"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip_vitb16")
    ap.add_argument("--heads", default="", help="теги обученных голов через запятую")
    ap.add_argument("--out", default="refusal.png")
    a = ap.parse_args()

    orig = np.load(FEAT / f"train_{a.backbone}_orig.npy")
    flip = np.load(FEAT / f"train_{a.backbone}_flip.npy")
    results, pairs = {}, []

    q, g, qry, gal = build_val(orig, flip, head=None)
    st, pos, neg = run(q, g, qry, gal)
    results[f"{a.backbone}_raw"] = st
    pairs.append((f"{a.backbone} без головы", pos, neg, st))
    print(f"{a.backbone} без головы:", st, flush=True)

    for tag in [t for t in a.heads.split(",") if t]:
        ck = torch.load(MODELS / f"head_{tag}.pt", map_location="cpu", weights_only=False)
        ar = ck["args"]
        head = ReidHead(ar.get("in_dim", 768), ar["hidden"], ar["out_dim"], ar["dropout"])
        head.load_state_dict(ck["head"]); head.eval()
        q, g, qry, gal = build_val(orig, flip, head=head)
        st, pos, neg = run(q, g, qry, gal)
        results[tag] = st; pairs.append((f"CLIP + голова ({tag})", pos, neg, st))
        print(f"{tag}:", st, flush=True)

    out = HERE / "results"; out.mkdir(exist_ok=True)
    (out / "refusal.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    plot(pairs, HERE / a.out)
    print("график:", HERE / a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
