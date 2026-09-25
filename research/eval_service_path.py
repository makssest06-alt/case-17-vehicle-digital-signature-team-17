#!/usr/bin/env python3
"""Метрики ТОГО пути, которым отвечает сервис, а не предрассчитанных признаков.

Обнаружено расхождение: вектор из /v1/embed и вектор из
train_siglip_base_*.npy для одного кадра дают косинус 0.57. Совпадают
сервис и текущий код (1.0000), расходится выгрузка признаков от 18.09.
Значит все метрики, посчитанные по .npy, описывают не сервис.

Этот скрипт считает валидацию заново тем же кодом, что в сервисе:
кроп -> timm-трансформ -> бэкбон -> голова -> flip-TTA -> L2.
Протокол оценки прежний: 150 отложенных машин, кросс-камерно, совпадения
своей камеры исключены. Плюс порог отказа в рабочем протоколе.
"""
import csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path("/opt/case17/app/research")
CROPS = Path("/opt/case17/data/crops/train")
sys.path.insert(0, str(HERE))
from head_model import ReidHead
import backbones


def build():
    ck = torch.load("/opt/case17/models/reid_current.pt", map_location="cpu", weights_only=False)
    ar = ck["args"]
    spec = backbones.get(ar.get("backbone", "siglip_base"))
    fwd, tf, model = backbones.build_encoder(spec)
    head = ReidHead(model.num_features, ar["hidden"], ar["out_dim"], ar["dropout"])
    head.load_state_dict(ck["head"]); head.eval()
    return model, head, tf


@torch.inference_mode()
def embed(rows, model, head, tf, batch=16, label=""):
    out = np.zeros((len(rows), 512), dtype="float32")
    t0 = time.perf_counter()
    for i in range(0, len(rows), batch):
        ch = rows[i:i + batch]
        ims = [Image.open(CROPS / f"{r['image_id']}.jpg").convert("RGB") for r in ch]
        x = torch.stack([tf(im) for im in ims])
        n = lambda t: F.normalize(model(t), p=2, dim=1)   # голова училась на нормированных
        f = head(n(x))[1] + head(n(torch.flip(x, dims=[3])))[1]
        out[i:i + len(ch)] = F.normalize(f, p=2, dim=1).numpy()
        if i and i % (batch * 10) == 0:
            print(f"    {label} {i}/{len(rows)}  {i/(time.perf_counter()-t0):.2f} кадр/с", flush=True)
    return out


def score(sim, qry, gal):
    g_vid = np.array([r["vehicle_id"] for r in gal])
    g_cam = np.array([r["camera_id"] for r in gal])
    aps, r1, r5, ranks = [], 0, 0, []
    for i, qr in enumerate(qry):
        keep = g_cam != qr["camera_id"]
        rel = g_vid[keep] == qr["vehicle_id"]
        if not keep.any() or not rel.any():
            continue
        s = sim[i][keep]; o = np.argsort(-s); rs = rel[o]
        hits = np.flatnonzero(rs)
        aps.append(float(((np.arange(len(hits)) + 1) / (hits + 1)).mean()))
        r1 += int(rs[0]); r5 += int(rs[:5].any()); ranks.append(int(hits[0]) + 1)
    ranks = np.array(ranks); n = len(ranks)
    return {"mAP": round(float(np.mean(aps)) * 100, 2), "rank1": round(r1/n*100, 2),
            "rank5": round(r5/n*100, 2), "top10_pct": round(float((ranks<=10).mean())*100, 1),
            "catastrophic_pct": round(float((ranks>50).mean())*100, 1), "queries": int(n)}


def refusal(q, g, qry, gal, cross, seed=42):
    rng = np.random.default_rng(seed)
    vids = sorted({r["vehicle_id"] for r in qry})
    unknown = set(rng.choice(vids, size=len(vids)//2, replace=False).tolist())
    g_vid = np.array([r["vehicle_id"] for r in gal])
    g_cam = np.array([r["camera_id"] for r in gal])
    sim = q @ g.T
    pos, neg = [], []
    for i, qr in enumerate(qry):
        v = qr["vehicle_id"]
        keep = (g_cam != qr["camera_id"]) if cross else np.ones(len(g_vid), bool)
        if v in unknown:
            keep = keep & (g_vid != v)
        if not keep.any():
            continue
        (neg if v in unknown else pos).append(float(sim[i][keep].max()))
    pos, neg = np.array(pos), np.array(neg)
    allv = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(allv); rk = np.empty(len(allv)); rk[o] = np.arange(1, len(allv)+1)
    auc = float((rk[lab==1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
    def at(t):
        tp, fp = int((pos>=t).sum()), int((neg>=t).sum())
        fn, tn = int((pos<t).sum()), int((neg<t).sum())
        pr = tp/(tp+fp) if tp+fp else 1.0; rc = tp/(tp+fn) if tp+fn else 0.0
        return {"threshold": round(float(t),4), "precision": round(pr,3), "recall": round(rc,3),
                "f1": round(2*pr*rc/(pr+rc),3) if pr+rc else 0.0,
                "TNR": round(tn/(tn+fp),3) if tn+fp else 0.0}
    grid = np.arange(float(allv.min()), float(allv.max()), 0.002)
    bal = max(grid, key=lambda t: at(t)["recall"] + at(t)["TNR"] - 1)
    strict = next((t for t in grid if at(t)["TNR"] >= 0.98), grid[-1])
    return {"auc": round(auc,4), "pos_median": round(float(np.median(pos)),4),
            "neg_median": round(float(np.median(neg)),4),
            "balanced": at(bal), "strict": at(strict)}


def main():
    vs = list(csv.DictReader((HERE / "val_split.csv").open()))
    gal = [r for r in vs if r["role"] == "gallery"]
    qry = [r for r in vs if r["role"] == "query"]
    model, head, tf = build()
    print(f"galлерея {len(gal)}, запросов {len(qry)}", flush=True)
    t0 = time.perf_counter()
    g = embed(gal, model, head, tf, label="gallery")
    q = embed(qry, model, head, tf, label="query")
    print(f"посчитано за {time.perf_counter()-t0:.0f} с", flush=True)

    rank = score(q @ g.T, qry, gal)
    print(f"\nранжирование: mAP {rank['mAP']}  Rank-1 {rank['rank1']}  Rank-5 {rank['rank5']}  "
          f"топ-10 {rank['top10_pct']}%  катастроф {rank['catastrophic_pct']}%")
    out = {"ranking": rank}
    for name, cross in (("jury_cross_camera", True), ("deployment", False)):
        r = refusal(q, g, qry, gal, cross); out[name] = r
        print(f"\nотказ, протокол «{name}»: AUC {r['auc']}  "
              f"медианы {r['pos_median']} / {r['neg_median']}")
        print(f"  сбалансированный порог {r['balanced']['threshold']}  F1 {r['balanced']['f1']}  "
              f"полнота {r['balanced']['recall']}  TNR {r['balanced']['TNR']}")
        print(f"  строгий          порог {r['strict']['threshold']}  F1 {r['strict']['f1']}  "
              f"точность {r['strict']['precision']}  TNR {r['strict']['TNR']}")
    p = HERE / "results" / "service_path.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nзаписано: {p}")


if __name__ == "__main__":
    main()
