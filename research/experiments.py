#!/usr/bin/env python3
"""Дешёвые способы поднять качество без GPU: объединение моделей, re-ranking, multi-crop.

    python experiments.py                 # все эксперименты
    python experiments.py --only rerank

Правило отбора задано заранее: если приём стоит дороже 20% времени запроса
и даёт меньше одного пункта mAP — не берём. Поэтому у каждого эксперимента
меряется и качество, и цена в миллисекундах на запрос.

Протокол тот же, что везде: 150 отложенных машин, кросс-камерный расчёт,
совпадения внутри своей камеры исключены.
"""
import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from head_model import ReidHead

HERE = Path(__file__).parent
FEAT = Path("/opt/case17/data/features")
MODELS = Path("/opt/case17/models")

# Стоимость одного прямого прохода модели на нашем CPU, мс. Замерено
# benchmark.py: latency батч=1 p50 = 674.73 мс на два прохода (оригинал
# и отражение), то есть 337 мс на проход.
MS_PER_FORWARD = 337.0


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


def load_head(tag: str, in_dim: int):
    if not tag:
        return None
    ck = torch.load(MODELS / f"head_{tag}.pt", map_location="cpu", weights_only=False)
    ar = ck["args"]
    h = ReidHead(ar.get("in_dim", in_dim), ar["hidden"], ar["out_dim"], ar["dropout"])
    h.load_state_dict(ck["head"]); h.eval()
    return h


def embed(orig, flip, rows, head):
    ii = [i for i, _ in rows]
    a, b = orig[ii], flip[ii]
    if head is not None:
        with torch.inference_mode():
            a = head(torch.from_numpy(a))[1].numpy()
            b = head(torch.from_numpy(b))[1].numpy()
    v = (a + b) / 2
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def score(sim, qry, gal):
    """mAP, Rank-1, Rank-5 и доля катастроф по матрице сходства query x gallery."""
    g_vid = np.array([r["vehicle_id"] for _, r in gal])
    g_cam = np.array([r["camera_id"] for _, r in gal])
    aps, r1, r5, ranks = [], 0, 0, []
    for i, (_, qr) in enumerate(qry):
        keep = g_cam != qr["camera_id"]
        if not keep.any():
            continue
        rel = g_vid[keep] == qr["vehicle_id"]
        if not rel.any():
            continue
        s = sim[i][keep]
        order = np.argsort(-s)
        rs = rel[order]
        hits = np.flatnonzero(rs)
        aps.append(float(((np.arange(len(hits)) + 1) / (hits + 1)).mean()))
        r1 += int(rs[0]); r5 += int(rs[:5].any())
        ranks.append(int(hits[0]) + 1)
    ranks = np.array(ranks); n = len(ranks)
    return {
        "mAP": round(float(np.mean(aps)) * 100, 2),
        "rank1": round(r1 / n * 100, 2),
        "rank5": round(r5 / n * 100, 2),
        "top10_pct": round(float((ranks <= 10).mean()) * 100, 1),
        "catastrophic_pct": round(float((ranks > 50).mean()) * 100, 1),
        "queries": n,
    }


# --------------------------------------------------------------------------
# k-reciprocal re-ranking, Zhong et al. 2017
# --------------------------------------------------------------------------
def re_ranking(q, g, k1=20, k2=6, lambda_value=0.3):
    """Пересортировка выдачи по взаимному соседству.

    Идея: если A нашёл B среди своих k ближайших И B нашёл A среди своих —
    это взаимные соседи, и такому совпадению можно верить сильнее, чем
    одностороннему. Дальше считается расстояние Жаккара между множествами
    взаимных соседей и смешивается с исходным косинусным.

    Ничего не обучает и не требует меток: чистая постобработка матрицы.
    """
    nq, ng = len(q), len(g)
    total = nq + ng
    feat = np.vstack([q, g]).astype(np.float32)

    d = 2.0 - 2.0 * (feat @ feat.T)          # квадрат евклидова для L2-нормированных
    np.maximum(d, 0, out=d)
    d = np.transpose(d / (np.max(d, axis=0) + 1e-12))
    rank = np.argsort(d).astype(np.int32)

    V = np.zeros_like(d, dtype=np.float32)
    half = int(round(k1 / 2.0)) + 1
    for i in range(total):
        fwd = rank[i, : k1 + 1]
        back = rank[fwd, : k1 + 1]
        recip = fwd[np.where(back == i)[0]]
        expanded = recip
        for c in recip:
            c_fwd = rank[c, :half]
            c_back = rank[c_fwd, :half]
            c_recip = c_fwd[np.where(c_back == c)[0]]
            if len(np.intersect1d(c_recip, recip)) > 2.0 / 3 * len(c_recip):
                expanded = np.append(expanded, c_recip)
        expanded = np.unique(expanded)
        w = np.exp(-d[i, expanded])
        V[i, expanded] = w / w.sum()

    if k2 != 1:                               # сглаживание по k2 ближайшим
        V = np.stack([V[rank[i, :k2]].mean(axis=0) for i in range(total)])

    inv = [np.where(V[:, i] != 0)[0] for i in range(total)]
    jac = np.zeros((nq, total), dtype=np.float32)
    for i in range(nq):
        tmp = np.zeros(total, dtype=np.float32)
        nz = np.where(V[i] != 0)[0]
        for j, ind in enumerate(nz):
            rows = inv[ind]
            tmp[rows] += np.minimum(V[i, ind], V[rows, ind])
        jac[i] = 1 - tmp / (2.0 - tmp)

    final = jac * (1 - lambda_value) + d[:nq] * lambda_value
    return -final[:, nq:]                     # возвращаем «сходство»: чем больше, тем лучше


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="concat | rerank | multicrop")
    ap.add_argument("--head-siglip", default="p_baseline")
    ap.add_argument("--head-clip", default="v1_arc512")
    ap.add_argument("--out", default="results/experiments.json")
    a = ap.parse_args()

    results = []

    # --- база ---
    s_orig, s_flip, qry, gal = load_split("siglip_base")
    h_s = load_head(a.head_siglip, s_orig.shape[1])
    q_s, g_s = embed(s_orig, s_flip, qry, h_s), embed(s_orig, s_flip, gal, h_s)
    base = score(q_s @ g_s.T, qry, gal)
    base.update(name="база: SigLIP + голова", ms_per_query=round(2 * MS_PER_FORWARD, 1),
                delta_map=0.0, delta_ms_pct=0.0, verdict="—")
    results.append(base)
    print(f"база: mAP {base['mAP']}  ({base['ms_per_query']} мс/запрос)", flush=True)

    # --- эксперимент 1: объединение двух моделей ---
    if a.only in (None, "concat"):
        c_orig, c_flip, qry_c, gal_c = load_split("clip_vitb16")
        assert [r["image_id"] for _, r in qry_c] == [r["image_id"] for _, r in qry], \
            "сплиты CLIP и SigLIP разошлись"
        h_c = load_head(a.head_clip, c_orig.shape[1])
        q_c, g_c = embed(c_orig, c_flip, qry_c, h_c), embed(c_orig, c_flip, gal_c, h_c)

        for w in (0.5, 0.3):
            # каждая часть уже L2-нормирована, вес задаёт вклад CLIP
            q_cat = np.hstack([q_s * (1 - w), q_c * w])
            g_cat = np.hstack([g_s * (1 - w), g_c * w])
            q_cat /= np.linalg.norm(q_cat, axis=1, keepdims=True)
            g_cat /= np.linalg.norm(g_cat, axis=1, keepdims=True)
            r = score(q_cat @ g_cat.T, qry, gal)
            ms = 4 * MS_PER_FORWARD            # две модели по два прохода
            r.update(name=f"конкатенация SigLIP+CLIP (вес CLIP {w})",
                     ms_per_query=round(ms, 1),
                     delta_map=round(r["mAP"] - base["mAP"], 2),
                     delta_ms_pct=round((ms / base["ms_per_query"] - 1) * 100, 1))
            r["verdict"] = verdict(r)
            results.append(r)
            print(f"  {r['name']}: mAP {r['mAP']} ({r['delta_map']:+.2f}), "
                  f"время {r['delta_ms_pct']:+.0f}% -> {r['verdict']}", flush=True)

    # --- эксперимент 2: re-ranking ---
    if a.only in (None, "rerank"):
        for k1, k2, lam in ((20, 6, 0.3), (10, 3, 0.3), (20, 6, 0.6)):
            t0 = time.perf_counter()
            sim_rr = re_ranking(q_s, g_s, k1, k2, lam)
            dt_ms = (time.perf_counter() - t0) * 1000 / len(qry)
            r = score(sim_rr, qry, gal)
            ms = base["ms_per_query"] + dt_ms
            r.update(name=f"k-reciprocal re-ranking (k1={k1}, k2={k2}, λ={lam})",
                     ms_per_query=round(ms, 1),
                     rerank_ms=round(dt_ms, 2),
                     delta_map=round(r["mAP"] - base["mAP"], 2),
                     delta_ms_pct=round((ms / base["ms_per_query"] - 1) * 100, 1))
            r["verdict"] = verdict(r)
            results.append(r)
            print(f"  {r['name']}: mAP {r['mAP']} ({r['delta_map']:+.2f}), "
                  f"+{dt_ms:.1f} мс ({r['delta_ms_pct']:+.1f}%) -> {r['verdict']}", flush=True)

    # --- эксперимент 3: multi-crop ---
    if a.only in (None, "multicrop"):
        extra = sorted(FEAT.glob("val_siglip_base_s*_orig.npy"))
        if not extra:
            print("  multi-crop: нет признаков доп. масштабов, сначала extract_multicrop.py",
                  flush=True)
        else:
            run_multicrop(extra, qry, gal, h_s, q_s, g_s, base, results)

    # --- эксперимент 4: складываются ли выигравшие приёмы ---
    if a.only in (None, "combo"):
        c_orig, c_flip, qry_c, gal_c = load_split("clip_vitb16")
        h_c = load_head(a.head_clip, c_orig.shape[1])
        q_c, g_c = embed(c_orig, c_flip, qry_c, h_c), embed(c_orig, c_flip, gal_c, h_c)
        q_cat = np.hstack([q_s * 0.5, q_c * 0.5])
        g_cat = np.hstack([g_s * 0.5, g_c * 0.5])
        q_cat /= np.linalg.norm(q_cat, axis=1, keepdims=True)
        g_cat /= np.linalg.norm(g_cat, axis=1, keepdims=True)

        t0 = time.perf_counter()
        sim_rr = re_ranking(q_cat, g_cat, 20, 6, 0.3)
        dt_ms = (time.perf_counter() - t0) * 1000 / len(qry)
        r = score(sim_rr, qry, gal)
        ms = 4 * MS_PER_FORWARD + dt_ms
        r.update(name="конкатенация + re-ranking",
                 ms_per_query=round(ms, 1), rerank_ms=round(dt_ms, 2),
                 delta_map=round(r["mAP"] - base["mAP"], 2),
                 delta_ms_pct=round((ms / base["ms_per_query"] - 1) * 100, 1))
        r["verdict"] = verdict(r)
        results.append(r)
        print(f"  {r['name']}: mAP {r['mAP']} ({r['delta_map']:+.2f}), "
              f"время {r['delta_ms_pct']:+.0f}% -> {r['verdict']}", flush=True)

    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print_table(results)
    print(f"\nзаписано: {out}")
    return 0


def verdict(r):
    """Правило отбора: дороже 20% времени и меньше пункта mAP — отбрасываем."""
    if r["delta_map"] <= 0:
        return "отбросить: качество не выросло"
    if r["delta_ms_pct"] > 20 and r["delta_map"] < 1.0:
        return "отбросить: дорого и мало"
    return "взять"


def run_multicrop(extra, qry, gal, head, q_base, g_base, base, results):
    """Усреднение векторов с нескольких масштабов вокруг bbox."""
    idx = {r["image_id"]: n for n, r in enumerate(
        csv.DictReader((FEAT / "val_multicrop_index.csv").open()))}
    scales = []
    for f in extra:
        tag = re.search(r"_s(\d+\.\d+)_", f.name).group(1)
        arr = np.load(f)
        scales.append((tag, arr))

    def emb_rows(rows, arr):
        ii = [idx[r["image_id"]] for _, r in rows]
        x = arr[ii]
        if head is not None:
            with torch.inference_mode():
                x = head(torch.from_numpy(x))[1].numpy()
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    acc_q, acc_g = q_base.copy(), g_base.copy()
    used = ["1.00"]
    for tag, arr in scales:
        acc_q = acc_q + emb_rows(qry, arr)
        acc_g = acc_g + emb_rows(gal, arr)
        used.append(tag)
        qn = acc_q / np.linalg.norm(acc_q, axis=1, keepdims=True)
        gn = acc_g / np.linalg.norm(acc_g, axis=1, keepdims=True)
        r = score(qn @ gn.T, qry, gal)
        ms = (2 + len(used) - 1) * MS_PER_FORWARD
        r.update(name=f"multi-crop, масштабы {'+'.join(used)}",
                 ms_per_query=round(ms, 1),
                 delta_map=round(r["mAP"] - base["mAP"], 2),
                 delta_ms_pct=round((ms / base["ms_per_query"] - 1) * 100, 1))
        r["verdict"] = verdict(r)
        results.append(r)
        print(f"  {r['name']}: mAP {r['mAP']} ({r['delta_map']:+.2f}), "
              f"время {r['delta_ms_pct']:+.0f}% -> {r['verdict']}", flush=True)


def print_table(rows):
    print()
    w = 46
    line = "-" * (w + 52)
    print(line)
    print(f"{'приём':<{w}}{'mAP':>7}{'Δ mAP':>8}{'катастр%':>10}{'мс/запрос':>11}{'Δ время':>9}")
    print(line)
    for r in rows:
        print(f"{r['name'][:w-1]:<{w}}{r['mAP']:>7}{r['delta_map']:>+8.2f}"
              f"{r['catastrophic_pct']:>10}{r['ms_per_query']:>11}"
              f"{r['delta_ms_pct']:>+8.0f}%")
    print(line)
    print("Правило отбора: дороже 20% времени и меньше пункта mAP — отбрасываем.")
    for r in rows[1:]:
        print(f"  {r['name'][:w-1]:<{w}} {r['verdict']}")


if __name__ == "__main__":
    raise SystemExit(main())
