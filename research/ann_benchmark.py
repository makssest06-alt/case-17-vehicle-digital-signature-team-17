#!/usr/bin/env python3
"""ANN на миллионе векторов: что мы теряем, когда галерея вырастает в 2770 раз.

    python ann_benchmark.py --build     # собрать базу на 1M векторов (один раз)
    python ann_benchmark.py --method exact|hnsw|ivfpq
    python ann_benchmark.py --report    # таблица и график из готовых json

Каждый метод запускается отдельным процессом: три индекса одновременно в
3.9 ГБ памяти не живут, а мерить пиковую память имеет смысл только когда в
процессе ничего лишнего нет.

Честность замера держится на трёх вещах.

1. Дистракторы не случайный шум. Изотропные гауссовы векторы в 512 измерениях
   почти ортогональны всему, ANN на них не ошибается никогда, и замер выходит
   бессмысленно оптимистичным. Поэтому миллион дистракторов порождается из
   реальных признаков: берутся две случайные реальные машины, смешиваются со
   случайным весом, добавляется шум, результат нормируется. Такие векторы лежат
   на том же многообразии, что и настоящие, и путать их действительно трудно.

2. Качество меряется на реальных запросах с реальными метками. В базу
   подмешана настоящая валидационная галерея (361 кадр, 150 машин), запросы —
   те же 551, что и везде. То есть падение mAP это падение нашего mAP, а не
   абстрактный recall.

3. Протокол жюри соблюдён и здесь: совпадения внутри своей камеры
   отбрасываются из выдачи уже после поиска — на входе camera_id по-прежнему
   не используется.
"""
import argparse
import csv
import gc
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FEAT = Path("/opt/case17/data/features")
MODELS = Path("/opt/case17/models")
OUT = HERE / "results" / "ann"
BASE_PATH = Path("/opt/case17/data/ann/base_1m.npy")
META_PATH = Path("/opt/case17/data/ann/meta.json")

TOP_K = 100          # сколько кандидатов достаём — столько же у всех методов
N_TOTAL = 1_000_000


def peak_rss_mb() -> float:
    """Пиковая память процесса. ru_maxrss на Linux в килобайтах."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_real():
    """Валидационный сплит через обученную голову: галерея, запросы, метки."""
    import torch

    sys.path.insert(0, str(HERE))
    from head_model import ReidHead

    idx = list(csv.DictReader((FEAT / "train_index.csv").open()))
    orig = np.load(FEAT / "train_siglip_base_orig.npy")
    flip = np.load(FEAT / "train_siglip_base_flip.npy")
    val_ids = {l.strip() for l in (HERE / "val_ids.txt").read_text().split() if l.strip()}

    ck = torch.load(MODELS / "head_p_baseline.pt", map_location="cpu", weights_only=False)
    ar = ck["args"]
    head = ReidHead(ar.get("in_dim", orig.shape[1]), ar["hidden"], ar["out_dim"], ar["dropout"])
    head.load_state_dict(ck["head"]); head.eval()

    def embed(rows):
        ii = [i for i, _ in rows]
        with torch.inference_mode():
            a = head(torch.from_numpy(orig[ii]))[1].numpy()
            b = head(torch.from_numpy(flip[ii]))[1].numpy()
        v = (a + b) / 2
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype("float32")

    rows = [(i, r) for i, r in enumerate(idx) if r["vehicle_id"] in val_ids]
    by = defaultdict(lambda: defaultdict(list))
    for i, r in rows:
        by[r["vehicle_id"]][r["camera_id"]].append((i, r))
    gal, qry = [], []
    for v in sorted(by):
        for cam in sorted(by[v], key=int):
            items = sorted(by[v][cam], key=lambda t: t[1]["image_id"])
            gal.append(items[0]); qry += items[1:]

    # Исходный материал для дистракторов — ТОЛЬКО машины, которых нет в
    # валидации. Иначе часть "чужих" векторов окажется клонами искомой
    # машины, они встанут выше правильного ответа, и метрика провалится
    # не из-за размера галереи, а из-за дефекта генератора.
    pool_rows = [(i, r) for i, r in enumerate(idx) if r["vehicle_id"] not in val_ids]
    return embed(gal), embed(qry), gal, qry, embed(pool_rows)


def build_base(seed: int = 42):
    """Миллион векторов: реальная галерея плюс правдоподобные дистракторы."""
    g, q, gal, qry, pool = load_real()
    n_real = len(g)
    n_fake = N_TOTAL - n_real
    print(f"реальная галерея {n_real}, дистракторов {n_fake}, всего {N_TOTAL}", flush=True)

    BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    base = np.lib.format.open_memmap(BASE_PATH, mode="w+", dtype="float32",
                                     shape=(N_TOTAL, g.shape[1]))
    base[:n_real] = g

    # Каждый дистрактор — реальная посторонняя машина плюс шум. Смешивать
    # две машины нельзя: результат лежит вне многообразия и оказывается
    # ближе к среднему, то есть похожим сразу на всё.
    rng = np.random.default_rng(seed)
    sigma = float(os.getenv("ANN_SIGMA", "0.12"))
    t0 = time.perf_counter()
    chunk = 50_000
    for start in range(n_real, N_TOTAL, chunk):
        m = min(chunk, N_TOTAL - start)
        v = pool[rng.integers(0, len(pool), m)].copy()
        v += rng.normal(0, sigma, v.shape).astype("float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        base[start : start + m] = v
        if (start - n_real) % 200_000 == 0:
            print(f"  {start}/{N_TOTAL}  {time.perf_counter() - t0:.0f} с", flush=True)
    base.flush()

    # Калибровка: насколько синтетические дистракторы трудны по сравнению с
    # настоящими чужими машинами. Если распределения расходятся — бенчмарк
    # меряет не то, и об этом надо сказать вслух, а не прятать.
    g_vid = np.array([r["vehicle_id"] for _, r in gal])
    q_vid = np.array([r["vehicle_id"] for _, r in qry])
    real_top, fake_top = [], []
    probe = base[n_real : n_real + 50_000]
    for i in range(0, len(q), 64):
        qq = q[i : i + 64]
        sr = qq @ g.T
        for j in range(len(qq)):
            mask = g_vid != q_vid[i + j]
            if mask.any():
                real_top.append(float(sr[j][mask].max()))
        fake_top.extend((qq @ probe.T).max(axis=1).tolist())
    calib = {
        "sigma": sigma,
        "real_negative_top1": {"median": round(float(np.median(real_top)), 4),
                               "p95": round(float(np.percentile(real_top, 95)), 4)},
        "synthetic_top1_per_50k": {"median": round(float(np.median(fake_top)), 4),
                                   "p95": round(float(np.percentile(fake_top, 95)), 4)},
    }
    print("калибровка дистракторов:", json.dumps(calib, ensure_ascii=False), flush=True)

    META_PATH.write_text(json.dumps({
        "n_total": N_TOTAL, "dim": int(g.shape[1]), "n_real_gallery": n_real,
        "gallery_vehicle_id": [r["vehicle_id"] for _, r in gal],
        "gallery_camera_id": [r["camera_id"] for _, r in gal],
        "query_vehicle_id": [r["vehicle_id"] for _, r in qry],
        "query_camera_id": [r["camera_id"] for _, r in qry],
        "calibration": calib,
    }, ensure_ascii=False))
    np.save(BASE_PATH.parent / "queries.npy", q)
    print(f"готово за {time.perf_counter() - t0:.0f} с -> {BASE_PATH} "
          f"({BASE_PATH.stat().st_size / 1024**3:.2f} ГиБ)", flush=True)


def score(order, meta):
    """mAP, Rank-1 и доля катастроф по усечённой выдаче, протокол жюри."""
    g_vid = np.array(meta["gallery_vehicle_id"])
    g_cam = np.array(meta["gallery_camera_id"])
    q_vid = meta["query_vehicle_id"]
    q_cam = meta["query_camera_id"]
    n_real = meta["n_real_gallery"]

    aps, r1, ranks, found = [], 0, [], 0
    for i in range(len(order)):
        ids = order[i]
        # оставляем то, что реально вернули: дистракторы считаются мимо,
        # совпадения своей камеры выбрасываем уже после поиска
        keep = []
        for j in ids:
            if j < n_real and g_cam[j] == q_cam[i]:
                continue
            keep.append(j)
        if not keep:
            continue
        rel = np.array([j < n_real and g_vid[j] == q_vid[i] for j in keep])
        if not rel.any():
            ranks.append(TOP_K + 1)       # правильного ответа в выдаче нет вовсе
            aps.append(0.0)
            continue
        found += 1
        hits = np.flatnonzero(rel)
        aps.append(float(((np.arange(len(hits)) + 1) / (hits + 1)).mean()))
        r1 += int(rel[0])
        ranks.append(int(hits[0]) + 1)

    ranks = np.array(ranks)
    n = len(ranks)
    return {
        "mAP": round(float(np.mean(aps)) * 100, 2),
        "rank1": round(r1 / n * 100, 2),
        "top10_pct": round(float((ranks <= 10).mean()) * 100, 1),
        "not_found_pct": round(float((ranks > TOP_K).mean()) * 100, 1),
        "queries": int(n),
        "with_hit": int(found),
    }


def recall_vs_exact(order, exact_order):
    """Сколько из точной выдачи ANN нашёл. Классическая метрика ANN."""
    r1 = float(np.mean([order[i][0] == exact_order[i][0] for i in range(len(order))]))
    r10 = float(np.mean([len(set(order[i][:10]) & set(exact_order[i][:10])) / 10
                         for i in range(len(order))]))
    return {"recall@1": round(r1, 4), "recall@10": round(r10, 4)}


def latency(fn, q, n=100):
    """p50/p95 одиночного запроса — так его видит сервис."""
    ts = []
    for i in range(min(n, len(q))):
        t = time.perf_counter()
        fn(q[i : i + 1])
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"p50_ms": round(ts[len(ts) // 2], 2),
            "p95_ms": round(ts[int(0.95 * (len(ts) - 1))], 2),
            "mean_ms": round(float(np.mean(ts)), 2)}


def run_exact(base, q, meta):
    """Точный перебор потоком. Материализовать всю матрицу сходств нельзя:
    551 запрос на миллион векторов это 2.2 ГБ float32, больше чем вся память
    сервера. Поэтому top-K держится бегущим и сливается с каждым куском."""
    res = {"method": "точный перебор", "index_mb": round(base.nbytes / 1024**2, 1)}
    nq = len(q)
    rows = np.arange(nq)[:, None]
    best_sim = np.full((nq, TOP_K), -2.0, dtype="float32")
    best_idx = np.zeros((nq, TOP_K), dtype="int64")

    t0 = time.perf_counter()
    chunk = 50_000
    for s0 in range(0, N_TOTAL, chunk):
        e = min(s0 + chunk, N_TOTAL)
        sim = q @ base[s0:e].T                       # (nq, m), ~110 МБ
        k = min(TOP_K, e - s0)
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        cand_sim = np.hstack([best_sim, sim[rows, part]])
        cand_idx = np.hstack([best_idx, part + s0])
        sel = np.argpartition(-cand_sim, TOP_K - 1, axis=1)[:, :TOP_K]
        best_sim = cand_sim[rows, sel]
        best_idx = cand_idx[rows, sel]
        del sim, part, cand_sim, cand_idx
    order = best_idx[rows, np.argsort(-best_sim, axis=1)]
    res["batch_total_s"] = round(time.perf_counter() - t0, 2)

    def one(x):
        bs, bi = np.full(TOP_K, -2.0, dtype="float32"), np.zeros(TOP_K, dtype="int64")
        for s0 in range(0, N_TOTAL, chunk):
            e = min(s0 + chunk, N_TOTAL)
            sim = (x @ base[s0:e].T)[0]
            k = min(TOP_K, e - s0)
            part = np.argpartition(-sim, k - 1)[:k]
            cs = np.concatenate([bs, sim[part]])
            ci = np.concatenate([bi, part + s0])
            sel = np.argpartition(-cs, TOP_K - 1)[:TOP_K]
            bs, bi = cs[sel], ci[sel]
        return bi

    res["latency"] = latency(one, q, n=10)
    res["build_s"] = 0.0
    res.update(score(order, meta))
    res["peak_rss_mb"] = round(peak_rss_mb(), 1)
    np.save(OUT / "order_exact.npy", order)
    return res


def run_hnsw(base, q, meta, M=16, ef_construction=40, ef_search=64):
    import faiss

    d = base.shape[1]
    res = {"method": f"FAISS HNSW (M={M}, efSearch={ef_search})"}
    try:
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    except Exception:
        # векторы L2-нормированы, значит порядок по евклиду и по косинусу
        # совпадает — падать на L2 безопасно
        print("  inner product недоступен для HNSW, беру L2", flush=True)
        index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter()
    step = 100_000
    for s in range(0, N_TOTAL, step):
        index.add(np.ascontiguousarray(base[s : s + step]))
        print(f"  добавлено {min(s + step, N_TOTAL)}/{N_TOTAL}  "
              f"{time.perf_counter() - t0:.0f} с", flush=True)
    res["build_s"] = round(time.perf_counter() - t0, 1)
    index.hnsw.efSearch = ef_search
    # HNSW хранит сами векторы плюс граф соседей
    res["index_mb"] = round((N_TOTAL * d * 4 + N_TOTAL * M * 2 * 4) / 1024**2, 1)

    t0 = time.perf_counter()
    _, order = index.search(q, TOP_K)
    res["batch_total_s"] = round(time.perf_counter() - t0, 2)
    res["latency"] = latency(lambda x: index.search(x, TOP_K), q)
    res.update(score(order, meta))
    res.update(recall_vs_exact(order, np.load(OUT / "order_exact.npy")))
    res["peak_rss_mb"] = round(peak_rss_mb(), 1)
    return res


def run_ivfpq(base, q, meta, nlist=4096, m=64, nbits=8, nprobe=32):
    import faiss

    d = base.shape[1]
    res = {"method": f"FAISS IVF-PQ (nlist={nlist}, m={m}, nprobe={nprobe})"}
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quant, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    t0 = time.perf_counter()
    train = np.ascontiguousarray(base[: 200_000])
    index.train(train)
    del train; gc.collect()
    step = 100_000
    for s in range(0, N_TOTAL, step):
        index.add(np.ascontiguousarray(base[s : s + step]))
    res["build_s"] = round(time.perf_counter() - t0, 1)
    index.nprobe = nprobe
    # PQ хранит m байт на вектор плюс кодовые книги и списки
    res["index_mb"] = round((N_TOTAL * m + nlist * d * 4) / 1024**2, 1)

    t0 = time.perf_counter()
    _, order = index.search(q, TOP_K)
    res["batch_total_s"] = round(time.perf_counter() - t0, 2)
    res["latency"] = latency(lambda x: index.search(x, TOP_K), q)
    res.update(score(order, meta))
    res.update(recall_vs_exact(order, np.load(OUT / "order_exact.npy")))
    res["peak_rss_mb"] = round(peak_rss_mb(), 1)
    return res


def report():
    rows = []
    for name in ("exact", "hnsw", "ivfpq"):
        p = OUT / f"{name}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if not rows:
        print("нет результатов, сначала прогони методы", file=sys.stderr)
        return 1

    base = rows[0]
    w = 40
    line = "-" * (w + 62)
    print(f"\nМиллион векторов, {base['queries']} реальных запросов, топ-{TOP_K}")
    print(line)
    print(f"{'метод':<{w}}{'p50 мс':>9}{'p95 мс':>9}{'индекс МБ':>11}"
          f"{'mAP':>8}{'R-1':>8}{'R@10':>8}")
    print(line)
    for r in rows:
        print(f"{r['method'][:w-1]:<{w}}{r['latency']['p50_ms']:>9}"
              f"{r['latency']['p95_ms']:>9}{r['index_mb']:>11}"
              f"{r['mAP']:>8}{r['rank1']:>8}"
              f"{r.get('recall@10', '—'):>8}")
    print(line)
    print(f"{'постройка индекса, с':<{w}}" + "".join(f"{r['build_s']:>9}" for r in rows))
    print(f"{'не найдено в топ-100, %':<{w}}" + "".join(f"{r['not_found_pct']:>9}" for r in rows))

    (OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    chart(rows)
    return 0


def chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    names = [r["method"].split(" (")[0] for r in rows]
    colors = ["#37a", "#3a7", "#c74"]

    ax = axes[0]
    vals = [r["latency"]["p50_ms"] for r in rows]
    bars = ax.bar(names, vals, color=colors[: len(rows)])
    ax.set_yscale("log")
    ax.set_ylabel("мс на запрос (log)")
    ax.set_title("Задержка одного запроса, p50")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    vals = [r["index_mb"] for r in rows]
    bars = ax.bar(names, vals, color=colors[: len(rows)])
    ax.set_ylabel("МБ")
    ax.set_title("Память индекса на 1M векторов")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    vals = [r["mAP"] for r in rows]
    bars = ax.bar(names, vals, color=colors[: len(rows)])
    ax.axhline(rows[0]["mAP"], color="#888", ls="--", lw=1)
    ax.set_ylabel("mAP на реальных запросах")
    ax.set_title("Качество: что теряем против точного перебора")
    for b, v in zip(bars, vals):
        d = v - rows[0]["mAP"]
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.2f}" + ("" if d == 0 else f"\n({d:+.2f})"),
                ha="center", va="bottom", fontsize=9)

    fig.suptitle("Поиск в галерее на миллион векторов: цена и качество", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "ann_1m.png", dpi=130)
    print(f"\nграфик: {HERE / 'ann_1m.png'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--method", choices=["exact", "hnsw", "ivfpq"])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if a.build:
        build_base()
        return 0
    if a.report:
        return report()
    if not a.method:
        ap.error("нужен --build, --method или --report")

    meta = json.loads(META_PATH.read_text())
    base = np.load(BASE_PATH, mmap_mode="r")
    q = np.load(BASE_PATH.parent / "queries.npy")
    print(f"{a.method}: база {base.shape}, запросов {len(q)}", flush=True)

    fn = {"exact": run_exact, "hnsw": run_hnsw, "ivfpq": run_ivfpq}[a.method]
    res = fn(base, q, meta)
    (OUT / f"{a.method}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
