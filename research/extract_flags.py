#!/usr/bin/env python3
"""Ускорения extract() (ответ Q36) и качество: каждый флаг — на галерее 361 / 546 запросах holdout.

Кадры holdout читаются из dataset.zip и идут полным циклом app/extractor.py (декодирование, кроп по bbox,
препроцессинг, forward, L2) — тем, что меряют организаторы (Q31). Для каждого варианта: косинус с эталонным
вектором того же кадра (min / среднее) и mAP@10, Rank-1, Rank-5 официальным ranking_metrics — по сырому
косинусу и по галерее после DBA k=1 (ранжирование сервиса). Эталон для CPU-вариантов — CPU FP32 без флагов;
для FP16 — тот же девайс в FP32 (на Маке — MPS: autocast CUDA здесь недоступен).

    python research/extract_flags.py --data-dir <данные> --zip dataset.zip --env REID_WEIGHTS=<веса a10> \\
        --env REID_TF_MODE=squash --variants cpu,cpu_draft,mps,mps_fp16
    -> research/honest_eval_results/extract_flags_a10.{json,md}
"""
import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "api"))
import bigallery_pp as bp  # noqa: E402
import honest_eval as he  # noqa: E402
import official_evaluate as oe  # noqa: E402

# имя: (REID_DEVICE, REID_FP16, EXTRACT_JPEG_DRAFT, эталон)
VARIANTS = {"cpu": ("cpu", "0", "0", None), "cpu_draft": ("cpu", "0", "1", "cpu"),
            "mps": ("mps", "0", "0", "cpu"), "mps_fp16": ("mps", "1", "0", "mps"),
            "mps_fp16_draft": ("mps", "1", "1", "mps"),
            "cuda": ("cuda", "0", "0", None), "cuda_fp16": ("cuda", "1", "0", "cuda"),
            "cuda_draft": ("cuda", "0", "1", "cuda"), "cuda_fp16_draft": ("cuda", "1", "1", "cuda")}


def embed_all(name: str, ids: list, boxes: dict, zf: zipfile.ZipFile, batch: int) -> tuple:
    dev, fp16, draft, _ = VARIANTS[name]
    os.environ.update({"REID_DEVICE": dev, "REID_FP16": fp16, "EXTRACT_JPEG_DRAFT": draft})
    from app.extractor import Extractor
    ex = Extractor(threads=4)
    ex.load()
    t0, out = time.perf_counter(), []
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch]
        out.append(ex.extract_batch([(zf.read(f"images/{i}.jpg"), boxes[i]) for i in chunk]))
    return np.vstack(out), round(time.perf_counter() - t0, 1), ex.describe()


def ranking(Q, G, q, g) -> dict:
    from app.pp.dba import dba
    out = {}
    for kind, gv in (("raw", g), ("dba", dba(g, k=1, weights="cos"))):
        order = np.argsort(-(q @ gv.T), axis=1, kind="stable")[:, :10]
        gids, qids = list(G.index), list(Q.index)
        rk = oe.ranking_metrics(Q, G, {qid: [gids[j] for j in order[i]] for i, qid in enumerate(qids)}, top_k=10)
        out[kind] = {"mAP@10": rk["mAP@10"], "R1": rk["Rank-1"], "R5": rk["Rank-5"]}
    return out


def main() -> int:
    ap = bp.data_args(argparse.ArgumentParser())
    ap.add_argument("--variants", default="cpu,cpu_draft,mps,mps_fp16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="a10")
    a = ap.parse_args()
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    D = he.Data(a)
    qrows, grows = D.valid(D.qry), D.valid(D.gal)
    Q, G = he.frame(qrows), he.frame(grows)
    ids = list(Q.index) + list(G.index)
    boxes = {i: tuple(int(float(v)) for v in D.boxes[i]) for i in ids}
    zf = zipfile.ZipFile(a.zip)
    nq = len(Q)
    res = {"tag": a.tag, "n_query": nq, "n_gallery": len(G), "variants": {}}
    vecs = {}
    for name in a.variants.split(","):
        V, took, info = embed_all(name, ids, boxes, zf, a.batch)
        vecs[name] = V
        ref = VARIANTS[name][3]
        cos = (V * vecs[ref]).sum(1) if ref in vecs else None
        res["variants"][name] = {**info, "took_s": took, "ref": ref if cos is not None else None,
                                 "cos_min": None if cos is None else float(cos.min()),
                                 "cos_mean": None if cos is None else float(cos.mean()),
                                 "metrics": ranking(Q, G, V[:nq], V[nq:])}
        x = res["variants"][name]
        print(f"{name:16s} {took:6.1f} с  cos min {x['cos_min']}  mAP raw {x['metrics']['raw']['mAP@10']:.4f} "
              f"dba {x['metrics']['dba']['mAP@10']:.4f}", flush=True)
    out = HERE / "honest_eval_results" / f"extract_flags_{a.tag}"
    out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    L = [f"# Ускорения extract() и качество: {a.tag}", "",
         f"Holdout: {nq} запросов, {len(G)} кадров галереи, официальный `ranking_metrics`. Полный цикл "
         "`api/app/extractor.py` по кадрам из dataset.zip. Косинус — с вектором того же кадра у эталона. "
         "`python research/extract_flags.py ...` (докстринг).", "",
         "| Вариант | Устройство | FP16 | JPEG draft | Эталон | cos min / среднее | mAP@10 raw | mAP@10 DBA | R1 / R5 DBA | Время, с |",
         "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for name, x in res["variants"].items():
        c = "—" if x["cos_min"] is None else f"{x['cos_min']:.5f} / {x['cos_mean']:.6f}"
        m = x["metrics"]
        L.append(f"| {name} | {x['device']} | {x['fp16']} | {x['jpeg_draft']} | {x['ref'] or '—'} | {c} | "
                 f"{m['raw']['mAP@10']:.4f} | {m['dba']['mAP@10']:.4f} | {m['dba']['R1']:.3f} / {m['dba']['R5']:.3f} | "
                 f"{x['took_s']} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {out}.json, .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
