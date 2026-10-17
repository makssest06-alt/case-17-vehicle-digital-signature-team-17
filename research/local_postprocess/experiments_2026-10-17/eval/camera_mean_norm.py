"""Camera-aware feature normalisation ceiling: e' = L2(e - alpha * mu_cam).

mu_cam = mean A10 embedding of TRAIN frames (clean train vehicles only) of that camera; test
cameras are train cameras, so this is a fixed 96-row lookup. Camera of each holdout / test frame:
'oracle' = true camera_id for holdout frames (pseudo for test-gallery distractors),
'pseudo' = Ilya's background camera vote for every frame. Organizer ranking_metrics, top-10.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
SCR = Path(__file__).resolve().parents[1]
EVALUATOR = Path(r"C:\Users\user\Downloads\evaluate.py")
REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}


def rows(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ev", EVALUATOR)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    train = {r["image_id"]: r for r in rows(D / "train.csv")}
    pseudo = {r["image_id"]: r["pseudo_cam"] for r in rows(SCR / "pseudo_camera.csv")}
    split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
    hold_v = {r["vehicle_id"] for r in split}
    seen, hg, hq = set(), [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        (hq if key in seen else hg).append(r)
        seen.add(key)
    z = np.load(D / "a10_clean_expanded.npz"); t = np.load(D / "a10_test.npz")
    emb = {str(i): e for i, e in zip(z["ids"].astype(str), z["emb"])}
    emb.update({str(i): e for i, e in zip(t["ids"].astype(str), t["emb"])})
    tr_ids = [i for i, r in train.items() if r["vehicle_id"] not in hold_v and i not in REMOVED]
    tr_e = l2(np.stack([emb[i] for i in tr_ids]))
    tr_cam = np.array([f"cam{train[i]['camera_id']}" for i in tr_ids])
    mu = {c: tr_e[tr_cam == c].mean(0) for c in np.unique(tr_cam)}
    glob_mu = tr_e.mean(0)
    tg = [r["image_id"] for r in rows(D / "test_gallery.csv")]
    qids = [r["image_id"] for r in hq]
    qdf = pd.DataFrame(hq).set_index("image_id")
    q = l2(np.stack([emb[i] for i in qids]))
    for name, gids in (("361", [r["image_id"] for r in hg]), ("1111", [r["image_id"] for r in hg] + tg)):
        g = l2(np.stack([emb[i] for i in gids]))
        recs = [{"image_id": i, "vehicle_id": train[i]["vehicle_id"], "camera_id": train[i]["camera_id"]} if i in train
                else {"image_id": i, "vehicle_id": "x_" + i, "camera_id": "x_" + i} for i in gids]
        gdf = pd.DataFrame(recs).set_index("image_id")
        cams = {
            "oracle": ([f"cam{train[i]['camera_id']}" for i in qids],
                       [f"cam{train[i]['camera_id']}" if i in train else pseudo[i] for i in gids]),
            "pseudo": ([pseudo[i] for i in qids], [pseudo[i] for i in gids]),
        }

        def score(qq: np.ndarray, gg: np.ndarray) -> dict:
            order = np.argsort(-(qq @ gg.T), axis=1, kind="stable")[:, :10]
            return ev.ranking_metrics(qdf, gdf, {qid: [gids[j] for j in order[i]] for i, qid in enumerate(qids)}, top_k=10)

        base = score(q, g)["mAP@10"]
        print(f"\n== gallery {name}: raw mAP@10 {base:.4f}")
        m = score(l2(q - glob_mu), l2(g - glob_mu))["mAP@10"]
        print(f"   global mean centering            {m:.4f} ({m - base:+.4f})")
        for tag, (qc, gc) in cams.items():
            for alpha in (0.5, 1.0):
                qn = l2(q - alpha * np.stack([mu.get(c, glob_mu) for c in qc]))
                gn = l2(g - alpha * np.stack([mu.get(c, glob_mu) for c in gc]))
                r = score(qn, gn)
                print(f"   camera mean [{tag:6s}] alpha {alpha}  {r['mAP@10']:.4f} ({r['mAP@10'] - base:+.4f})"
                      f"  R1 {r['Rank-1']:.3f}  R5 {r['Rank-5']:.3f}")


if __name__ == "__main__":
    main()
