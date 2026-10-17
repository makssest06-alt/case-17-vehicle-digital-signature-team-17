"""Training-free local (patch-token) re-ranking of the global top-K, A10 holdout.

final(q, g) = base(q, g) + lam * (local(q, g) - mean_topK local(q, .)) for g in top-K of base,
items outside top-K keep their base order below. base = raw cosine or hubness beta 0.5.
lam / variant are chosen 2-fold by vehicle (tune on one half of the query vehicles, report on
the other) so the reported 'oof' number is not tuned on the queries it is scored on.
Scored with the organizer evaluate.ranking_metrics (top-10 first, junk stripped after).
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
EVALUATOR = Path(r"C:\Users\user\Downloads\evaluate.py")
REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}
TOP_K = 20
LAMS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
DEV = torch.device("cuda")


def rows(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def hub(g: np.ndarray, k: int = 5) -> np.ndarray:
    s = g @ g.T
    np.fill_diagonal(s, -np.inf)
    return np.sort(s, axis=1)[:, -k:].mean(axis=1)


def inner_mask(side: int = 14, ring: int = 1) -> torch.Tensor:
    m = torch.zeros(side, side, dtype=torch.bool)
    m[ring:side - ring, ring:side - ring] = True
    return m.flatten()


@torch.inference_mode()
def local_scores(qt: torch.Tensor, gt: torch.Tensor, cand: np.ndarray, qglob: torch.Tensor,
                 mask: torch.Tensor) -> dict[str, np.ndarray]:
    """qt (Q,N,D), gt (G,N,D) fp16 normalised tokens; cand (Q,K) gallery indices."""
    out = {k: np.zeros(cand.shape, dtype=np.float32) for k in ("maxsim", "mnn", "wmaxsim")}
    qt, gt = qt[:, mask], gt[:, mask]
    for i in range(len(cand)):
        q = qt[i].float()                                    # (n, d)
        g = gt[torch.as_tensor(cand[i], device=DEV)].float()  # (K, n, d)
        s = torch.einsum("nd,kmd->knm", q, g)                # (K, n, n)
        q2g, g2q = s.max(2), s.max(1)
        out["maxsim"][i] = ((q2g.values.mean(1) + g2q.values.mean(1)) / 2).cpu().numpy()
        n = q.shape[0]
        mutual = g2q.indices.gather(1, q2g.indices) == torch.arange(n, device=DEV)[None, :]
        out["mnn"][i] = mutual.float().mean(1).cpu().numpy()
        w = torch.clamp(q @ qglob[i], min=0)                 # saliency of each query patch
        out["wmaxsim"][i] = ((q2g.values * w).sum(1) / w.sum().clamp(min=1e-6)).cpu().numpy()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ev", EVALUATOR)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    z = np.load(HERE / "tokens.npz")
    ids = z["ids"].astype(str).tolist(); pos = {i: k for k, i in enumerate(ids)}
    train = {r["image_id"]: r for r in rows(D / "train.csv")}
    split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
    seen, hold_g, hold_q = set(), [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        (hold_q if key in seen else hold_g).append(r)
        seen.add(key)
    qids = [r["image_id"] for r in hold_q]; hg = [r["image_id"] for r in hold_g]
    tg = [r["image_id"] for r in rows(D / "test_gallery.csv")]
    qdf = pd.DataFrame(hold_q).set_index("image_id")
    qveh = qdf.vehicle_id.astype(int).to_numpy()
    fold = (qveh % 2).astype(int)  # 2 folds by vehicle
    qi = [pos[i] for i in qids]
    glob = z["glob"]
    mask = inner_mask().to(DEV)
    for tok_name in ("head_tok", "raw_tok"):
        tok = torch.from_numpy(z[tok_name]).to(DEV)
        qt = tok[qi]
        qglob_tok = None
        if tok_name == "head_tok":
            qglob_tok = torch.from_numpy(glob[qi]).to(DEV).half().float()
        for case, gids in (("holdout_361", hg), ("holdout+test_1111", hg + tg)):
            gi = [pos[i] for i in gids]
            recs = [{"image_id": i, "vehicle_id": train[i]["vehicle_id"], "camera_id": train[i]["camera_id"]}
                    if i in train else {"image_id": i, "vehicle_id": "x_" + i, "camera_id": "x_" + i} for i in gids]
            gdf = pd.DataFrame(recs).set_index("image_id")
            q, g = glob[qi], glob[gi]
            raw = q @ g.T
            gt = tok[gi]
            for base_name, base in (("raw", raw), ("hub0.5", raw - 0.5 * hub(g)[None, :])):
                cand = np.argsort(-base, axis=1, kind="stable")[:, :TOP_K]
                qg = qglob_tok if qglob_tok is not None else torch.from_numpy(q).to(DEV)
                if qglob_tok is None:  # raw tokens are 768-D: weight by mean-token direction instead
                    qg = torch.nn.functional.normalize(qt.float().mean(1), dim=1)
                loc = local_scores(qt, gt, cand, qg, mask)
                bk = np.take_along_axis(base, cand, axis=1)

                def per_query_ap(var: str, lam: float) -> tuple[dict, np.ndarray]:
                    l = loc[var]
                    final = bk + lam * (l - l.mean(axis=1, keepdims=True))
                    order = np.take_along_axis(cand, np.argsort(-final, axis=1, kind="stable"), axis=1)[:, :10]
                    ranked = {qid: [gids[j] for j in order[i]] for i, qid in enumerate(qids)}
                    return ranked, order

                def fold_map(ranked: dict, f: int | None) -> float:
                    sel = qdf if f is None else qdf[fold == f]
                    return ev.ranking_metrics(sel, gdf, ranked, top_k=10)["mAP@10"]

                grid = {}
                for var in loc:
                    for lam in LAMS:
                        ranked, _ = per_query_ap(var, lam)
                        grid[(var, lam)] = (ranked, [fold_map(ranked, 0), fold_map(ranked, 1)], fold_map(ranked, None))
                base_map = grid[("maxsim", 0.0)][2]
                # out-of-fold: pick (var, lam) on fold f, score on fold 1-f, combine by query count
                oof_ranked, picks = {}, []
                for f in (0, 1):
                    best = max(grid, key=lambda k: grid[k][1][f])
                    picks.append(best)
                    other = qdf.index[fold == 1 - f]
                    oof_ranked.update({qid: grid[best][0][qid] for qid in other})
                oof = ev.ranking_metrics(qdf, gdf, oof_ranked, top_k=10)
                print(f"\n== {tok_name} | {case} | base {base_name}: base mAP@10 {base_map:.4f}")
                for var in loc:
                    cells = "  ".join(f"λ{lam}:{grid[(var, lam)][2] - base_map:+.4f}" for lam in LAMS[1:])
                    print(f"   {var:8s} {cells}")
                print(f"   OUT-OF-FOLD: mAP@10 {oof['mAP@10']:.4f} ({oof['mAP@10'] - base_map:+.4f})  "
                      f"R1 {oof['Rank-1']:.4f}  R5 {oof['Rank-5']:.4f}  picks {picks}")


if __name__ == "__main__":
    main()
