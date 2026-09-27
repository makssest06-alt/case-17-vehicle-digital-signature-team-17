#!/usr/bin/env python3
"""Две рабочие точки порога отказа для демо-режима: сбалансированная и строгая.

Считается на том же распределении, в котором работает сервис (галерея не
фильтруется по камере) — обоснование в docs/SUBMISSION_README.md.

  сбалансированная — максимум индекса Юдена (TPR + TNR - 1), обе ошибки
                     штрафуются одинаково; режим по умолчанию;
  строгая          — минимальный порог, при котором TNR не ниже 0.98,
                     то есть из двадцати чужих запросов ошибочно принимается
                     не больше одного. Нужен там, где цена ложного совпадения
                     высока: показать оператору лишнюю машину дешевле, чем
                     уверенно показать чужую.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_refusal import build_val, run
from head_model import ReidHead

FEAT = Path("/opt/case17/data/features")
MODELS = Path("/opt/case17/models")
HERE = Path(__file__).parent


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "p_baseline"
    backbone = sys.argv[2] if len(sys.argv) > 2 else "siglip_base"

    orig = np.load(FEAT / f"train_{backbone}_orig.npy")
    flip = np.load(FEAT / f"train_{backbone}_flip.npy")
    ck = torch.load(MODELS / f"head_{tag}.pt", map_location="cpu", weights_only=False)
    ar = ck["args"]
    head = ReidHead(ar.get("in_dim", orig.shape[1]), ar["hidden"], ar["out_dim"], ar["dropout"])
    head.load_state_dict(ck["head"]); head.eval()

    q, g, qry, gal = build_val(orig, flip, head=head)
    st, pos, neg = run(q, g, qry, gal, cross_camera_only=False)

    def metrics(t):
        tp = int((pos >= t).sum()); fp = int((neg >= t).sum())
        fn = int((pos < t).sum());  tn = int((neg < t).sum())
        pr = tp / (tp + fp) if tp + fp else 1.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        return {"threshold": round(float(t), 3), "precision": round(pr, 3),
                "recall": round(rc, 3), "f1": round(f1, 3),
                "TNR": round(tn / (tn + fp), 3)}

    grid = np.arange(float(min(pos.min(), neg.min())), float(max(pos.max(), neg.max())), 0.002)

    balanced = max(grid, key=lambda t: metrics(t)["recall"] + metrics(t)["TNR"] - 1)
    strict = next((t for t in grid if metrics(t)["TNR"] >= 0.98), grid[-1])

    out = {"backbone": backbone, "head": tag, "auc": st["auc"],
           "balanced": metrics(balanced), "strict": metrics(strict)}

    print(f"AUC {st['auc']}  (positives {len(pos)}, negatives {len(neg)})")
    for name, m in (("сбалансированный", out["balanced"]), ("строгий", out["strict"])):
        print(f"  {name:<18} порог {m['threshold']:.3f}  F1 {m['f1']:.3f}  "
              f"точность {m['precision']:.3f}  полнота {m['recall']:.3f}  TNR {m['TNR']:.3f}")

    path = HERE / "results" / "operating_points.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"записано: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
