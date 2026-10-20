"""Model soup прод-модели ветки: soup150s2 = среднее весов двух студентов ViT-L/16@256.

Вход — два чекпоинта одного рецепта (run_clean_student_large.sh), разные сиды и длины:
  clean_student_large_150_ep150.pt   (seed 7, 150 эпох;  clean 0.9005 / expanded 0.6467)
  clean_student_large_s2_ep75.pt     (seed 42, 75 эпох;  clean 0.8876 / expanded 0.6241)
Выход — поэлементное среднее state_dict["model"], args/margin берутся от первого файла.
Результат на официальном evaluator (protocol bm_improve/veri_pretrain/evaluation):
  clean mAP@10 0.9133 / expanded 0.6592 — выше обоих родителей (run-to-run шум ±0.002–0.004;
  контроль: soup чекпойнтов одного сида прироста не даёт — 0.8966/0.6268, т.е. работает
  именно разнообразие сидов, а не сглаживание хвоста обучения).

    python research/make_soup.py soup150s2.pt clean_student_large_150_ep150.pt clean_student_large_s2_ep75.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def make_soup(paths: list[Path], out_path: Path) -> None:
    acc: dict[str, torch.Tensor] | None = None
    args = None
    margin = None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"]
        if acc is None:
            acc = {k: v.detach().clone().float() for k, v in sd.items()}
            args = ck.get("args")
            margin = ck.get("margin")
            ref_keys = set(sd)
        else:
            assert set(sd) == ref_keys, f"несовпадение ключей в {p}"
            for k in acc:
                acc[k] += sd[k].float()
    assert acc is not None
    n = float(len(paths))
    torch.save({"model": {k: v / n for k, v in acc.items()}, "margin": margin, "args": args}, out_path)
    print(f"soup of {len(paths)} -> {out_path} ({out_path.stat().st_size} байт)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("usage: python research/make_soup.py <out.pt> <ckpt1.pt> <ckpt2.pt> [ckpt3.pt ...]")
    make_soup([Path(p) for p in sys.argv[2:]], Path(sys.argv[1]))
