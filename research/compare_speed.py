#!/usr/bin/env python3
"""Таблица ДО/ПОСЛЕ по скорости из json, которые пишут benchmark.py и profile_pipeline.py.

    python3 research/compare_speed.py --dir research/results/bench
    python3 research/compare_speed.py --dir ... --md > research/results/bench/table.md

ДО    — feat/research-baselines: torch + Pillow + поиск в pgvector.
ПОСЛЕ — feat/fast-inference: ONNX Runtime + TurboJPEG + in-memory индекс.
"""
import argparse
import json
from pathlib import Path


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def gain(before, after, lower_is_better=True):
    """Во сколько раз стало лучше. None, если считать не из чего."""
    if not before or not after:
        return None
    if lower_is_better:
        return before / after if after else None
    return after / before if before else None


def fmt_gain(g):
    if g is None:
        return "—"
    if g >= 1.0:
        return f"x{g:.2f}"
    return f"x{g:.2f} хуже"


def rows(b, a):
    """Строки таблицы: (метрика, до, после, выигрыш)."""
    out = []
    if b and a:
        out.append(("модель", b["model"], a["model"], "—"))
        out.append(("размер весов, МБ", b["weights_mb"], a["weights_mb"],
                    fmt_gain(gain(b["weights_mb"], a["weights_mb"]))))
        out.append(("загрузка модели, с", b["load_s"], a["load_s"],
                    fmt_gain(gain(b["load_s"], a["load_s"]))))
        for k in ("p50", "p95", "p99"):
            lb, la = b["latency_batch1_ms"][k], a["latency_batch1_ms"][k]
            out.append((f"latency батч=1 {k}, мс", lb, la, fmt_gain(gain(lb, la))))
        for bs in sorted(set(b["throughput_fps"]) & set(a["throughput_fps"]), key=int):
            fb, fa = b["throughput_fps"][bs], a["throughput_fps"][bs]
            out.append((f"FPS батч={bs}", fb, fa, fmt_gain(gain(fb, fa, lower_is_better=False))))
        out.append(("пиковая RAM, МБ", b["peak_rss_mb"], a["peak_rss_mb"],
                    fmt_gain(gain(b["peak_rss_mb"], a["peak_rss_mb"]))))
    return out


def stage_rows(pb, pa):
    out = []
    if not (pb and pa):
        return out
    keys = sorted(set(pb["stages"]) | set(pa["stages"]))
    for k in keys:
        vb = pb["stages"].get(k, {}).get("p50")
        va = pa["stages"].get(k, {}).get("p50")
        out.append((k, vb if vb is not None else "—", va if va is not None else "—",
                    fmt_gain(gain(vb, va))))
    out.append(("ИТОГО сумма p50, мс", pb["total_p50_ms"], pa["total_p50_ms"],
                fmt_gain(gain(pb["total_p50_ms"], pa["total_p50_ms"]))))
    return out


def shorten(v, width):
    v = str(v)
    return v if len(v) <= width else v[: width - 1] + "…"


def print_table(title, data, md=False):
    if not data:
        return
    print(f"\n### {title}\n" if md else f"\n{title}")
    if md:
        print("| метрика | ДО | ПОСЛЕ | выигрыш |")
        print("|---|---:|---:|---:|")
        for name, b, a, g in data:
            print(f"| {name} | {b} | {a} | {g} |")
        return
    w, c, gw = 26, 20, 11
    line = "-" * (w + 2 * c + gw)
    print(line)
    print(f"{'метрика':<{w}}{'ДО':>{c}}{'ПОСЛЕ':>{c}}{'выигрыш':>{gw}}")
    print(line)
    for name, b, a, g in data:
        print(f"{shorten(name, w - 1):<{w}}{shorten(b, c - 1):>{c}}"
              f"{shorten(a, c - 1):>{c}}{str(g):>{gw}}")
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="research/results/bench")
    ap.add_argument("--md", action="store_true", help="markdown вместо текста")
    a = ap.parse_args()
    d = Path(a.dir)

    b = load(d, "benchmark_before.json")
    af = load(d, "benchmark_after.json")
    pb = load(d, "profile_before.json")
    pa = load(d, "profile_after.json")

    missing = [n for n, v in (("benchmark_before", b), ("benchmark_after", af),
                              ("profile_before", pb), ("profile_after", pa)) if v is None]
    if missing:
        print(f"нет файлов: {', '.join(missing)} — таблица будет неполной")

    print_table("Пропускная способность и память", rows(b, af), a.md)
    print_table("Разложение запроса по этапам (p50, мс)", stage_rows(pb, pa), a.md)


if __name__ == "__main__":
    main()
