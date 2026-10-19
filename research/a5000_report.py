#!/usr/bin/env python3
"""Сводка захода research/a5000_session.sh: одна таблица скорости (a10 и L75, EXTRACT_THREADS) с порогом и
формулой отказа, окружение машины — в session.json и session.md. Только стандартная библиотека.

    python3 research/a5000_report.py <OUT_DIR>
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JURY_CPU = "2× Intel Xeon Gold 6338, 128 логических CPU (ответ Q30)"
A10_REF = REPO / "research" / "honest_eval_results" / "closed_gallery_a10_q17.json"
FINAL_KEY = "(1, 'off', 'both', 'raw')"
Q34 = ("Баллы скорости — наша оценка на нашем замере по формуле ответа Q34: 10 % × latency_score + "
       "10 % × throughput_score (задержка и FPS поровну — так в Q34; ≤ 40 мс и ≥ 100 FPS — полный балл, "
       "40–80 мс и 50–100 FPS — линейно). Организаторы меряют сами, на своём CPU.")


def kv(path: Path) -> dict:
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def refusal(out: Path) -> dict:
    """Порог и формула: L75 — пересчёт этого захода, a10 — пресет конфига (closed_gallery_a10_q17.json)."""
    res = {}
    r = load(out / "recalibrate_l75.json")
    if r:
        p, m = r["presets"]["итог"], r["metrics"]["итог"]
        res["l75"] = {"threshold": p["balanced"], "strict": p["strict"], "Q15_361": m["361"]["Q15"],
                      "Q15_1111": m["1111"]["Q15"], "mAP_361": m["361"]["mAP@10"], "mAP_1111": m["1111"]["mAP@10"],
                      "tau": r["tau"], "rule": r["rule"], "source": "пересчёт этого захода (recalibrate_l75.json)"}
    a = load(A10_REF)
    if a:
        cp, pg = a["config_presets"][FINAL_KEY], a["presets"][a["final"]]["per_gallery"]
        res["a10"] = {"threshold": cp["balanced"], "strict": cp["strict"], "Q15_361": pg["361"]["Q15"],
                      "Q15_1111": pg["1111tr"]["Q15"], "mAP_361": a["galleries"]["361"][a["final"]]["mAP@10"],
                      "mAP_1111": a["galleries"]["1111tr"][a["final"]]["mAP@10"], "tau": a["tau"],
                      "rule": a["rules"]["ours"], "source": "пресет конфига (closed_gallery_a10_q17.json)"}
    return res


def bench_rows(out: Path, models: list, threads: list) -> list:
    rows = []
    for m in models:
        for t in threads:
            b, err = load(out / f"bench_{m}_t{t}.json"), out / f"bench_{m}_t{t}.err"
            if b is None:
                rows.append({"model": m, "threads": t, "error": err.read_text(encoding="utf-8").strip()[-300:]
                             if err.exists() else "нет результата"})
                continue
            rows.append({"model": m, "threads": t, "name": b.get("model"), "device": b.get("device"), "fp16": b.get("fp16"),
                         "latency_b1_ms": b["latency_b1_ms"]["median"], "latency_p95_ms": b["latency_b1_ms"]["p95"],
                         "best_fps": b["best_fps"], "best_batch": b["best_batch"], "fps": b["throughput_fps"],
                         "peak_vram_MB": b["peak_vram_MB"], "weights_MB": b["weights_size_q37"]["MB"],
                         "latency_score": b["latency_score"], "throughput_score": b["throughput_score"],
                         "performance_of_20": b["performance_pct_of_20"]})
    return rows


def fmt(x, spec: str) -> str:
    return "—" if x is None else format(x, spec)


def md(s: dict) -> str:
    e, g, ref = s["env"], s["gpu_check"] or {}, s["refusal"]
    L = ["# Заход на арендованной машине: a10 против L75", "",
         f"Статус: **{s['status']}**. Коммит `{e.get('git_commit', '—')}`, образ `{s['image'].get('image', '—')}` "
         f"({s['image'].get('built', '—')}).", "",
         "## Окружение", "",
         f"- GPU: {e.get('gpu', '—')}; драйвер {e.get('driver', '—')}, CUDA драйвера {e.get('cuda_driver', '—')}; "
         f"память {e.get('gpu_memory', '—')}.",
         f"- CPU: {e.get('cpu_model', '—')}; логических CPU {e.get('cpu_threads', '—')}, ядер на сокет "
         f"{e.get('cores_per_socket', '—')}, сокетов {e.get('sockets', '—')}; RAM {e.get('ram', '—')}.",
         f"- В образе: torch {g.get('torch') or '—'}, CUDA сборки {g.get('torch_cuda') or '—'}, cuDNN {g.get('cudnn') or '—'}; "
         f"видеокарта из контейнера: {'да, ' + str(g.get('gpu')) if g.get('cuda') else 'нет'}.",
         f"- Docker {e.get('docker', '—')}, nvidia-container-toolkit {e.get('nvidia_ctk', '—')}.",
         f"- У жюри: NVIDIA RTX A5000, драйвер CUDA 12.2, {JURY_CPU}. У a10 узкое место — декодирование JPEG на CPU, "
         "поэтому FPS отсюда переносить на стенд жюри только с поправкой на CPU.", ""]
    if s["rows"]:
        L += ["## Скорость extract() (bench_extract.py, протокол Q31) и отказ", "",
              "| Модель | EXTRACT_THREADS | latency_b1, мс (p95) | лучший FPS (батч) | VRAM пик, МБ | веса, МБ | "
              "баллы скорости, из 20 | порог / строгий | 0.7·F1+0.3·TNR 361 / 1111 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in s["rows"]:
            rf = ref.get(r["model"], {})
            thr = f"{rf['threshold']} / {rf['strict']}" if rf else "—"
            q = f"{rf['Q15_361']:.3f} / {rf['Q15_1111']:.3f}" if rf else "—"
            if "error" in r:
                L.append(f"| {r['model']} | {r['threads']} | ошибка: {r['error'][:120]} | | | | | {thr} | {q} |")
                continue
            L.append(f"| {r['model']} | {r['threads']} | {r['latency_b1_ms']:.2f} ({r['latency_p95_ms']:.2f}) | "
                     f"{r['best_fps']:.1f} ({r['best_batch']}) | {fmt(r['peak_vram_MB'], '.0f')} | {r['weights_MB']:.0f} | "
                     f"{r['performance_of_20']:.2f} | {thr} | {q} |")
        L += ["", Q34, ""]
        for m, rf in ref.items():
            L.append(f"- {m}: порог и формула — {rf['source']}; mAP@10 (DBA) {rf['mAP_361']:.4f} / {rf['mAP_1111']:.4f}, "
                     f"τ фона {rf['tau']}, почти-дубль: косинус ≥ {rf['rule']['s_hi']}, IoU ≥ {rf['rule']['t_iou']}, "
                     f"сдвиг центра ≤ {rf['rule']['t_center']}.")
        L += ["- Отказ — протокол Q17: галереи 361 и 1111 (+750 кадров обучающих машин), 20 % машин без пары, без "
              "данных теста. a10 считался на CPU Мака (FP32), L75 — здесь на GPU (FP16).", ""]
    sub = s.get("submission_gpu")
    if sub:
        L += ["## Артефакты сдачи на GPU (как у жюри) против artifacts/ репозитория", "",
              ("make_submission a10 в контейнере на GPU (FP16): " + sub["summary"] + "." if "summary" in sub
               else "make_submission a10 на GPU: ошибка — " + sub["error"].replace("\n", " ")), ""]
    if s.get("error"):
        L += ["## Остановка", "", s["error"], ""]
    if s.get("seconds"):
        L += ["Время по шагам: " + ", ".join(f"{k} {int(v) // 60} мин {int(v) % 60} с" for k, v in s["seconds"].items())
              + f"; всего {sum(int(v) for v in s['seconds'].values()) // 60} мин.", ""]
    L += ["Файлы: `session.json` (всё вместе), `bench_<модель>_t<потоки>.json`, `recalibrate_l75.json`, `session.log`."]
    return "\n".join(L) + "\n"


def submission(out: Path):
    """Шаг 5: make_submission на GPU против artifacts/ — сводка сравнения или ошибка; None — шаг не запускался."""
    c, err = load(out / "submission_compare.json"), out / "submission_gpu.err"
    if c:
        return {k: c[k] for k in ("summary", "same_files", "top1", "accepted", "confidence", "embeddings")}
    if err.exists():
        return {"error": err.read_text(encoding="utf-8").strip()[-400:]}
    return None


def line(path: Path) -> str:
    """Одна строка итога замера — для лога research/a5000_session.sh."""
    b = load(path)
    return (f"latency_b1 {b['latency_b1_ms']['median']} мс, лучший FPS {b['best_fps']} (батч {b['best_batch']}), "
            f"VRAM {b['peak_vram_MB']} МБ, баллы {b['performance_pct_of_20']} из 20")


def main() -> int:
    if sys.argv[1] == "--line":
        print(line(Path(sys.argv[2])))
        return 0
    if sys.argv[1] == "--compare-line":
        print(load(Path(sys.argv[2]))["summary"])
        return 0
    out = Path(sys.argv[1])
    cfg = kv(out / "session.cfg")
    models = cfg.get("models", "a10 l75").split()
    threads = [int(t) for t in cfg.get("threads", "4 8 16").split()]
    status = (out / "status.txt").read_text(encoding="utf-8").strip() if (out / "status.txt").exists() else "не завершён"
    s = {"status": status.splitlines()[0], "error": "\n".join(status.splitlines()[1:]) or None,
         "env": kv(out / "env_host.txt"), "image": kv(out / "image.txt"), "gpu_check": load(out / "gpu_check.json"),
         "config": cfg, "seconds": kv(out / "timings.txt"), "rows": bench_rows(out, models, threads) if (out / "bench_done").exists() else [],
         "refusal": refusal(out), "submission_gpu": submission(out), "q34_note": Q34, "jury_cpu": JURY_CPU,
         "recalibrate_l75": {k: v for k, v in (load(out / "recalibrate_l75.json") or {}).items()
                             if k not in ("closed_gallery", "calibration")} or None}
    s["benches"] = {f"{r['model']}_t{r['threads']}": load(out / f"bench_{r['model']}_t{r['threads']}.json")
                    for r in s["rows"]}
    (out / "session.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "session.md").write_text(md(s), encoding="utf-8")
    print(md(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
