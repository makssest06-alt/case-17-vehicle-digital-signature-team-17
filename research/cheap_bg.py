#!/usr/bin/env python3
"""Дешёвый признак фона для штрафа за ту же камеру: без второго прогона сети.

resnet18 по кадру с замазанной машиной (research/pseudo_camera.py) стоит 19–25 мс
CPU на запрос (research/bench_bigallery_pp.py). Здесь фон описывается пикселями
кадра, который сервис и так декодирует (api/app/imaging.load_image): кадр ужимается
до 128×72 (PIL reduce — среднее по блокам), машина (bbox + 10 %, как у кропа)
выбрасывается маской с запасом в пиксель, дальше признаки:
  grad64   модуль градиента размытого серого кадра 64×36 — геометрия сцены;
  grad32   то же 32×18;
  tiny     цвет 32×18, каждый канал z-нормирован по пикселям фона;
  hist     гистограммы RGB 4×4×4 в зонах 4×4 по пикселям фона, корень + L2;
  a+b      склейка: каждая часть L2, затем вместе L2.
Справочно — resnet18 (emb) и grad из research/pseudo_camera.py (кадр в 1/4,
машина залита средним, контур рамки виден).

Правило штрафа то же, что в research/bigallery_pp.py, заново ничего не
подбирается: τ — самый низкий порог с точностью «та же камера» ≥ 0.90 на парах
обучающих машин (их camera_id), проверка по группам камер; кандидат «с той же
камеры» уходит в конец списка (λ = inf). Штраф — поверх DBA k=1, галереи «как
сервис» (почти-дубли в галерее). Признак для рекомендации выбирается по
калибровке на обучающих машинах и цене, а не по итогу на галереях.

Время — CPU этого Мака на запрос: от декодированного кадра до косинусов с фоном
top-100 кандидатов (фон кадра галереи считается один раз при добавлении). FPS —
оценка: прогон a10 на RTX 4060 FP16 из research/CLEAN_RETRAIN_REPORT.md ветки
team17/finals-student-gpu-handoff (8.27 мс на кадр при b1, 4.88 мс при b8, только
модель) плюс это время последовательно, без перекрытия CPU и GPU.

    python research/cheap_bg.py --zip dataset.zip --data-dir <данные> --env REID_WEIGHTS=<веса> --tag a10
    python research/cheap_bg.py --data-dir <данные> --tag a10 --report
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bigallery_pp as bp  # noqa: E402
import honest_eval as he  # noqa: E402
import pseudo_camera as pc  # noqa: E402
from dba import l2  # noqa: E402

SMALL = (128, 72)                       # 1920×1080 / 15
GRAY = np.array([0.299, 0.587, 0.114], "float32")
ZONES, BINS = 4, 4
PARTS = {"grad64": 64 * 36, "grad32": 32 * 18, "tiny": 32 * 18 * 3, "hist": ZONES * ZONES * BINS ** 3}
KINDS = ("grad64", "grad32", "tiny", "hist", "grad64+tiny", "grad64+hist", "grad32+tiny+hist")
REFERENCE = {"resnet18": "emb", "grad_pc": "grad"}       # из pseudo_camera_desc.npz
GPU_MS = {"b1": 8.27, "b8": 4.88}       # a10, RTX 4060 FP16, только модель
FPS_TARGET = 100
PENALTY = {"dba": 1, "kr": False, "lambda": float("inf"), "tau": "p90"}


# --------------------------------------------------------------------------
# признак
# --------------------------------------------------------------------------

def small_frame(im: Image.Image, box: tuple) -> tuple:
    """Кадр 128×72 (float32, RGB) и маска фона: False — машина с запасом."""
    f = max(1, min(im.width // SMALL[0], im.height // SMALL[1]))
    s = im.reduce(f) if f > 1 else im
    if s.size != SMALL:                       # BOX: пиксель — среднее своего прямоугольника, маска с запасом его накрывает
        s = s.resize(SMALL, Image.BOX)
    a = np.asarray(s, dtype="float32")
    x, y, w, h = box
    sx, sy = SMALL[0] / im.width, SMALL[1] / im.height
    x0, x1 = int(np.floor((x - pc.PAD * w) * sx)) - 1, int(np.ceil((x + (1 + pc.PAD) * w) * sx)) + 1
    y0, y1 = int(np.floor((y - pc.PAD * h) * sy)) - 1, int(np.ceil((y + (1 + pc.PAD) * h) * sy)) + 1
    bg = np.ones(SMALL[::-1], bool)
    bg[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = False
    return a, bg


def pool(x: np.ndarray, f: int) -> np.ndarray:
    h, w = x.shape[0] // f * f, x.shape[1] // f * f
    return x[:h, :w].reshape(h // f, f, w // f, f, *x.shape[2:]).mean((1, 3))


def blur121(g: np.ndarray) -> np.ndarray:
    p = np.pad(g, 1, mode="edge")
    p = (p[:-2] + 2 * p[1:-1] + p[2:]) / 4
    return (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]) / 4


def erode(m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(m)
    out[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return out


def grad_part(a: np.ndarray, bg: np.ndarray, f: int) -> np.ndarray:
    g = a @ GRAY
    g = np.where(bg, g, g[bg].mean() if bg.any() else 0.0)       # машина — средним фона, не своими пикселями
    p = pool(blur121(g), f)
    gx, gy = np.zeros_like(p), np.zeros_like(p)
    gx[:, 1:-1] = p[:, 2:] - p[:, :-2]
    gy[1:-1] = p[2:] - p[:-2]
    ok = erode(pool(bg.astype("float32"), f) == 1.0)
    return l2(np.where(ok, np.sqrt(gx ** 2 + gy ** 2), 0.0).ravel())


def tiny_part(a: np.ndarray, bg: np.ndarray) -> np.ndarray:
    p = pool(a, 4)
    ok = pool(bg.astype("float32"), 4) == 1.0
    mu, sd = (p[ok].mean(0), p[ok].std(0) + 1e-6) if ok.any() else (0.0, 1.0)
    return l2(np.where(ok[..., None], (p - mu) / sd, 0.0).ravel())


def hist_part(a: np.ndarray, bg: np.ndarray) -> np.ndarray:
    q = np.minimum(a.astype(int) // (256 // BINS), BINS - 1)
    code = (q[..., 0] * BINS + q[..., 1]) * BINS + q[..., 2]
    H, W = bg.shape
    zone = (np.arange(H) * ZONES // H)[:, None] * ZONES + (np.arange(W) * ZONES // W)[None, :]
    h = np.bincount((zone * BINS ** 3 + code)[bg], minlength=PARTS["hist"]).reshape(ZONES * ZONES, -1)
    h = h / np.maximum(h.sum(1, keepdims=True), 1)
    return l2(np.sqrt(h).ravel().astype("float32"))


def describe(im: Image.Image, box: tuple, parts=tuple(PARTS)) -> dict:
    a, bg = small_frame(im, box)
    fn = {"grad64": lambda: grad_part(a, bg, 2), "grad32": lambda: grad_part(a, bg, 4),
          "tiny": lambda: tiny_part(a, bg), "hist": lambda: hist_part(a, bg)}
    return {p: fn[p]().astype("float32") for p in parts}


def signature() -> str:
    """Отпечаток устройства признака: кэш с другим отпечатком пересчитывается."""
    import hashlib
    import inspect

    src = "".join(inspect.getsource(f) for f in (small_frame, pool, blur121, erode, grad_part, tiny_part, hist_part,
                                                 describe))
    return hashlib.sha256(repr((SMALL, ZONES, BINS, PARTS, pc.PAD, GRAY.tolist(), src)).encode()).hexdigest()[:16]


def combine(mats: list) -> np.ndarray:
    """Склейка частей (N, d_i): каждая L2, затем вместе L2 — веса частей равны."""
    return l2(np.concatenate([l2(m) for m in mats], axis=1))


# --------------------------------------------------------------------------
# все кадры (полный декод, как сервис) — в кэш, не в репозиторий
# --------------------------------------------------------------------------

_ZIP = {}


def _chunk(job: tuple) -> dict:
    zip_path, items = job
    if zip_path not in _ZIP:                      # один zip на процесс
        _ZIP[zip_path] = zipfile.ZipFile(zip_path)
    zf = _ZIP[zip_path]
    out ={p: np.empty((len(items), n), "float32") for p, n in PARTS.items()}
    for k, (iid, box) in enumerate(items):
        d = describe(Image.open(io.BytesIO(zf.read(f"images/{iid}.jpg"))).convert("RGB"), box)
        for p in PARTS:
            out[p][k] = d[p]
    return out


def descriptors(rows: list, zip_path: Path, cache: Path, workers: int) -> dict:
    ids = [r["image_id"] for r in rows]
    if cache.exists():
        z = np.load(cache)
        if list(z["ids"]) == ids and "sig" in z.files and str(z["sig"]) == signature():
            return {p: z[p] for p in PARTS}
    if zip_path is None:
        raise SystemExit("признаков фона нет в кэше: нужен --zip dataset.zip")
    items = [(r["image_id"], r["box"]) for r in rows]
    jobs = [(str(zip_path), items[s:s + 64]) for s in range(0, len(items), 64)]
    t0, parts = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, out in enumerate(ex.map(_chunk, jobs)):
            parts.append(out)
            if i % 40 == 0:
                print(f"  фон: {min((i + 1) * 64, len(items))}/{len(items)} кадров, {time.perf_counter() - t0:.0f} с",
                      flush=True)
    res = {p: np.concatenate([o[p] for o in parts]) for p in PARTS}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), sig=np.array(signature()), **res)
    return res


def kind_bg(parts: dict, ids: list, kind: str) -> dict:
    return dict(zip(ids, combine([parts[p] for p in kind.split("+")])))


# --------------------------------------------------------------------------
# время на запрос
# --------------------------------------------------------------------------

def bench(rows: list, zip_path: Path, n: int = 200) -> dict:
    from bench_bigallery_pp import timed          # тянет torch — только здесь

    zf = zipfile.ZipFile(zip_path)
    sample = [r for r in rows if r["split"] == "train"][:40]
    raws = [zf.read(f"images/{r['image_id']}.jpg") for r in sample]
    ims = [Image.open(io.BytesIO(x)).convert("RGB") for x in raws]
    boxes = [r["box"] for r in sample]
    rng = np.random.default_rng(0)
    gal = {kind: l2(rng.standard_normal((bp.TOPK, sum(PARTS[p] for p in kind.split("+")))).astype("float32"))
           for kind in KINDS}
    gal["grad_pc"] = l2(rng.standard_normal((bp.TOPK, pc.GRID[0] * pc.GRID[1])).astype("float32"))
    i = [0]

    def nxt():
        i[0] += 1
        return ims[i[0] % len(ims)], boxes[i[0] % len(ims)]

    def query(kind):
        im, box = nxt()
        parts = kind.split("+")
        d = describe(im, box, parts)
        return gal[kind] @ combine([d[p][None] for p in parts])[0]

    def query_pc():
        im, box = nxt()
        small = im.reduce(pc.DRAFT)
        return gal["grad_pc"] @ pc.grad_descriptor(pc.masked_frame(small, box, small.size[0] / im.size[0]))

    out = {kind: timed(lambda k=kind: query(k), n=n) for kind in KINDS}
    out["grad_pc"] = timed(query_pc, n=n)
    j = [0]

    def decode():
        j[0] += 1
        return Image.open(io.BytesIO(raws[j[0] % len(raws)])).convert("RGB")

    return {"ms": out, "decode_full_ms": timed(decode, n=n // 2), "small_frame_ms": timed(lambda: small_frame(*nxt()), n=n)}


# --------------------------------------------------------------------------
# прогон
# --------------------------------------------------------------------------

def with_bg(blocks: list, bg: dict) -> list:
    """Те же блоки с другим фоном; фон галереи общий для фолдов с той же галереей."""
    out, cache = [], {}
    for b in blocks:
        key = (len(b["G"]), b["G"].index[0], b["G"].index[-1])
        if key not in cache:
            cache[key] = np.stack([bg[i] for i in b["G"].index])
        out.append({**b, "qbg": np.stack([bg[i] for i in b["Q"].index]), "gbg": cache[key]})
    return out


def fps(ms: float) -> dict:
    return {b: 1000 / (g + ms) for b, g in GPU_MS.items()}


def main() -> int:
    ap = bp.data_args(argparse.ArgumentParser())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--report", action="store_true", help="только собрать cheap_bg_<tag>.md из JSON")
    a = ap.parse_args()
    if a.report:
        (a.out_dir / f"cheap_bg_{a.tag}.md").write_text(report_md(a.out_dir, a.tag), encoding="utf-8")
        print(f"-> {a.out_dir / f'cheap_bg_{a.tag}.md'}")
        return 0

    bench_json = HERE / "honest_eval_results" / "bench_bigallery_pp.json"
    if a.zip is None or not bench_json.exists():
        raise SystemExit("нужны --zip (время на запрос меряется на кадрах) и bench_bigallery_pp.json "
                         "(research/bench_bigallery_pp.py — время resnet18)")
    rows = pc.read_rows(a.data_dir)
    ids = [r["image_id"] for r in rows]
    parts = descriptors(rows, a.zip, a.cache_dir / "cheap_bg_desc.npz", a.workers)
    bgs = {name: bp.load_bg(a.cache_dir, k) for name, k in REFERENCE.items()}
    bgs.update({kind: kind_bg(parts, ids, kind) for kind in KINDS})
    timing = bench(rows, a.zip)
    B = json.loads(bench_json.read_text(encoding="utf-8"))
    timing["ms"]["resnet18"] = B["mac_ms"]["resnet18_448x256_forward"]
    timing["resnet18_full_query_ms"] = B["mac_ms"]["background_full_query"]
    timing["resnet18_gpu_fps"] = B["gpu_4060_estimate"]["fps_after"]

    D = he.Data(a)
    params = he.pp_params(a.pp_params)
    folds = he.folds_by_vehicle(sorted(D.hold), a.folds, a.seed)
    kinds = {}
    for name, bg in bgs.items():
        cal = bp.calibrate_tau(D, bg)
        ms = timing.get("ms", {}).get(name)
        kinds[name] = {"dims": int(len(next(iter(bg.values())))), "calibration": cal, "ms": ms,
                       "fps": fps(ms) if ms is not None else None}
        print(f"{name:18s} d={kinds[name]['dims']:5d}  p90: τ {cal['p90']['tau']:.4f} P {cal['p90']['cv_mean']['precision']:.3f} "
              f"R {cal['p90']['cv_mean']['recall']:.3f} | f1: P {cal['f1']['cv_mean']['precision']:.3f} "
              f"R {cal['f1']['cv_mean']['recall']:.3f}  {'' if ms is None else f'{ms:.2f} мс'}", flush=True)

    base = bp.build_blocks(D, folds, a.galleries.split(","), bgs["resnet18"])
    for bl in base.values():
        for b in bl:
            bp.dba_data(b, max(bp.DBA_KS))
    ref = json.loads((a.out_dir / f"bigallery_pp_{a.tag}.json").read_text(encoding="utf-8"))["results"]
    res = {}

    def put(key, gname, blocks, variant, tau):
        src = res.get(key, {}).get(bp.THR_FROM.get(gname))          # 8996: пороги фолдов 9.0k
        res.setdefault(key, {})[gname] = bp.run(blocks, folds, variant, "neardup", tau, params,
                                                src["thresholds"] if src else None)
        return res[key][gname]

    for gname, bl in base.items():
        put("raw | neardup", gname, bl, {"dba": 0, "kr": False, "lambda": 0}, {})
        put("dba k=1 | neardup", gname, bl, {"dba": 1, "kr": False, "lambda": 0}, {})
        for name, bg in bgs.items():
            tau = {rule: kinds[name]["calibration"][rule]["tau"] for rule in ("f1", "p90")}
            r = put(f"dba k=1 + campen {name} | neardup", gname, with_bg(bl, bg), PENALTY, tau)
            print(f"{gname:5s} {name:18s} mAP@10 {r['mAP@10']['mean']:.4f} F1 {r['F1']['mean']:.3f} "
                  f"TNR {r['TNR']['mean']:.3f} без пары→та же камера {r['open_same_cam_top1_all']}/{r['n_open_all']}",
                  flush=True)
        # сверка с bigallery_pp: те же блоки, тот же фон resnet18 — те же числа
        for mine, theirs in (("raw | neardup", "raw | neardup"), ("dba k=1 | neardup", "dba k=1 | neardup"),
                             ("dba k=1 + campen resnet18 | neardup", "dba k=1 + campen τ=p90 λ=inf | neardup")):
            if gname in ref.get(theirs, {}):
                for k in ("mAP@10", "F1", "TNR"):
                    if abs(res[mine][gname][k]["mean"] - ref[theirs][gname][k]["mean"]) > 1e-9:
                        raise AssertionError(f"{mine} {gname} {k} не совпадает с bigallery_pp_{a.tag}.json")
        print(f"{gname}: сверка с bigallery_pp_{a.tag}.json — ок", flush=True)
    w = {k: v.replace(str(he.ROOT), "<repo>").replace(str(Path.home()), "~") for k, v in D.env.items()}
    report = {"tag": a.tag, "weights": w.get("REID_WEIGHTS", ""), "penalty": PENALTY, "gpu_ms": GPU_MS,
              "timing": timing, "kinds": kinds, "results": res}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"cheap_bg_{a.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


# --------------------------------------------------------------------------
# отчёт
# --------------------------------------------------------------------------

LABEL = {"resnet18": "resnet18 по кадру 448×256 (справочно, сейчас)",
         "grad_pc": "grad из pseudo_camera: кадр в 1/4, рамка видна (справочно)",
         "grad64": "градиенты 64×36", "grad32": "градиенты 32×18", "tiny": "цвет 32×18",
         "hist": "гистограммы цвета, зоны 4×4", "grad64+tiny": "градиенты 64×36 + цвет 32×18",
         "grad64+hist": "градиенты 64×36 + гистограммы", "grad32+tiny+hist": "градиенты 32×18 + цвет + гистограммы"}


def choose(kinds: dict) -> str:
    """Рекомендация только по калибровке на обучающих машинах: среди дешёвых признаков
    в бюджете FPS — с точностью «та же камера» ≥ 0.90 на отложенных группах камер
    (цель самого правила τ), из них максимум полноты."""
    fast = [k for k in KINDS if kinds[k]["ms"] is not None and kinds[k]["fps"]["b1"] >= FPS_TARGET] or list(KINDS)
    sure = [k for k in fast if kinds[k]["calibration"]["p90"]["cv_mean"]["precision"] >= 0.90] or fast
    return max(sure, key=lambda k: kinds[k]["calibration"]["p90"]["cv_mean"]["recall"])


def report_md(res_dir: Path, tag: str) -> str:
    R = json.loads((res_dir / f"cheap_bg_{tag}.json").read_text(encoding="utf-8"))
    K, r, T = R["kinds"], R["results"], R["timing"]
    pick = choose(K)
    gal = ("9.0k", "10.8k", "8996")
    m = lambda key, g, k="mAP@10": r[key][g][k]["mean"]
    cv = lambda k, rule: K[k]["calibration"][rule]["cv_mean"]
    order = ["resnet18", "grad_pc", *KINDS]
    lines = [f"# Дешёвый признак фона для штрафа за ту же камеру: {tag}", "",
             f"Веса `{R['weights']}`. Признак — пиксели кадра, который сервис и так декодирует: кадр 1920×1080 "
             "ужат до 128×72, машина (bbox + 10 %) выброшена маской. Правило штрафа то же, что в "
             "`bigallery_pp_" + tag + ".md`: τ — самый низкий порог с точностью «та же камера» ≥ 0.90 на парах "
             "обучающих машин, кандидат с той же камеры уходит в конец списка, поверх DBA k=1, галереи «как сервис». "
             "Ничего не подбиралось на галереях; признак для рекомендации выбран по калибровке и цене. Время — CPU "
             f"этого Мака на запрос (декод кадра не входит: сервис его уже сделал; полный декод PIL — "
             f"{T.get('decode_full_ms', float('nan')):.1f} мс, ужатие с маской — {T.get('small_frame_ms', float('nan')):.2f} мс). "
             f"FPS — оценка: прогон a10 на RTX 4060 ({R['gpu_ms']['b1']} мс на кадр при b1, {R['gpu_ms']['b8']} мс при "
             "b8, только модель) плюс это время последовательно.", "",
             "Воспроизвести: `python research/cheap_bg.py --zip dataset.zip --data-dir <данные> --env "
             f"REID_WEIGHTS=<веса> --tag {tag}`, затем `--report`.", "",
             "## «Та же камера» по фону (пары обучающих машин, проверка по группам камер)", "",
             "| Признак | Размер | τ точн. ≥ 0.90: P / R | τ по F1: P / R | мс/запрос CPU | FPS b1 / b8 (оценка) |",
             "|---|---:|---:|---:|---:|---:|"]
    for k in order:
        ms = K[k]["ms"]
        cost = "—" if ms is None else f"{ms:.2f}"
        f = "—" if ms is None else f"{K[k]['fps']['b1']:.0f} / {K[k]['fps']['b8']:.0f}"
        if k == "resnet18" and "resnet18_full_query_ms" in T:
            cost = f"{ms:.1f} (прогон) – {T['resnet18_full_query_ms']:.1f} (с декодом в 1/4)"
            f += f" на CPU; на GPU ≈ {T['resnet18_gpu_fps']:.0f} при b1 (по FLOPs)" if "resnet18_gpu_fps" in T else ""
        lines.append(f"| {LABEL[k]}{' **← выбран**' if k == pick else ''} | {K[k]['dims']} | "
                     f"{cv(k, 'p90')['precision']:.2f} / {cv(k, 'p90')['recall']:.2f} | "
                     f"{cv(k, 'f1')['precision']:.2f} / {cv(k, 'f1')['recall']:.2f} | {cost} | {f} |")
    lines += ["", "## Итог: DBA k=1 + штраф, галереи «как сервис»", "",
              "| Вариант | mAP@10 9.0k / 10.8k / 8996 (Δ к DBA k=1) | F1 / TNR 9.0k | F1 / TNR 10.8k | F1 / TNR 8996 | "
              "без пары → та же машина с той же камеры, 9.0k / 10.8k | мс/запрос | FPS b1 (оценка) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    dba = "dba k=1 | neardup"
    rows_ = [("raw | neardup", "raw (так сейчас)", None), (dba, "DBA k=1", 0.0)]
    rows_ += [(f"dba k=1 + campen {k} | neardup", f"DBA k=1 + штраф: {LABEL[k]}", K[k]["ms"]) for k in order]
    for key, label, ms in rows_:
        maps = " / ".join(f"{m(key, g):.3f}" + ("" if key in ("raw | neardup", dba) else f" ({m(key, g) - m(dba, g):+.3f})")
                          for g in gal)
        fr = lambda g: (f"{m(key, g, 'F1'):.3f} / {m(key, g, 'TNR'):.3f}" +
                        ("" if key in ("raw | neardup", dba) else
                         f" ({m(key, g, 'F1') - m(dba, g, 'F1'):+.3f} / {m(key, g, 'TNR') - m(dba, g, 'TNR'):+.3f})"))
        same = " / ".join(f"{r[key][g]['open_same_cam_top1_all']} из {r[key][g]['n_open_all']}" for g in ("9.0k", "10.8k"))
        cost = "—" if ms is None else f"{ms:.2f}" if ms < 5 else f"{ms:.1f}"
        f = "—" if ms is None else f"{1000 / (R['gpu_ms']['b1'] + ms):.0f}"
        if key.endswith("campen resnet18 | neardup") and "resnet18_gpu_fps" in T:
            f += f" на CPU / {T['resnet18_gpu_fps']:.0f} на GPU"
        lines.append(f"| {label} | {maps} | {fr('9.0k')} | {fr('10.8k')} | {fr('8996')} | {same} | {cost} | {f} |")
    p, ref = f"dba k=1 + campen {pick} | neardup", "dba k=1 + campen resnet18 | neardup"
    d = lambda key, g, k: m(key, g, k) - m(dba, g, k)
    budget = 1000 / FPS_TARGET - R["gpu_ms"]["b1"]
    two = ("9.0k", "10.8k")
    key_of = lambda k: f"dba k=1 + campen {k} | neardup"
    f1_ok = [k for k in KINDS if all(m(key_of(k), g, "F1") >= m(ref, g, "F1") for g in two)]
    both_ok = [k for k in KINDS if all(m(key_of(k), g, "F1") >= m(ref, g, "F1") and m(key_of(k), g, "TNR") >= m(ref, g, "TNR")
                                       for g in two)]
    loose = max(KINDS, key=lambda k: cv(k, "p90")["recall"])
    lines += ["", "## Вывод", "",
              f"- Выбран «{LABEL[pick]}»: по калибровке на обучающих машинах — точность {cv(pick, 'p90')['precision']:.2f}, "
              f"полнота {cv(pick, 'p90')['recall']:.2f} против {cv('resnet18', 'p90')['precision']:.2f} / "
              f"{cv('resnet18', 'p90')['recall']:.2f} у resnet18; {K[pick]['ms']:.2f} мс CPU на запрос против "
              f"{K['resnet18']['ms']:.0f}+ мс.",
              f"- Вклад штрафа поверх DBA k=1: F1 {d(p, '9.0k', 'F1'):+.3f} / {d(p, '10.8k', 'F1'):+.3f}, TNR "
              f"{d(p, '9.0k', 'TNR'):+.3f} / {d(p, '10.8k', 'TNR'):+.3f} на 9.0k / 10.8k (resnet18: F1 "
              f"{d(ref, '9.0k', 'F1'):+.3f} / {d(ref, '10.8k', 'F1'):+.3f}, TNR {d(ref, '9.0k', 'TNR'):+.3f} / "
              f"{d(ref, '10.8k', 'TNR'):+.3f}); mAP@10 {d(p, '9.0k', 'mAP@10'):+.3f} / {d(p, '10.8k', 'mAP@10'):+.3f} / "
              f"{d(p, '8996', 'mAP@10'):+.3f}. На 8996 без пары всего 7 запросов, её TNR не показатель.",
              f"- Как выбирался признак: условие «точность на отложенных группах камер ≥ 0.90» добавлено после первого "
              "прогона, когда таблицы калибровки и итога уже были видны; без него по полноте выбран бы "
              f"«{LABEL[loose]}» (точность {cv(loose, 'p90')['precision']:.3f}). Вывод от выбора не зависит: F1 поверх DBA "
              f"не ниже, чем с resnet18, на 9.0k и 10.8k у {len(f1_ok)} дешёвых признаков из {len(KINDS)}; F1 и TNR "
              f"одновременно — у {len(both_ok)} ({', '.join(LABEL[k] for k in both_ok) or 'ни у одного'}). У остальных "
              "порог ушёл в сторону F1: TNR ниже, чем с resnet18.",
              f"- FPS: при b1 бюджет CPU на запрос для ≥ {FPS_TARGET} FPS — {budget:.2f} мс поверх {R['gpu_ms']['b1']} мс "
              f"модели; выбранный признак — {K[pick]['ms']:.2f} мс → ≈ {K[pick]['fps']['b1']:.0f} FPS при b1, "
              f"{K[pick]['fps']['b8']:.0f} при b8 (оценка; CPU сервера 4060 не мерили, CPU и GPU не перекрывали). Запас по "
              f"CPU — в {budget / K[pick]['ms']:.1f} раза: если CPU сервера медленнее Мака больше чем во столько раз, при "
              "b1 будет меньше 100 FPS; при b8 запас большой. Признак можно считать на CPU, пока GPU считает "
              "эмбеддинг, — тогда он не добавляет к задержке ничего, но это правка сервиса.",
              "- Что нужно сервису (api/ — не трогал): при добавлении кадра в галерею хранить его фон "
              f"({K[pick]['dims']} float32 = {K[pick]['dims'] * 4 / 1024:.1f} КБ на кадр, 10.8k кадров — "
              f"{K[pick]['dims'] * 4 * 10847 / 2 ** 20:.0f} МБ), на запросе — фон запроса из уже декодированного кадра "
              "и косинусы с фоном top-100 кандидатов, порог τ из JSON.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
