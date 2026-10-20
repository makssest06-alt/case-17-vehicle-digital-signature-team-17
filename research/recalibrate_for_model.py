#!/usr/bin/env python3
"""Пороги под другую модель одной командой — тем же кодом, которым они посчитаны для a10.

Модель — из окружения, как у сервиса: EMBED_MODEL, REID_WEIGHTS, INPUT_SIZE (и при необходимости EMBED_DIM,
REID_TF_MODE, EMBED_TTA_FLIP, REID_FP16, REID_DEVICE). Шаги:
  1. эмбеддинги кадров train.csv (holdout и обучающие машины) — research/service_embeddings: код сервиса, кэш
     ~/lct-case17/cache/honest_eval, GPU и FP16, если есть CUDA. Кадры теста не декодируются и не считаются;
  2. признак фона сервиса для кадров train.csv (research/cheap_bg, от модели не зависит);
  3. τ фона — bigallery_pp.calibrate_tau, правило p90: самый низкий τ с точностью «та же камера» ≥ 0.90 на парах
     обучающих машин (как cheap_bg_a10), округление до 5 знаков — как в конфиге;
  4. правило почти-дублей — dedup_compare.calibrate: наибольшая полнота при точности ≥ 0.90 (как dedup_compare_a10);
  5. порог отказа и строгий — closed_gallery.run, протокол q17: галереи 361 и 1111 (361 + 750 кадров обучающих
     машин), у 20 % машин в галерее нет ни одного кадра, максимум 0.7·F1 + 0.3·TNR (Q15), медиана 10 порогов вне
     фолда; строгий — наименьший с TNR ≥ 0.98 на обеих. τ и правило — из шагов 3–4.
Печатает пресет для api/app/config.py и строки .env, mAP@10, F1, TNR и 0.7·F1 + 0.3·TNR на 361 и 1111, сверку с
текущим конфигом; всё — в research/honest_eval_results/recalibrate_<tag>.json.

    EMBED_MODEL=reid:vit_large_patch16_siglip_256 REID_WEIGHTS=/путь/l75.pt INPUT_SIZE=256 \\
        python research/recalibrate_for_model.py --data-dir <данные> --zip dataset.zip --tag l75
    (то же — research/recalibrate_for_model.sh)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "api"))
import bigallery_pp as bp  # noqa: E402
import closed_gallery as cg  # noqa: E402
import dedup_compare as dc  # noqa: E402
import honest_eval as he  # noqa: E402

MODEL_ENV = ("EMBED_MODEL", "REID_WEIGHTS", "INPUT_SIZE", "EMBED_DIM", "REID_TF_MODE", "EMBED_TTA_FLIP",
             "REID_FP16", "REID_DEVICE")
GAL = {"361": "361", "1111tr": "1111"}
# итог с 20.10 — «DBA» без исключения почти-дублей (cg.PRODUCTION); итог 18.10 — запасной (docs/refusal.md)
PRESETS = {"итог": ((1, "off", "off", "raw"), cg.PRODUCTION), "итог 18.10": ((1, "off", "both", "raw"), cg.FINAL),
           "запасной без фона": ((1, "off", "bbox", "raw"), cg.FALLBACK),
           "итог, z100": ((1, "off", "off", "z100"), cg.PRODUCTION), "итог 18.10, z100": ((1, "off", "both", "z100"), cg.FINAL),
           "запасной, z100": ((1, "off", "bbox", "z100"), cg.FALLBACK)}
RULE_ENV = (("s_hi", "PP_DEDUP_SIM"), ("t_iou", "PP_DEDUP_IOU"), ("t_center", "PP_DEDUP_CENTER"),
            ("t_size", "PP_DEDUP_SIZE"))


def say(msg: str) -> None:
    print(msg, flush=True)


def train_bg(a) -> dict:
    """Признак фона сервиса (градиенты 64×36 + гистограммы) только для кадров train.csv — кадры теста не читаются.
    Есть общий кэш cheap_bg_desc.npz с той же сигнатурой — строки берутся из него, иначе считается только train."""
    import cheap_bg as cb
    import pseudo_camera as pc

    rows = [r for r in pc.read_rows(a.data_dir) if r["split"] == "train"]
    ids, kind = [r["image_id"] for r in rows], "+".join(cb.bgm.SERVICE_PARTS)
    full = a.cache_dir / "cheap_bg_desc.npz"
    if full.exists():
        z = np.load(full)
        if "sig" in z.files and str(z["sig"]) == cb.signature():
            pos = {i: k for k, i in enumerate(z["ids"].tolist())}
            if all(i in pos for i in ids):
                idx = np.array([pos[i] for i in ids])
                return cb.kind_bg({p: z[p][idx] for p in cb.PARTS}, ids, kind)
    return cb.kind_bg(cb.descriptors(rows, a.zip, a.cache_dir / "cheap_bg_desc_train.npz", 4), ids, kind)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def num(x) -> str:
    return "inf" if x == float("inf") else f"{x:g}"


def metrics(res: dict, variant: str, presets: dict) -> dict:
    out = {}
    for g, name in GAL.items():
        pg, gv = presets[variant]["per_gallery"][g], res["galleries"][g][variant]
        out[name] = {"N": res["galleries"][g]["N"], "mAP@10": gv["mAP@10"], "mAP@10_raw": res["galleries"][g]["raw"]["mAP@10"],
                     "R1": gv["R1"], "R5": gv["R5"], **{k: pg[k] for k in ("F1", "TNR", "recall", "Q15", "PR-AUC",
                                                                           "strict_TNR", "strict_recall")}}
    return out


def current_config() -> dict:
    from app.config import REFUSAL_PRESETS, Settings

    s = Settings()
    return {"preset": REFUSAL_PRESETS.get(PRESETS["итог"][0]), "tau": s.pp_camera_tau,
            "rule": {"s_hi": s.pp_dedup_sim, "t_iou": s.pp_dedup_iou, "t_center": s.pp_dedup_center,
                     "t_size": s.pp_dedup_size}}


def main() -> int:
    ap = bp.data_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--tag", required=True, help="имя модели в отчёте и в имени JSON")
    ap.add_argument("--out", type=Path, default=None, help="JSON (по умолчанию honest_eval_results/recalibrate_<tag>.json)")
    a = ap.parse_args()
    a.env = [f"{k}={os.environ[k]}" for k in MODEL_ENV if os.environ.get(k)] + a.env      # --env перекрывает окружение
    if a.source == "crops" and not (a.data_dir / "crops" / "train").is_dir():
        if a.zip is None:
            raise SystemExit(f"нет {a.data_dir / 'crops' / 'train'} и не задан --zip: кадрам неоткуда взяться")
        a.source = "frames"                       # полный кадр из zip, кроп кодом сервиса (open_image → crop_frame)
    if a.source == "crops":
        from app.imaging import jpeg_draft_max  # noqa: E402

        draft = dict(e.split("=", 1) for e in a.env).get("EXTRACT_JPEG_DRAFT")
        if jpeg_draft_max(draft) > 1:
            raise SystemExit("кропы <data-dir>/crops нарезаны полным декодом, а сервис режет JPEG с draft "
                             "(app.imaging.crop_frame): --source frames --zip dataset.zip или EXTRACT_JPEG_DRAFT=0")
    cg.set_protocol("q17")
    took, t_all = {}, time.perf_counter()

    D = he.Data(a)
    w = Path(D.env.get("REID_WEIGHTS", ""))
    model = {k: D.env.get(k, "") for k in MODEL_ENV if D.env.get(k, "")}
    model.update(weights_sha256=sha256(w) if w.is_file() else None, weights_bytes=w.stat().st_size if w.is_file() else None)
    say(f"модель {model.get('EMBED_MODEL')}, веса {w} ({model['weights_bytes']} байт, sha256 {model['weights_sha256']}), "
        f"кадры: {a.source}")

    t = time.perf_counter()
    train_ids = [r["image_id"] for r in D.train]
    dim = D.vectors(train_ids).shape[1]
    took["эмбеддинги train.csv"] = time.perf_counter() - t
    say(f"1. эмбеддинги {len(train_ids)} кадров train.csv, размерность {dim}: {took['эмбеддинги train.csv']:.0f} с")

    t = time.perf_counter()
    bg = train_bg(a)
    took["признак фона"] = time.perf_counter() - t
    say(f"2. признак фона {len(bg)} кадров: {took['признак фона']:.0f} с")

    t = time.perf_counter()
    cal_tau = bp.calibrate_tau(D, bg)
    tau = round(float(cal_tau["p90"]["tau"]), 5)
    took["τ фона"] = time.perf_counter() - t
    say(f"3. τ фона {tau} (p90; точность / полнота по группам камер {cal_tau['p90']['cv_mean']['precision']:.3f} / "
        f"{cal_tau['p90']['cv_mean']['recall']:.3f}): {took['τ фона']:.0f} с")

    t = time.perf_counter()
    boxes = {r["image_id"]: tuple(int(float(r[k])) for k in ("x", "y", "w", "h")) for r in D.train}
    cal_rule = dc.calibrate(D, boxes, bg, tau)
    rule = cal_rule["rule"]
    took["правило почти-дублей"] = time.perf_counter() - t
    say(f"4. почти-дубль: косинус ≥ {rule['s_hi']}, IoU ≥ {rule['t_iou']}, сдвиг центра ≤ {num(rule['t_center'])}, "
        f"размер ≤ {num(rule['t_size'])} (точность / полнота {cal_rule['bbox_cv']['precision']:.3f} / "
        f"{cal_rule['bbox_cv']['recall']:.3f}): {took['правило почти-дублей']:.0f} с")

    t = time.perf_counter()
    res = cg.run(a, tau=tau, rule=rule, D=D, bg=bg)
    took["порог отказа (q17)"] = time.perf_counter() - t
    say(f"5. порог отказа, протокол q17: {took['порог отказа (q17)']:.0f} с")

    how = (f"361 и 1111 (+750 кадров обучающих машин), без кадров теста; 0.7·F1 + 0.3·TNR (Q15), 20 % машин без "
           f"пары (Q17); research/recalibrate_for_model.py --tag {a.tag}")
    presets = {}
    for name, (key, variant) in PRESETS.items():
        cp = res["config_presets"][repr(key)]
        presets[name] = {"key": list(key), "balanced": cp["balanced"], "strict": cp["strict"],
                         "note": f"{a.tag}: {variant}, {key[3]}: {how}"}
    fin = presets["итог"]
    report = {"итог": metrics(res, cg.PRODUCTION, res["presets"]), "итог 18.10": metrics(res, cg.FINAL, res["presets"]),
              "запасной без фона": metrics(res, cg.FALLBACK, res["presets"])}
    env_lines = [f"{k}={v}" for k, v in model.items() if k in MODEL_ENV and k not in ("REID_FP16", "REID_DEVICE")]
    env_lines.insert(next(i for i, l in enumerate(env_lines) if l.startswith("REID_WEIGHTS=")) + 1,
                     f"# в контейнере REID_WEIGHTS=/models/{w.name} (том ./weights) или /opt/weights/{w.name} (в образе)")
    env_lines += [f"SEARCH_THRESHOLD={fin['balanced']}", f"SEARCH_THRESHOLD_STRICT={fin['strict']}",
                  f"PP_CAMERA_TAU={tau}"] + [f"{e}={num(rule[k])}" for k, e in RULE_ENV]
    config_lines = [f"    {tuple(p['key'])!r}: {{\"balanced\": {p['balanced']}, \"strict\": {p['strict']},\n"
                    f"        \"note\": {json.dumps(p['note'], ensure_ascii=False)}}},".replace("'", '"')
                    for p in presets.values()]
    cur = current_config()
    took["всего"] = time.perf_counter() - t_all

    say("\n=== api/app/config.py: REFUSAL_PRESETS ===")
    say("\n".join(config_lines))
    say("\n=== .env (или environment сервиса api в docker-compose.yml) ===")
    say("\n".join(env_lines))
    say("\n=== Протокол Q17 (без данных теста): итог — DBA k=1, отказ raw без исключения почти-дублей ===")
    say("| Галерея | mAP@10 (без DBA) | F1 | TNR | полнота | 0.7·F1 + 0.3·TNR | строгий: TNR / полнота |")
    say("|---|---:|---:|---:|---:|---:|---:|")
    for g, m in report["итог"].items():
        say(f"| {g} | {m['mAP@10']:.4f} ({m['mAP@10_raw']:.4f}) | {m['F1']:.3f} | {m['TNR']:.3f} | {m['recall']:.3f} | "
            f"{m['Q15']:.3f} | {m['strict_TNR']:.3f} / {m['strict_recall']:.3f} |")
    say("\n=== Сверка с текущим конфигом ===")
    pairs = [("порог отказа", cur["preset"] and cur["preset"]["balanced"], fin["balanced"]),
             ("строгий", cur["preset"] and cur["preset"]["strict"], fin["strict"]), ("τ фона", cur["tau"], tau)]
    pairs += [(f"почти-дубль {k}", cur["rule"][k], rule[k]) for k, _ in RULE_ENV]
    for name, old, new in pairs:
        say(f"  {name}: {num(old) if old is not None else '—'} -> {num(new)}{'  (совпадает)' if old == new else ''}")
    say("  время: " + ", ".join(f"{k} {v:.0f} с" for k, v in took.items()))

    out = a.out or (a.out_dir / f"recalibrate_{a.tag}.json")
    out.write_text(json.dumps({
        "tag": a.tag, "model": model, "dim": int(dim), "source": a.source, "profile": a.profile, "env_overrides": a.env,
        "protocol": "q17: галереи 361 и 1111 (+750 кадров обучающих машин), 20 % машин без кадров в галерее, "
                    "0.7·F1 + 0.3·TNR (Q15), медиана 10 порогов вне фолда; кадры теста не используются",
        "presets": presets, "tau": tau, "rule": rule, "metrics": report, "env_lines": env_lines,
        "config_lines": config_lines, "current_config": cur, "seconds": {k: round(v, 1) for k, v in took.items()},
        "calibration": {"tau": cal_tau, "rule": cal_rule}, "closed_gallery": res}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    say(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
