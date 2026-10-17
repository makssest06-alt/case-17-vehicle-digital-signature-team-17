#!/usr/bin/env python3
"""Аудит утечек и почти-дублей (D3, D4) для честной валидации.

Что проверяется:

  1. Сплит make_split.py (150 машин, seed 42) пересобирается заново и
     сверяется с research/val_ids.txt; дальше — research/audit_generalization.audit
     (пересечение vehicle_id train ∩ holdout, метаданные, точные дубли файлов).
  2. На каких машинах учился студент. В чекпоинте нет списка строк обучения
     (args.rows_csv указывает на D:\\...), но есть прототипы ArcFace
     margin.weight (N_классов × 512). train_distill.py нумерует классы как
     sorted(vehicle_id) строк обучения. Гипотеза «учили на train без val_ids»
     проверяется так: средний вектор каждой машины должен попадать в
     прототип со своим номером в этой нумерации.
  3. Точные дубли (sha256) и перцептивные хэши (pHash, dHash, 64 бита) по
     кропам train + test: пары с малым расстоянием Хэмминга, разложенные по
     видам (train↔test, holdout↔train, внутри holdout и т.д.).
  4. Почти-дубли одной машины: кадры одной (vehicle_id, camera_id) —
     сколько их, насколько они похожи по хэшу и по вектору студента, и
     у скольких query в holdout в gallery лежит такой соседний кадр.

    python research/audit_leaks.py --data-dir /opt/case17/data \\
        --out-dir research/honest_eval_results

--data-dir: train.csv, test_query.csv, test_gallery.csv, crops/crops_train.csv,
crops/{train,query,gallery}/<id>.jpg.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import audit_generalization as ag  # noqa: E402
import service_embeddings as se  # noqa: E402

STUDENT = ROOT / "weights" / "siglip2_distill_ensemble_ep28.pt"
HAMMING = (0, 4, 8)          # пороги для таблицы: точная копия хэша, почти копия, похожие


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# 1. сплит
# --------------------------------------------------------------------------

def rebuild_split(crops_train_csv: Path, out: Path) -> tuple:
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "make_split.py"), "--train-csv", str(crops_train_csv),
                    "--out", str(out)], check=True, capture_output=True, text=True)
    split = read_csv(out / "val_split.csv")
    ids_new = (out / "val_ids.txt").read_text().split()
    ids_repo = (HERE / "val_ids.txt").read_text().split()
    return split, ids_new, ids_repo


# --------------------------------------------------------------------------
# 2. на каких машинах учился студент
# --------------------------------------------------------------------------

def student_identities(train_rows: list, emb: np.ndarray, val_ids: set, weights: Path) -> dict:
    import torch

    ck = torch.load(weights, map_location="cpu", weights_only=False)
    proto = ck["margin"]["weight"].float().numpy()
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    n_cls = proto.shape[0]

    by_vid = defaultdict(list)
    for i, r in enumerate(train_rows):
        by_vid[r["vehicle_id"]].append(i)
    all_vids = sorted(by_vid)                                  # строки, как в train_distill.py
    means = np.stack([emb[by_vid[v]].mean(0) for v in all_vids])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    sim = means @ proto.T                                      # машина × прототип
    best = sim.argmax(1)

    trained_h0 = [v for v in all_vids if v not in val_ids]     # гипотеза: train без val_ids
    lab = {v: k for k, v in enumerate(trained_h0)}
    pos = {v: k for k, v in enumerate(all_vids)}
    match = np.array([best[pos[v]] == lab[v] for v in trained_h0])
    own = np.array([sim[pos[v], lab[v]] for v in trained_h0])
    val_max = np.array([sim[pos[v]].max() for v in all_vids if v in val_ids])
    # Обратное направление: какой машине принадлежит каждый прототип.
    owner = sim.argmax(0)
    owners_in_val = int(sum(all_vids[o] in val_ids for o in owner))
    return {
        "arcface_classes": int(n_cls),
        "train_vehicles_total": len(all_vids),
        "hypothesis": "классы = sorted(vehicle_id train) без research/val_ids.txt",
        "hypothesis_class_count_matches": len(trained_h0) == n_cls,
        "mean_vector_hits_own_prototype_pct": round(float(match.mean()) * 100, 2),
        "own_prototype_cos_median": round(float(np.median(own)), 4),
        "val_vehicles_best_prototype_cos_median": round(float(np.median(val_max)), 4),
        "prototypes_whose_nearest_vehicle_is_in_val": owners_in_val,
        "args": {k: str(v) for k, v in ck.get("args", {}).items()
                 if k in ("rows_csv", "teacher", "init_from", "epochs", "seed")},
    }


# --------------------------------------------------------------------------
# 3. хэши
# --------------------------------------------------------------------------

def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m[0] /= np.sqrt(2)
    return m


_DCT32 = _dct_matrix(32)


def hashes(path: Path) -> tuple:
    """pHash (DCT 32×32 -> 8×8 без постоянной, > медианы) и dHash (9×8, разность соседей)."""
    im = Image.open(path).convert("L")
    a = np.asarray(im.resize((32, 32), Image.LANCZOS), dtype="float64")
    d = _DCT32 @ a @ _DCT32.T
    low = d[:8, :8].flatten()[1:]
    ph = np.concatenate([[False], low > np.median(low)])
    b = np.asarray(im.resize((9, 8), Image.LANCZOS), dtype="float64")
    dh = (b[:, 1:] > b[:, :-1]).flatten()
    return ph, dh


def all_hashes(items: list, cache: Path) -> tuple:
    if cache.exists():
        z = np.load(cache)
        if list(z["ids"]) == [i for i, _ in items]:
            return z["phash"], z["dhash"], z["sha"]
    ph = np.zeros((len(items), 64), bool)
    dh = np.zeros((len(items), 64), bool)
    sha = []
    t0 = time.perf_counter()
    for k, (_, path) in enumerate(items):
        ph[k], dh[k] = hashes(path)
        sha.append(ag.sha256(path))
        if k and k % 2000 == 0:
            print(f"  хэши {k}/{len(items)}  {k / (time.perf_counter() - t0):.0f} кадр/с", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array([i for i, _ in items]), phash=ph, dhash=dh, sha=np.array(sha))
    return ph, dh, np.array(sha)


def close_pairs(bits: np.ndarray, max_dist: int, chunk: int = 1024):
    """Все пары i<j с расстоянием Хэмминга <= max_dist (матричное умножение по кускам)."""
    x = bits.astype("float32")
    y = 1.0 - x
    out = []
    for s in range(0, len(x), chunk):
        same = x[s:s + chunk] @ x.T + y[s:s + chunk] @ y.T
        dist = 64 - same
        ii, jj = np.nonzero(dist <= max_dist)
        keep = jj > ii + s
        out.append(np.stack([ii[keep] + s, jj[keep], dist[ii[keep], jj[keep]]], 1).astype("int64"))
    return np.concatenate(out) if out else np.zeros((0, 3), "int64")


def frame_duplicates(zip_path: Path, train: list, val_ids: set, test_role: dict, pad: float = 0.10) -> dict:
    """Побайтно одинаковые ПОЛНЫЕ кадры под разными image_id (dataset.zip).

    Хэш кропов этого не видит: один кадр размечен дважды — под две разные
    машины с разными рамками, кропы разные. Для пары holdout↔обучение
    считается, какая доля машины holdout попадает в кроп обучающей машины
    (bbox + pad): модель учится на кропах, значит видит эту долю.
    Кандидаты — по CRC32 и размеру из оглавления zip, подтверждение — sha256.
    """
    import hashlib
    import zipfile

    zf = zipfile.ZipFile(zip_path)
    groups = defaultdict(list)
    for info in zf.infolist():
        if info.filename.startswith("images/"):
            groups[(info.CRC, info.file_size)].append(info.filename[len("images/"):-len(".jpg")])
    rows = {r["image_id"]: r for r in train}

    def where(i):
        if i in rows:
            return "holdout" if rows[i]["vehicle_id"] in val_ids else "train1391"
        return test_role.get(i, "unknown")

    def crop_box(r):
        x, y, w, h = (int(float(r[k])) for k in ("x", "y", "w", "h"))
        px, py = round(w * pad), round(h * pad)
        return x - px, y - py, x + w + px, y + h + py

    def share_inside(r_obj, r_crop):
        x, y, w, h = (int(float(r_obj[k])) for k in ("x", "y", "w", "h"))
        c0, c1, c2, c3 = crop_box(r_crop)
        ix = max(0, min(x + w, c2) - max(x, c0))
        iy = max(0, min(y + h, c3) - max(y, c1))
        return ix * iy / (w * h)

    kinds, pairs, holdout_frames = Counter(), [], set()
    for ids in groups.values():
        if len(ids) < 2:
            continue
        if len({hashlib.sha256(zf.read(f"images/{i}.jpg")).hexdigest() for i in ids}) != 1:
            continue                                   # совпал только CRC
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                kind = "↔".join(sorted((where(a), where(b))))
                kinds[kind] += 1
                rec = {"kind": kind, "a": a, "b": b}
                if {where(a), where(b)} == {"holdout", "train1391"}:
                    h, t = (a, b) if where(a) == "holdout" else (b, a)
                    holdout_frames.add(h)
                    rec.update(holdout_vid=rows[h]["vehicle_id"], train_vid=rows[t]["vehicle_id"],
                               holdout_share_in_train_crop=round(share_inside(rows[h], rows[t]), 3))
                pairs.append(rec)
    return {"groups_by_kind": dict(kinds), "pairs": pairs, "holdout_frames": sorted(holdout_frames)}


def pair_sheet(examples: list, paths: dict, out: Path, limit: int = 40) -> None:
    """Подозрительные пары бок о бок — для проверки глазами (в кэш, не в репозиторий)."""
    from PIL import ImageDraw

    ex = sorted(examples, key=lambda e: (not e["both_hashes_le_4"], e["hamming_phash"]))[:limit]
    if not ex:
        return
    w, h = 220, 150
    sheet = Image.new("RGB", (2 * w, len(ex) * (h + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for r, e in enumerate(ex):
        for c, side in enumerate("ab"):
            im = Image.open(paths[e[side]]).convert("RGB")
            im.thumbnail((w, h))
            sheet.paste(im, (c * w, r * (h + 16) + 16))
        # шрифт PIL по умолчанию кириллицу не рисует — подпись латиницей
        d.text((2, r * (h + 16)), f"{e['a_kind']} vid {e['a_vid']} | {e['b_kind']} vid {e['b_vid']} | "
               f"dH {e['hamming_dhash']} pH {e['hamming_phash']}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "honest_eval_results")
    ap.add_argument("--cache-dir", type=Path, default=se.DEFAULT_CACHE)
    ap.add_argument("--weights", type=Path, default=STUDENT)
    ap.add_argument("--near-dup-cos", type=float, default=0.95,
                    help="косинус студента, выше которого кадры одной машины считаем почти-дублями")
    ap.add_argument("--zip", type=Path, default=None,
                    help="dataset.zip организаторов: проверить дубли ПОЛНЫХ кадров")
    a = ap.parse_args()
    d = a.data_dir
    report = {}

    # 1. сплит и аудит Ростислава
    split, ids_new, ids_repo = rebuild_split(d / "crops" / "crops_train.csv", a.cache_dir / "split")
    source = read_csv(d / "crops" / "crops_train.csv")
    base = ag.audit(source, split, d / "crops" / "train", None)
    base["val_ids_rebuilt_equals_repo"] = ids_new == ids_repo
    report["split_audit"] = base
    val_ids = set(ids_repo)

    # векторы студента по всему train (кэш service_embeddings)
    train = read_csv(d / "train.csv")
    emb = se.embed("env_example", d / "crops" / "train", [r["image_id"] for r in train], a.cache_dir)
    report["student_training_identities"] = student_identities(train, emb, val_ids, a.weights)

    # 3. хэши по train + test
    q_rows, g_rows = read_csv(d / "test_query.csv"), read_csv(d / "test_gallery.csv")
    tr_ids = {r["image_id"] for r in train}
    tq, tg = {r["image_id"] for r in q_rows}, {r["image_id"] for r in g_rows}
    report["image_id_overlap"] = {"test_query∩test_gallery": len(tq & tg),
                                  "test∩train": len((tq | tg) & tr_ids),
                                  "holdout_query∩holdout_gallery": len(
                                      {r["image_id"] for r in split if r["role"] == "query"}
                                      & {r["image_id"] for r in split if r["role"] == "gallery"})}
    role = {r["image_id"]: r["role"] for r in split}
    meta = []
    for r in train:
        kind = role.get(r["image_id"]) or ("holdout" if r["vehicle_id"] in val_ids else "train")
        meta.append({"id": r["image_id"], "set": "train", "kind": kind if kind != "train" else "train1391",
                     "vid": r["vehicle_id"], "cam": r["camera_id"], "path": d / "crops" / "train" / f"{r['image_id']}.jpg"})
    for rows, kind in ((q_rows, "test_query"), (g_rows, "test_gallery")):
        for r in rows:
            meta.append({"id": r["image_id"], "set": "test", "kind": kind, "vid": None, "cam": None,
                         "path": d / "crops" / kind.split("_")[1] / f"{r['image_id']}.jpg"})
    ph, dh, sha = all_hashes([(m["id"], m["path"]) for m in meta], a.cache_dir / "hashes_crops.npz")

    def pair_kind(i, j):
        mi, mj = meta[i], meta[j]
        sets = {mi["set"], mj["set"]}
        if sets == {"train", "test"}:
            return "train↔test"
        if mi["set"] == "test":
            return "test: " + "↔".join(sorted({mi["kind"], mj["kind"]}))
        hv = {"query", "gallery", "holdout"}
        in_hold = (mi["kind"] in hv, mj["kind"] in hv)
        if in_hold == (True, False) or in_hold == (False, True):
            return "holdout↔train1391"
        where = "holdout" if all(in_hold) else "train1391"
        if mi["vid"] != mj["vid"]:
            return f"{where}: разные машины"
        return f"{where}: та же машина, " + ("та же камера" if mi["cam"] == mj["cam"] else "другая камера")

    sha_groups = defaultdict(list)
    for k, s in enumerate(sha):
        sha_groups[s].append(k)
    exact = Counter()
    for ks in sha_groups.values():
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                exact[pair_kind(ks[x], ks[y])] += 1
    report["exact_file_duplicates"] = dict(exact)
    if a.zip:
        test_role = {r["image_id"]: "test_query" for r in q_rows}
        test_role.update({r["image_id"]: "test_gallery" for r in g_rows})
        fd = frame_duplicates(a.zip, train, val_ids, test_role)
        report["exact_full_frame_duplicates"] = fd
        # Кадры holdout, побайтно совпадающие с кадром обучающей машины:
        # research/honest_eval.py исключает их по умолчанию.
        (a.out_dir / "holdout_frame_dups.txt").parent.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "holdout_frame_dups.txt").write_text("\n".join(fd["holdout_frames"]) + "\n")

    hash_tables, close = {}, {}
    for name, bits in (("phash", ph), ("dhash", dh)):
        pairs = close_pairs(bits, max(HAMMING))
        table = {}
        for t in HAMMING:
            sel = pairs[pairs[:, 2] <= t]
            table[f"hamming<={t}"] = dict(Counter(pair_kind(i, j) for i, j, _ in sel))
        hash_tables[name] = table
        for i, j, _ in pairs[pairs[:, 2] <= 4]:
            close[(int(i), int(j))] = True
    # Один хэш ошибается на тёмных и бедных текстурой кропах; дублем считаем
    # пару, где ОБА хэша <= 4. Подозрительные пары (чужие машины, train↔test,
    # holdout↔train) пишутся в leak_pairs.csv, если близок хотя бы один хэш.
    both, examples = Counter(), []
    for i, j in close:
        k = pair_kind(i, j)
        d_dh, d_ph = int((dh[i] != dh[j]).sum()), int((ph[i] != ph[j]).sum())
        if d_dh <= 4 and d_ph <= 4:
            both[k] += 1
        if k in ("train↔test", "holdout↔train1391") or "разные машины" in k:
            examples.append({"kind": k, "hamming_dhash": d_dh, "hamming_phash": d_ph,
                             "both_hashes_le_4": d_dh <= 4 and d_ph <= 4,
                             "a": meta[i]["id"], "a_kind": meta[i]["kind"], "a_vid": meta[i]["vid"],
                             "b": meta[j]["id"], "b_kind": meta[j]["kind"], "b_vid": meta[j]["vid"]})
    hash_tables["both_hashes_le_4"] = dict(both)
    report["perceptual_hash_pairs"] = hash_tables

    # 4. почти-дубли одной машины (кадры одной пары машина-камера)
    pos = {r["image_id"]: k for k, r in enumerate(train)}   # train идёт первым в meta: индексы совпадают
    groups = defaultdict(list)
    for k, r in enumerate(train):
        groups[(r["vehicle_id"], r["camera_id"])].append(k)
    # Косинусы раздельно для машин, на которых студент учился (train1391), и
    # отложенных (holdout): на первых он заведомо завышен.
    part = lambda k: "holdout" if train[k]["vehicle_id"] in val_ids else "train1391"
    same_cam_cos, same_cam_dh = defaultdict(list), []
    cross_cam_cos, neg_cos = defaultdict(list), defaultdict(list)
    rng = np.random.default_rng(42)
    for ks in groups.values():
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                same_cam_cos[part(ks[x])].append(float(emb[ks[x]] @ emb[ks[y]]))
                same_cam_dh.append(int((dh[ks[x]] != dh[ks[y]]).sum()))
    by_vid = defaultdict(list)
    for k, r in enumerate(train):
        by_vid[r["vehicle_id"]].append(k)
    for ks in by_vid.values():
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                if train[ks[x]]["camera_id"] != train[ks[y]]["camera_id"]:
                    cross_cam_cos[part(ks[x])].append(float(emb[ks[x]] @ emb[ks[y]]))
    for i, j in zip(rng.integers(0, len(train), 40000), rng.integers(0, len(train), 40000)):
        if train[i]["vehicle_id"] != train[j]["vehicle_id"] and part(i) == part(j):
            neg_cos[part(i)].append(float(emb[i] @ emb[j]))
    q = lambda v: {"n": len(v), "p10": round(float(np.percentile(v, 10)), 4),
                   "median": round(float(np.median(v)), 4), "p90": round(float(np.percentile(v, 90)), 4)}
    by_part = lambda d: {p: q(d[p]) for p in ("holdout", "train1391")}
    same_cam_all = np.array(same_cam_cos["holdout"] + same_cam_cos["train1391"])

    # query holdout, у которых в gallery есть кадр той же машины с той же камеры
    gal_by_vc = {(r["vehicle_id"], r["camera_id"]): r["image_id"] for r in split if r["role"] == "gallery"}
    qd = [float(emb[pos[r["image_id"]]] @ emb[pos[gal_by_vc[(r["vehicle_id"], r["camera_id"])]]])
          for r in split if r["role"] == "query" and (r["vehicle_id"], r["camera_id"]) in gal_by_vc]
    n_query = sum(r["role"] == "query" for r in split)
    report["near_duplicates_same_vehicle"] = {
        "vehicle_camera_groups": len(groups),
        "groups_by_frames": dict(sorted(Counter(len(v) for v in groups.values()).items())),
        "student_cos_same_vehicle_same_camera": by_part(same_cam_cos),
        "student_cos_same_vehicle_other_camera": by_part(cross_cam_cos),
        "student_cos_different_vehicles": by_part(neg_cos),
        "dhash_hamming_same_vehicle_same_camera_median": float(np.median(same_cam_dh)),
        f"same_camera_pairs_cos>={a.near_dup_cos}": int((same_cam_all >= a.near_dup_cos).sum()),
        "same_camera_pairs": int(len(same_cam_all)),
        "holdout_query": n_query,
        "holdout_query_with_same_camera_frame_in_gallery": len(qd),
        f"holdout_query_same_camera_gallery_cos>={a.near_dup_cos}": int((np.array(qd) >= a.near_dup_cos).sum()),
        "holdout_query_same_camera_gallery_cos_median": round(float(np.median(qd)), 4),
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "leak_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (a.out_dir / "leak_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["kind", "hamming_dhash", "hamming_phash", "both_hashes_le_4",
                "a", "a_kind", "a_vid", "b", "b_kind", "b_vid"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(examples, key=lambda e: (e["kind"], e["hamming_dhash"])))
    pair_sheet(examples, {m["id"]: m["path"] for m in meta}, a.cache_dir / "leak_pairs_sheet.jpg")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n-> {a.out_dir / 'leak_audit.json'}, {a.out_dir / 'leak_pairs.csv'} ({len(examples)} пар)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
