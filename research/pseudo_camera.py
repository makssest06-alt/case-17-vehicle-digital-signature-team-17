#!/usr/bin/env python3
"""Псевдо-камеры: восстановить camera_id по фону полного кадра.

В train.csv camera_id есть (это и есть эталон для проверки), в test_*.csv —
нет. Скрипт строит дескриптор фона каждого кадра, кластеризует, подбирает
порог по НАСТОЯЩИМ камерам train и затем раскладывает по тем же кластерам
тестовые кадры. Так видно, (а) насколько точно фон определяет камеру и
(б) сняты ли тестовые машины теми же камерами, что и обучающие.

Дескриптор фона (машина замазана: bbox + 10% заливается средним цветом кадра):
  * emb   — глобальный вектор предобученного resnet18 (timm, ImageNet),
            тот же бэкбон, что у профиля compose без .env;
  * grad  — карта градиентов размытого уменьшенного кадра 64×36 в оттенках
            серого: геометрия сцены (разметка, столбы, бордюры), мало зависит
            от освещения;
  * both  — конкатенация, каждая часть L2-нормирована.

Кадры читаются прямо из dataset.zip (распаковывать 7.5 ГБ не нужно);
JPEG декодируется в 1/4 размера (draft), это ~7 мс на кадр.

    python research/pseudo_camera.py --zip dataset.zip --data-dir /opt/case17/data \\
        --out-dir research/honest_eval_results

Псевдо-камера кадра — камера train, за которую проголосовали 5 ближайших по
фону кадров train (для кадров train — только кадры ДРУГИХ машин). Агломеративные
кластеры считаются справочно: они дробят одну камеру на день/ночь.

Выход: pseudo_camera.csv (image_id → pseudo_cam, доля голосов, косинус до
ближайшего соседа, номер кластера), pseudo_camera.json (точность против
настоящих камер train, сравнение test ↔ train), листы для просмотра глазами —
в кэш, не в репозиторий.
"""
import argparse
import csv
import io
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import service_embeddings as se  # noqa: E402  (только путь кэша)

PAD = 0.10           # тот же запас, что у кропа машины (BBOX_PAD в api/app/config.py)
DRAFT = 4            # декодировать JPEG в 1/4
GRID = (64, 36)      # размер карты градиентов


def read_rows(data_dir: Path) -> list:
    rows = []
    for name, split in (("train.csv", "train"), ("test_query.csv", "test_query"),
                        ("test_gallery.csv", "test_gallery")):
        with (data_dir / name).open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"image_id": r["image_id"], "split": split,
                             "box": tuple(int(float(r[k])) for k in ("x", "y", "w", "h")),
                             "cam": r.get("camera_id", ""), "vid": r.get("vehicle_id", "")})
    return rows


def masked_frame(im: Image.Image, box: tuple, scale: float) -> Image.Image:
    """Кадр с замазанной машиной: bbox + PAD заливается средним цветом кадра."""
    x, y, w, h = box
    x0, y0 = (x - PAD * w) * scale, (y - PAD * h) * scale
    x1, y1 = (x + w * (1 + PAD)) * scale, (y + h * (1 + PAD)) * scale
    mean = tuple(int(c) for c in np.asarray(im).reshape(-1, 3).mean(0))
    out = im.copy()
    ImageDraw.Draw(out).rectangle([x0, y0, x1, y1], fill=mean)
    return out


def grad_descriptor(im: Image.Image) -> np.ndarray:
    g = im.convert("L").filter(ImageFilter.GaussianBlur(2)).resize(GRID, Image.BILINEAR)
    a = np.asarray(g, dtype="float32")
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1] = a[2:] - a[:-2]
    m = np.sqrt(gx ** 2 + gy ** 2).flatten()
    return m / (np.linalg.norm(m) + 1e-8)


def descriptors(rows: list, zip_path: Path, cache: Path, batch: int = 32) -> dict:
    ids = [r["image_id"] for r in rows]
    if cache.exists():
        z = np.load(cache)
        if list(z["ids"]) == ids:
            return {k: z[k] for k in ("emb", "grad", "size")}
    import timm
    import torch

    model = timm.create_model("resnet18", pretrained=True, num_classes=0).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    emb = np.zeros((len(rows), model.num_features), "float32")
    grad = np.zeros((len(rows), GRID[0] * GRID[1]), "float32")
    size = np.zeros((len(rows), 2), "int32")
    zf = zipfile.ZipFile(zip_path)
    t0 = time.perf_counter()
    buf, idx = [], []
    for k, r in enumerate(rows):
        im = Image.open(io.BytesIO(zf.read(f"images/{r['image_id']}.jpg")))
        size[k] = im.size
        im.draft("RGB", (im.size[0] // DRAFT, im.size[1] // DRAFT))
        im = im.convert("RGB")
        m = masked_frame(im, r["box"], im.size[0] / size[k][0])
        grad[k] = grad_descriptor(m)
        buf.append(np.asarray(m.resize((448, 256), Image.BILINEAR), dtype="float32") / 255.0)
        idx.append(k)
        if len(buf) == batch or k == len(rows) - 1:
            x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2)
            with torch.inference_mode():
                f = model((x - mean) / std)
            emb[idx] = torch.nn.functional.normalize(f, dim=1).numpy()
            buf, idx = [], []
        if k and k % 1000 == 0:
            print(f"  кадры {k}/{len(rows)}  {k / (time.perf_counter() - t0):.1f} кадр/с", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), emb=emb, grad=grad, size=size)
    return {"emb": emb, "grad": grad, "size": size}


def combine(d: dict, kind: str) -> np.ndarray:
    if kind == "emb":
        x = d["emb"]
    elif kind == "grad":
        x = d["grad"]
    else:
        x = np.concatenate([d["emb"], d["grad"]], axis=1)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cluster(x: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                   metric="cosine", linkage="average").fit_predict(x)


def scores(true: list, pred: np.ndarray) -> dict:
    from sklearn import metrics as M

    true = np.asarray(true)
    purity = sum(Counter(true[pred == c]).most_common(1)[0][1] for c in np.unique(pred)) / len(true)
    inv = sum(Counter(pred[true == t]).most_common(1)[0][1] for t in np.unique(true)) / len(true)
    return {"clusters": int(len(np.unique(pred))), "true_cameras": int(len(np.unique(true))),
            "ARI": round(float(M.adjusted_rand_score(true, pred)), 4),
            "NMI": round(float(M.normalized_mutual_info_score(true, pred)), 4),
            "purity": round(float(purity), 4), "inverse_purity": round(float(inv), 4)}


def knn_cameras(x: np.ndarray, tr: list, te: list, rows: list, k: int = 5) -> dict:
    """Камера кадра голосованием k ближайших кадров train с известной камерой.

    Проверка на train честная: соседи ищутся только среди кадров ДРУГИХ машин
    (leave-one-vehicle-out) — у тестового кадра в train нет ни своей машины,
    ни соседних кадров того же проезда. Возвращает предсказания для train
    (LOVO) и test, косинус до ближайшего соседа и долю голосов.
    """
    xt = x[tr]
    cams = np.array([rows[k_]["cam"] for k_ in tr])
    vids = np.array([rows[k_]["vid"] for k_ in tr])

    def vote(sim: np.ndarray) -> tuple:
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        part = np.take_along_axis(sim, idx, 1)
        order = np.argsort(-part, axis=1)
        idx = np.take_along_axis(idx, order, 1)
        pred, share = [], []
        for row in cams[idx]:
            c, n = Counter(row).most_common(1)[0]
            pred.append(c)
            share.append(n / k)
        return np.array(pred), np.array(share), np.take_along_axis(sim, idx[:, :1], 1)[:, 0], idx[:, 0]

    out = {}
    pred, share, nn, nn_idx = [], [], [], []
    for s in range(0, len(tr), 1024):
        sim = xt[s:s + 1024] @ xt.T
        sim[vids[s:s + 1024, None] == vids[None, :]] = -np.inf     # своя машина — не сосед
        p, sh, n, i = vote(sim)
        pred.append(p); share.append(sh); nn.append(n); nn_idx.append(i)
    out["train"] = {"pred": np.concatenate(pred), "share": np.concatenate(share),
                    "nn": np.concatenate(nn), "nn_idx": np.concatenate(nn_idx), "true": cams}
    p, sh, n, i = vote(x[te] @ xt.T)
    out["test"] = {"pred": p, "share": sh, "nn": n, "nn_idx": i}
    return out


def knn_sheet(rows: list, tr: list, te: list, knn: dict, zip_path: Path, out: Path,
              n: int = 24, seed: int = 0) -> None:
    """Тестовый кадр и ближайший к нему кадр train — та ли это камера, видно глазами."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(te), size=min(n, len(te)), replace=False)
    zf = zipfile.ZipFile(zip_path)
    tw, th = 288, 162
    sheet = Image.new("RGB", (2 * tw, len(pick) * (th + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for r, t in enumerate(pick):
        pair = (rows[te[t]], rows[tr[knn["test"]["nn_idx"][t]]])
        for c, row in enumerate(pair):
            im = Image.open(io.BytesIO(zf.read(f"images/{row['image_id']}.jpg")))
            im.draft("RGB", (480, 270))
            sheet.paste(im.convert("RGB").resize((tw, th)), (c * tw, r * (th + 14) + 14))
        d.text((2, r * (th + 14)), f"{pair[0]['split']} -> pseudo cam{knn['test']['pred'][t]} "
               f"(vote {knn['test']['share'][t]:.1f}, nn cos {knn['test']['nn'][t]:.3f}) | "
               f"nearest train: cam{pair[1]['cam']}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def contact_sheet(rows: list, labels: np.ndarray, zip_path: Path, out: Path,
                  n_clusters: int = 25, per: int = 6, seed: int = 0) -> list:
    """Лист для проверки глазами: строка = кластер, кадры с подписью настоящей камеры."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for k, c in enumerate(labels):
        by[int(c)].append(k)
    big = [c for c, ks in by.items() if len(ks) >= 3]
    picked = rng.choice(big, size=min(n_clusters, len(big)), replace=False)
    zf = zipfile.ZipFile(zip_path)
    tw, th = 192, 108
    sheet = Image.new("RGB", (per * tw, len(picked) * (th + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for row, c in enumerate(picked):
        ks = rng.choice(by[c], size=min(per, len(by[c])), replace=False)
        for col, k in enumerate(ks):
            im = Image.open(io.BytesIO(zf.read(f"images/{rows[k]['image_id']}.jpg")))
            im.draft("RGB", (480, 270))
            im = im.convert("RGB").resize((tw, th))
            sheet.paste(im, (col * tw, row * (th + 14) + 14))
            tag = f"c{c} cam{rows[k]['cam']}" if rows[k]["cam"] else f"c{c} {rows[k]['split']}"
            d.text((col * tw + 2, row * (th + 14)), tag, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return [int(c) for c in picked]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True, help="dataset.zip организаторов (images/<id>.jpg)")
    ap.add_argument("--data-dir", type=Path, required=True, help="train.csv, test_query.csv, test_gallery.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "honest_eval_results")
    ap.add_argument("--cache-dir", type=Path, default=se.DEFAULT_CACHE)
    ap.add_argument("--thresholds", default="0.1,0.12,0.15,0.18,0.2")
    a = ap.parse_args()

    rows = read_rows(a.data_dir)
    d = descriptors(rows, a.zip, a.cache_dir / "pseudo_camera_desc.npz")
    tr = [k for k, r in enumerate(rows) if r["split"] == "train"]
    te = [k for k, r in enumerate(rows) if r["split"] != "train"]
    true_tr = [rows[k]["cam"] for k in tr]
    report = {"frames": len(rows), "frame_sizes": {f"{w}x{h}": n for (w, h), n in
                                                   Counter(map(tuple, d["size"].tolist())).items()}}

    # 1. подбор дескриптора и порога по настоящим камерам train
    grid = {}
    best = None
    for kind in ("emb", "grad", "both"):
        x = combine(d, kind)[tr]
        for t in (float(v) for v in a.thresholds.split(",")):
            s = scores(true_tr, cluster(x, t))
            grid[f"{kind}@{t}"] = s
            print(f"  {kind:5s} порог {t:.2f}: кластеров {s['clusters']:4d}  ARI {s['ARI']:.4f}  "
                  f"NMI {s['NMI']:.4f}  чистота {s['purity']:.4f}  полнота {s['inverse_purity']:.4f}", flush=True)
            if best is None or s["ARI"] > grid[best]["ARI"]:
                best = f"{kind}@{t}"
    report["train_grid"] = grid
    kind, t = best.split("@")[0], float(best.split("@")[1])
    report["chosen"] = {"descriptor": kind, "threshold": t, **grid[best]}

    # 2. совместная кластеризация train + test тем же порогом
    x = combine(d, kind)
    lab = cluster(x, t)
    report["joint_train_scores"] = scores(true_tr, lab[tr])
    tr_set = set(tr)
    has_train = {c for k, c in enumerate(lab) if k in tr_set}
    test_in_train_clusters = sum(lab[k] in has_train for k in te)
    test_only = {int(lab[k]) for k in te} - has_train
    # ближайший кадр train для каждого тестового и для каждого train (без себя)
    xt = x[tr]
    nn_test = (x[te] @ xt.T).max(1)
    s_tr = xt @ xt.T
    np.fill_diagonal(s_tr, -1)
    nn_tr = s_tr.max(1)
    same_cam_nn = np.array(true_tr)[s_tr.argmax(1)] == np.array(true_tr)
    report["test_vs_train_cameras"] = {
        "test_frames": len(te),
        "test_frames_in_clusters_with_train": int(test_in_train_clusters),
        "clusters_with_only_test_frames": len(test_only),
        "test_frames_in_test_only_clusters": int(sum(lab[k] in test_only for k in te)),
        "nn_cos_test_to_train": {"p10": round(float(np.percentile(nn_test, 10)), 4),
                                 "median": round(float(np.median(nn_test)), 4)},
        "nn_cos_train_to_train_leave_one_out": {"p10": round(float(np.percentile(nn_tr, 10)), 4),
                                                "median": round(float(np.median(nn_tr)), 4)},
        "train_nn_is_same_camera_pct": round(float(same_cam_nn.mean()) * 100, 2),
    }

    # 3. камера голосованием ближайших кадров train (основной способ: кластеры
    # выше дробят одну камеру на день/ночь, а ближайший сосед почти всегда с той же камеры)
    knn = knn_cameras(combine(d, kind), tr, te, rows)
    ktr, kte = knn["train"], knn["test"]
    ok = ktr["pred"] == ktr["true"]
    p01, p05 = np.percentile(ktr["nn"], 1), np.percentile(ktr["nn"], 5)
    bins = np.quantile(ktr["nn"], [0, 0.1, 0.25, 0.5, 0.75, 1.0])
    report["knn_camera"] = {
        "k": 5, "descriptor": kind,
        "train_leave_one_vehicle_out_accuracy": round(float(ok.mean()), 4),
        "train_accuracy_by_nn_cos_quantile": {
            f"{bins[i]:.3f}..{bins[i + 1]:.3f}": round(float(ok[(ktr["nn"] >= bins[i]) & (ktr["nn"] <= bins[i + 1])].mean()), 4)
            for i in range(len(bins) - 1)},
        "train_accuracy_vote_share_ge_0_8": round(float(ok[ktr["share"] >= 0.8].mean()), 4),
        "train_share_vote_ge_0_8": round(float((ktr["share"] >= 0.8).mean()), 4),
        "train_cameras_worst_accuracy": sorted(
            ((c, round(float(ok[ktr["true"] == c].mean()), 3)) for c in np.unique(ktr["true"])),
            key=lambda t: t[1])[:5],
        "nn_cos_train_lovo": {"p01": round(float(p01), 4), "p05": round(float(p05), 4),
                              "median": round(float(np.median(ktr["nn"])), 4)},
        "nn_cos_test": {"p01": round(float(np.percentile(kte["nn"], 1)), 4),
                        "p05": round(float(np.percentile(kte["nn"], 5)), 4),
                        "median": round(float(np.median(kte["nn"])), 4)},
        "test_frames_below_train_p01": int((kte["nn"] < p01).sum()),
        "test_frames_below_train_p05": int((kte["nn"] < p05).sum()),
        "test_vote_share_ge_0_8": round(float((kte["share"] >= 0.8).mean()), 4),
        "test_predicted_cameras": int(len(np.unique(kte["pred"]))),
    }

    # 4. сохранить
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pos_tr = {k: i for i, k in enumerate(tr)}
    pos_te = {k: i for i, k in enumerate(te)}
    with (a.out_dir / "pseudo_camera.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # pseudo_cam — камера train, предсказанная по фону (для train — без кадров
        # своей машины); cluster — номер агломеративного кластера, справочно
        w.writerow(["image_id", "split", "pseudo_cam", "vote_share", "nn_cos", "cluster"])
        for k, r in enumerate(rows):
            src, i = (ktr, pos_tr[k]) if k in pos_tr else (kte, pos_te[k])
            w.writerow([r["image_id"], r["split"], f"cam{src['pred'][i]}",
                        round(float(src["share"][i]), 2), round(float(src["nn"][i]), 4), f"pc{int(lab[k]):04d}"])
    sheet = a.cache_dir / "pseudo_camera_sheet.jpg"
    report["contact_sheet"] = {"path": f"<cache>/{sheet.name}",
                               "clusters": contact_sheet(rows, lab, a.zip, sheet)}
    knn_sheet(rows, tr, te, knn, a.zip, a.cache_dir / "pseudo_camera_knn_sheet.jpg")
    (a.out_dir / "pseudo_camera.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "train_grid"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
