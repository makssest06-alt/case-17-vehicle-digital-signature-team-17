#!/usr/bin/env python3
"""Обучение ReID-головы поверх замороженных признаков CLIP. Только CPU.

Сплит: 150 машин из val_ids.txt — валидация, все остальные — обучение головы.
В валидацию не подглядываем: ни порог, ни выбор эпохи по ней не настраиваем
внутри обучения, метрики считаются в конце для отчёта.
"""
import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

import backbones
from head_model import MarginHead, ReidHead, batch_hard_triplet

HERE = Path(__file__).parent
FEAT = Path("/opt/case17/data/features")
MODELS = Path("/opt/case17/models")

# Те же пресеты, что в ноутбуке. baseline — как обучались первые головы,
# viewpoint и centroid целятся ровно в проблему ракурса из ERROR_ANALYSIS.md.
PRESETS = {
    "baseline":  dict(cross_camera=False, hard_batches=False, centroid_weight=0.0, tri_weight=1.0),
    "viewpoint": dict(cross_camera=True,  hard_batches=True,  centroid_weight=0.0, tri_weight=1.0),
    "centroid":  dict(cross_camera=True,  hard_batches=True,  centroid_weight=1.0, tri_weight=0.5),
}


def load_train_features(backbone: str = "clip_vitb16"):
    idx = list(csv.DictReader((FEAT / "train_index.csv").open()))
    orig = np.load(FEAT / f"train_{backbone}_orig.npy")
    flip = np.load(FEAT / f"train_{backbone}_flip.npy")
    val_ids = {l.strip() for l in (HERE / "val_ids.txt").read_text().split() if l.strip()}
    tr_rows = [(i, r) for i, r in enumerate(idx) if r["vehicle_id"] not in val_ids]
    va_rows = [(i, r) for i, r in enumerate(idx) if r["vehicle_id"] in val_ids]
    return orig, flip, tr_rows, va_rows


def build_vehicle_neighbors(orig, rows, top=64):
    """Для каждой машины — список визуально похожих машин по центроиду признаков.

    Считается один раз по замороженным признакам, обучения не требует.
    Нужно для трудных батчей: если в батче собраны похожие машины,
    batch-hard triplet получает осмысленные трудные негативы, а не
    случайный седан против случайного грузовика.
    """
    by = defaultdict(list)
    for _, (i, r) in enumerate(rows):
        by[r["vehicle_id"]].append(i)
    vids = sorted(by)
    cent = np.stack([orig[by[v]].mean(0) for v in vids])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    sim = cent @ cent.T
    np.fill_diagonal(sim, -2.0)
    order = np.argsort(-sim, axis=1)[:, :top]
    return {vids[i]: [vids[j] for j in order[i]] for i in range(len(vids))}


class PKSampler:
    """P машин по K кадров в батче — обязательное условие для batch-hard triplet.

    Два режима поверх этого:
      * cross_camera — K кадров набираются с максимума разных камер, чтобы
        позитивная пара внутри батча почти всегда была кросс-камерной;
      * neighbors — машины в батче берутся не случайно, а группой похожих.
    """

    def __init__(self, rows, P=32, K=4, seed=42, cross_camera=False, neighbors=None):
        self.by_vid = defaultdict(list)
        self.by_vid_cam = defaultdict(lambda: defaultdict(list))
        for pos, (i, r) in enumerate(rows):
            self.by_vid[r["vehicle_id"]].append(pos)
            self.by_vid_cam[r["vehicle_id"]][r["camera_id"]].append(pos)
        self.vids = sorted(self.by_vid)
        self.P, self.K = P, K
        self.rng = random.Random(seed)
        self.cross_camera = cross_camera
        self.neighbors = neighbors

    def _pick(self, v):
        """K кадров одной машины."""
        if not self.cross_camera:
            pool = self.by_vid[v]
            return (self.rng.sample(pool, self.K) if len(pool) >= self.K
                    else [self.rng.choice(pool) for _ in range(self.K)])
        # по очереди с каждой камеры: сначала по одному кадру с разных камер,
        # и только когда камеры кончились — второй кадр с уже использованных
        buckets = []
        for pool in self.by_vid_cam[v].values():
            b = pool[:]
            self.rng.shuffle(b)
            buckets.append(b)
        self.rng.shuffle(buckets)
        out = []
        while len(out) < self.K:
            took = False
            for b in buckets:
                if b:
                    out.append(b.pop())
                    took = True
                    if len(out) == self.K:
                        break
            if not took:   # кадров у машины меньше K — добираем с повтором
                out.append(self.rng.choice(self.by_vid[v]))
        return out

    def _batch(self, group):
        batch = []
        for v in group:
            batch += self._pick(v)
        return batch

    def __iter__(self):
        vids = self.vids[:]
        self.rng.shuffle(vids)
        if self.neighbors is None:
            for s in range(0, len(vids) - self.P + 1, self.P):
                yield self._batch(vids[s : s + self.P])
            return
        left = set(vids)
        for seed_vid in vids:
            if len(left) < self.P:
                break          # хвост короче батча отбрасываем, как и в обычном режиме
            if seed_vid not in left:
                continue
            group = [seed_vid]
            left.discard(seed_vid)
            for nb in self.neighbors.get(seed_vid, ()):
                if len(group) == self.P:
                    break
                if nb in left:
                    group.append(nb)
                    left.discard(nb)
            if len(group) < self.P:   # соседи кончились — добираем случайными
                extra = self.rng.sample(sorted(left), self.P - len(group))
                left.difference_update(extra)
                group += extra
            yield self._batch(group)

    def __len__(self):
        return len(self.vids) // self.P


def centroid_loss(feat, y, margin=0.3):
    """Триплет к центроидам: кадр тянем к среднему своей машины, отталкиваем от чужих.

    Обычный batch-hard triplet сравнивает кадр с кадром, поэтому вид спереди
    и вид сзади одной машины он честно считает далёкой парой и тянет их друг
    к другу рывком. Центроид усредняет ракурсы: кадр тянется к «средней»
    машине, а не к конкретному чужому ракурсу — сходимость мягче, а разрыв
    между ракурсами закрывается целенаправленно.

    Центроид считается по всем K кадрам машины в батче, включая сам кадр:
    при K=4 вклад собственного кадра — четверть, смещение невелико, зато
    не нужен отдельный проход leave-one-out.
    """
    labs, inv = torch.unique(y, return_inverse=True)
    cent = torch.zeros(len(labs), feat.size(1), dtype=feat.dtype)
    cent.index_add_(0, inv, feat)
    cnt = torch.zeros(len(labs), dtype=feat.dtype).index_add_(
        0, inv, torch.ones(len(inv), dtype=feat.dtype))
    cent = F.normalize(cent / cnt[:, None], p=2, dim=1)
    f = F.normalize(feat, p=2, dim=1)
    d = torch.cdist(f, cent)
    pos = d.gather(1, inv[:, None]).squeeze(1)
    neg = d.scatter(1, inv[:, None], float("inf")).min(1).values
    return F.relu(pos - neg + margin).mean()


def evaluate_val(head, orig, flip, va_rows, device="cpu"):
    """Кросс-камерная метрика на val: gallery — по одному кадру с каждой камеры."""
    head.eval()
    by_v = defaultdict(lambda: defaultdict(list))
    for pos, (i, r) in enumerate(va_rows):
        by_v[r["vehicle_id"]][r["camera_id"]].append((i, r))
    gal, qry = [], []
    for v in sorted(by_v):
        for cam in sorted(by_v[v], key=int):
            items = sorted(by_v[v][cam], key=lambda t: t[1]["image_id"])
            gal.append(items[0]); qry += items[1:]

    def emb(rows):
        idxs = [i for i, _ in rows]
        with torch.inference_mode():
            a = head(torch.from_numpy(orig[idxs]))[1]
            b = head(torch.from_numpy(flip[idxs]))[1]
        return F.normalize((a + b) / 2, p=2, dim=1).numpy()

    q, g = emb(qry), emb(gal)
    g_vid = np.array([r["vehicle_id"] for _, r in gal])
    g_cam = np.array([r["camera_id"] for _, r in gal])
    sim = q @ g.T

    aps, r1, r5, used, pos_s, neg_s = [], 0, 0, 0, [], []
    for i, (_, qr) in enumerate(qry):
        keep = g_cam != qr["camera_id"]
        rel = g_vid[keep] == qr["vehicle_id"]
        if not keep.any() or not rel.any():
            continue
        used += 1
        s = sim[i][keep]; order = np.argsort(-s); rs = rel[order]
        hits = np.flatnonzero(rs)
        aps.append(float(((np.arange(len(hits)) + 1) / (hits + 1)).mean()))
        r1 += int(rs[0]); r5 += int(rs[:5].any())
        (pos_s if rs[0] else neg_s).append(float(s[order[0]]))
    head.train()
    return {"mAP": round(float(np.mean(aps)) * 100, 2), "rank1": round(r1 / used * 100, 2),
            "rank5": round(r5 / used * 100, 2), "queries_used": used}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip_vitb16")
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--loss", default="arcface", choices=["arcface", "cosface"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    # пресеты — те же, что в kaggle/train_reid.ipynb, чтобы цифры на CPU и на GPU
    # относились к одной и той же конфигурации. Геометрических аугментаций тут нет:
    # признаки посчитаны заранее и замороженный бэкбон переснять нельзя, поэтому на
    # CPU проверяются ровно три из четырёх механизмов пресета.
    ap.add_argument("--preset", default="baseline", choices=sorted(PRESETS),
                    help="baseline | viewpoint | centroid")
    ap.add_argument("--cross-camera", dest="cross_camera", action="store_true", default=None,
                    help="K кадров с максимума разных камер")
    ap.add_argument("--hard-batches", dest="hard_batches", action="store_true", default=None,
                    help="батч из визуально похожих машин")
    ap.add_argument("--centroid-weight", type=float, default=None)
    ap.add_argument("--tri-weight", type=float, default=None)
    a = ap.parse_args()

    # пресет задаёт значения, явный флаг их перебивает
    cfg = dict(PRESETS[a.preset])
    for k in ("cross_camera", "hard_batches", "centroid_weight", "tri_weight"):
        if getattr(a, k) is None:
            setattr(a, k, cfg[k])

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.set_num_threads(2)

    spec = backbones.get(a.backbone)
    orig, flip, tr_rows, va_rows = load_train_features(a.backbone)
    vids = sorted({r["vehicle_id"] for _, r in tr_rows})
    vid2lab = {v: i for i, v in enumerate(vids)}
    print(f"[{a.tag}] train: {len(tr_rows)} кадров / {len(vids)} машин | val: {len(va_rows)} кадров", flush=True)

    head = ReidHead(spec.dim, a.hidden, a.out_dim, a.dropout)
    margin = MarginHead(a.out_dim, len(vids), a.loss, margin=a.margin)
    opt = AdamW(list(head.parameters()) + list(margin.parameters()), lr=a.lr, weight_decay=a.wd)

    neighbors = None
    if a.hard_batches:
        t_nb = time.perf_counter()
        neighbors = build_vehicle_neighbors(orig, tr_rows)
        print(f"  трудные батчи: соседи по центроидам посчитаны за "
              f"{time.perf_counter() - t_nb:.1f} с", flush=True)
    sampler = PKSampler(tr_rows, a.P, a.K, a.seed,
                        cross_camera=a.cross_camera, neighbors=neighbors)
    print(f"[{a.tag}] пресет {a.preset}: cross_camera={a.cross_camera} "
          f"hard_batches={a.hard_batches} centroid_weight={a.centroid_weight} "
          f"tri_weight={a.tri_weight}", flush=True)
    steps = max(1, len(sampler))
    rng = random.Random(a.seed)
    t0 = time.perf_counter()

    for ep in range(a.epochs):
        # warmup затем cosine
        if ep < a.warmup:
            lr = a.lr * (ep + 1) / a.warmup
        else:
            p = (ep - a.warmup) / max(1, a.epochs - a.warmup)
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * p))
        for gparam in opt.param_groups:
            gparam["lr"] = lr

        tot_c, tot_t, nb = 0.0, 0.0, 0
        for batch in sampler:
            # аугментация на уровне признаков: оригинал или отражение
            idxs = [tr_rows[p][0] for p in batch]
            src = np.where(np.array([rng.random() < 0.5 for _ in idxs]), 1, 0)
            x = np.where(src[:, None] == 1, flip[idxs], orig[idxs])
            x = torch.from_numpy(np.ascontiguousarray(x))
            y = torch.tensor([vid2lab[tr_rows[p][1]["vehicle_id"]] for p in batch])

            feat, feat_bn = head(x)
            l_cls = margin(feat_bn, y)
            l_tri = batch_hard_triplet(feat, y, 0.3)
            loss = l_cls + a.tri_weight * l_tri
            if a.centroid_weight > 0:
                loss = loss + a.centroid_weight * centroid_loss(feat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_c += l_cls.item(); tot_t += l_tri.item(); nb += 1

        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            print(f"  эпоха {ep+1}/{a.epochs}  lr {lr:.2e}  cls {tot_c/nb:.3f}  tri {tot_t/nb:.3f}", flush=True)

    train_s = time.perf_counter() - t0
    res = evaluate_val(head, orig, flip, va_rows)
    res.update(tag=a.tag, backbone=a.backbone, loss=a.loss, out_dim=a.out_dim, hidden=a.hidden, epochs=a.epochs,
               P=a.P, K=a.K, margin=a.margin, train_s=round(train_s, 1),
               preset=a.preset, cross_camera=a.cross_camera, hard_batches=a.hard_batches,
               centroid_weight=a.centroid_weight, tri_weight=a.tri_weight,
               head_params_mb=round(sum(p.numel() * 4 for p in head.parameters()) / 1024**2, 2))
    print(f"  val: mAP {res['mAP']}  Rank-1 {res['rank1']}  Rank-5 {res['rank5']}  "
          f"(обучение {train_s:.0f} с)", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    ck_args = dict(vars(a)); ck_args["in_dim"] = spec.dim
    # Имя timm-модели кладём рядом с ключом реестра: по ключу образ не собрать,
    # а установщик весов не обязан знать про наш реестр.
    ck_args["backbone_model_name"] = spec.model_name
    torch.save({"head": head.state_dict(), "args": ck_args}, MODELS / f"head_{a.tag}.pt")
    out = HERE / "results"; out.mkdir(exist_ok=True)
    (out / f"head_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
