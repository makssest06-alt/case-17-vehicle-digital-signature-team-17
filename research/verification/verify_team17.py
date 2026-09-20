#!/usr/bin/env python3
"""Independent verification of team17/improvement-plan claims.

Checks, on our own val split (551 query / 361 gallery, 150 ids):
  * our protocol (drop the whole camera of the query) vs official junk-only
    protocol (drop only same vehicle_id AND same camera_id);
  * timm center-crop transform (production) vs squash (whole car, Resize SxS);
  * flip-TTA;
  * postprocessing (QE + CSLS + k-reciprocal re-ranking);
  * ensemble geom-squash 0.8 + ep29-base 0.2.

Nothing is taken from the participant's .npy files - embeddings are recomputed
from raw frames and checkpoints.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import backbones
from finetune_clip import FullModel

HERE = Path(__file__).parent
IMAGES = Path("D:/bm/dataset_extracted/images")
TRAIN_CSV = Path("D:/bm/dataset_extracted/train.csv")


# ------------------------------------------------------------------ transforms
def build_transform(spec, mode):
    import timm
    from torchvision import transforms as T

    enc = timm.create_model(spec.model_name, pretrained=False, num_classes=0)
    cfg = timm.data.resolve_data_config({}, model=enc)
    if mode == "timm":
        cfg = dict(cfg)
        cfg["input_size"] = (3, spec.input_size, spec.input_size)
        return timm.data.create_transform(**cfg, is_training=False)
    if mode == "squash":
        mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
        s = spec.input_size
        return T.Compose([
            T.Resize((s, s), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    raise SystemExit(f"unknown mode {mode}")


# --------------------------------------------------------------------- loading
def load_split():
    rows = list(csv.DictReader((HERE / "val_split.csv").open(encoding="utf-8")))
    box = {r["image_id"]: r for r in csv.DictReader(TRAIN_CSV.open(encoding="utf-8"))}
    for r in rows:
        b = box[r["image_id"]]
        r["x"], r["y"], r["w"], r["h"] = b["x"], b["y"], b["w"], b["h"]
    gal = [r for r in rows if r["role"] == "gallery"]
    qry = [r for r in rows if r["role"] == "query"]
    return qry, gal


def crop(img, r, context):
    x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
    px, py = w * context, h * context
    return img.crop((max(0, int(round(x - px))), max(0, int(round(y - py))),
                     min(img.width, int(round(x + w + px))),
                     min(img.height, int(round(y + h + py)))))


def load_model(ckpt, device):
    spec = backbones.get(ckpt.get("args", {}).get("backbone", "siglip_base"))
    import timm
    enc = timm.create_model(spec.model_name, pretrained=False, num_classes=0)
    a = ckpt["args"]
    model = FullModel(enc, spec.dim, a["hidden"], a["out_dim"], a["dropout"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), spec


@torch.inference_mode()
def embed(model, tf, rows, context, device, flip, bs=16):
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        imgs = [crop(Image.open(IMAGES / f"{r['image_id']}.jpg").convert("RGB"), r, context)
                for r in chunk]
        x = torch.stack([tf(im) for im in imgs]).to(device)
        f = model(x)[1]
        if flip:
            f = (f + model(torch.flip(x, dims=[3]))[1]) / 2
        out.append(torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy())
    return np.vstack(out).astype("float32")


# --------------------------------------------------------------------- metrics
def _ap(rel_sorted, at=None):
    if at is not None:
        rel_sorted = rel_sorted[:at]
    nrel = int(rel_sorted.sum())
    if nrel == 0:
        return None
    hits = np.flatnonzero(rel_sorted)
    prec = (np.arange(len(hits)) + 1) / (hits + 1)
    if at is not None:
        return float(prec.sum() / min(nrel, at))
    return float(prec.mean())


def metrics(sim, qry, gal, protocol, at=None):
    g_vid = np.array([r["vehicle_id"] for r in gal])
    g_cam = np.array([r["camera_id"] for r in gal])
    aps, r1, r5, used = [], 0, 0, 0
    for i, q in enumerate(qry):
        if protocol == "ours":
            keep = g_cam != q["camera_id"]
        else:  # official: drop junk (same vehicle AND same camera)
            keep = ~((g_vid == q["vehicle_id"]) & (g_cam == q["camera_id"]))
        if not keep.any():
            continue
        s = sim[i][keep]
        rel = g_vid[keep] == q["vehicle_id"]
        if not rel.any():
            continue
        used += 1
        order = np.argsort(-s)
        rel_sorted = rel[order]
        ap = _ap(rel_sorted, at)
        if ap is not None:
            aps.append(ap)
        r1 += int(rel_sorted[0])
        r5 += int(rel_sorted[:5].any())
    return dict(used=used, mAP=round(float(np.mean(aps)) * 100, 2),
                R1=round(r1 / used * 100, 2), R5=round(r5 / used * 100, 2))


# ---------------------------------------------------------------- postprocess
def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def qq_expand(q, k=1, alpha=1.0):
    if k <= 0:
        return q
    sim = q @ q.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argsort(-sim, axis=1)[:, :k]
    w = np.clip(sim[np.arange(len(q))[:, None], idx], 0, None) ** alpha
    return l2(q + np.einsum("nk,nkd->nd", w, q[idx]))


def csls(sim, k=10):
    if k <= 0:
        return sim
    k = min(k, sim.shape[1] - 1)
    rq = np.sort(sim, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    rg = np.sort(sim, axis=0)[-k:, :].mean(axis=0, keepdims=True)
    return 2 * sim - rq - rg


def k_reciprocal_rerank(q, g, k1=20, k2=6, lambda_value=0.7):
    feat = np.concatenate([q, g], axis=0)
    nq, n = len(q), len(feat)
    orig = np.power(1 - feat @ feat.T, 2)
    V = np.zeros_like(orig, dtype=np.float32)
    rank = np.argsort(orig).astype(np.int32)
    for i in range(n):
        fwd = rank[i, : k1 + 1]
        bwd = rank[fwd, : k1 + 1]
        kr = fwd[np.where(bwd == i)[0]]
        exp = kr
        half = int(np.around(k1 / 2)) + 1
        for c in kr:
            cf = rank[c, :half]
            cr = cf[np.where(rank[cf, :half] == c)[0]]
            if len(cr) and len(np.intersect1d(cr, kr)) > 2 / 3 * len(cr):
                exp = np.append(exp, cr)
        exp = np.unique(exp)
        w = np.exp(-orig[i, exp])
        V[i, exp] = w / np.sum(w)
    if k2 != 1:
        Vq = np.zeros_like(V)
        for i in range(n):
            Vq[i] = np.mean(V[rank[i, :k2]], axis=0)
        V = Vq
    inv = [np.where(V[:, i] != 0)[0] for i in range(n)]
    jac = np.zeros_like(orig, dtype=np.float32)
    for i in range(n):
        tmp = np.zeros(n, dtype=np.float32)
        for j in np.where(V[i] != 0)[0]:
            idx = inv[j]
            tmp[idx] += np.minimum(V[i, j], V[idx, j])
        jac[i] = 1 - tmp / (2 - tmp)
    return (jac * (1 - lambda_value) + orig * lambda_value)[:nq, nq:]


def pipeline(q_sets, g_sets, weights, qq_k=1, csls_k=10, lam=0.7, k1=20, k2=6):
    q = l2(sum(w * l2(e) for w, e in zip(weights, q_sets)))
    g = l2(sum(w * l2(e) for w, e in zip(weights, g_sets)))
    q = qq_expand(q, qq_k)
    sim = -k_reciprocal_rerank(q, g, k1, k2, lam)
    return csls(sim, csls_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=float, default=0.10)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    qry, gal = load_split()
    print(f"query {len(qry)}  gallery {len(gal)}", flush=True)

    ckpts = {
        "geom_ep30": torch.load("D:/bm/models/siglip_vitb16_reid_geom_ep30.pt", map_location="cpu"),
        "ep29": torch.load("D:/bm/models/siglip_vitb16_reid_ep29.pt", map_location="cpu"),
    }

    emb = {}  # (tag, mode) -> (q, g)
    for tag, ck in ckpts.items():
        model, spec = load_model(ck, device)
        for mode in ("timm", "squash"):
            tf = build_transform(spec, mode)
            t0 = time.perf_counter()
            q = embed(model, tf, qry, a.context, device, flip=True, bs=a.bs)
            g = embed(model, tf, gal, a.context, device, flip=True, bs=a.bs)
            emb[(tag, mode)] = (l2(q.astype("float64")), l2(g.astype("float64")))
            print(f"  {tag}/{mode}: {time.perf_counter()-t0:.1f}s", flush=True)
            np.save(HERE / f"v17_{tag}_{mode}_qry.npy", q)
            np.save(HERE / f"v17_{tag}_{mode}_gal.npy", g)
        del model
        torch.cuda.empty_cache()

    def show(label, q, g, pp=False, w=None, qsets=None, gsets=None):
        if pp:
            sim = pipeline(qsets, gsets, w)
        else:
            sim = q @ g.T
        mo = metrics(sim, qry, gal, "ours")
        mf = metrics(sim, qry, gal, "official")
        mf10 = metrics(sim, qry, gal, "official", at=10)
        print(f"{label:38s} | ours mAP {mo['mAP']:6.2f} R1 {mo['R1']:6.2f} "
              f"| off mAP {mf['mAP']:6.2f} R1 {mf['R1']:6.2f} R5 {mf['R5']:6.2f} "
              f"| off@10 {mf10['mAP']:6.2f} (used {mf['used']})", flush=True)
        return sim

    print("\n== base (no postprocessing) ==", flush=True)
    for tag in ("geom_ep30", "ep29"):
        for mode in ("timm", "squash"):
            q, g = emb[(tag, mode)]
            show(f"{tag} {mode}", q, g)

    print("\n== with postprocessing (QE+CSLS+rerank) ==", flush=True)
    for tag in ("geom_ep30", "ep29"):
        for mode in ("timm", "squash"):
            q, g = emb[(tag, mode)]
            show(f"{tag} {mode} +pp", q, g, pp=True, w=[1.0], qsets=[q], gsets=[g])

    print("\n== ensembles (with postprocessing) ==", flush=True)
    qg, gg = emb[("geom_ep30", "squash")]
    qb, gb = emb[("geom_ep30", "timm")]
    q29, g29 = emb[("ep29", "timm")]
    show("geom squash0.8+base0.2 +pp", None, None, pp=True, w=[0.8, 0.2], qsets=[qg, qb], gsets=[gg, gb])
    show("geom-squash0.8+ep29base0.2 +pp", None, None, pp=True, w=[0.8, 0.2], qsets=[qg, q29], gsets=[gg, g29])
    show("geom-squash0.7+ep29base0.3 +pp", None, None, pp=True, w=[0.7, 0.3], qsets=[qg, q29], gsets=[gg, g29])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
