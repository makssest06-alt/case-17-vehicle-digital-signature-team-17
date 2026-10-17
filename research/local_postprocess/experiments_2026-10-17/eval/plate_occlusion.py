"""Does A10 rely on the (blurred) licence-plate area? Occlusion profile on the holdout.

Organizers (FAQ Q48) re-run finalists on a control test where the plate is covered with a solid
fill. We have no plate boxes, so we slide a plate-sized grey box (30 % of bbox width x 10 % of
bbox height) over bbox-relative positions, applied to BOTH queries and gallery before the
service crop, and measure mAP@10 (organizer evaluator, raw cosine). A plate shortcut shows up as
a sharp drop when the box sits at the horizontal centre of the lower body (plate zone) compared
with the same box at the sides (headlights / bumper corners) or higher (bonnet / windshield).
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SCR = Path(__file__).resolve().parents[1]
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
CKPT = r"C:\Users\user\Downloads\clean_student_a10_ep28.pt"
EVALUATOR = Path(r"C:\Users\user\Downloads\evaluate.py")
os.environ.update(REID_WEIGHTS=CKPT, REID_TF_MODE="squash", EMBED_TTA_FLIP="0",
                  EMBED_MODEL="reid:vit_base_patch16_siglip_224.webli", OMP_NUM_THREADS="4")
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCR / "tta" / "api"))
from app.imaging import crop_with_pad  # noqa: E402
from app.models.reid_checkpoint import ReidCheckpointModel  # noqa: E402

REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}
BOX_W, BOX_H = 0.30, 0.10
FILL = (128, 128, 128)
# (name, x-centre, y-centre) in bbox-relative units
CONFIGS = [("none", None, None)] + \
          [(f"centre y={y}", 0.5, y) for y in (0.35, 0.55, 0.65, 0.75, 0.85)] + \
          [(f"left y={y}", 0.18, y) for y in (0.65, 0.75)] + \
          [(f"right y={y}", 0.82, y) for y in (0.65, 0.75)] + \
          [("centre band y=0.55-0.95 (generous plate mask)", "band", None)]


def rows(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def masked_crop(img: Image.Image, b: dict, cx, cy) -> Image.Image:
    x, y, w, h = (int(b[k]) for k in "xywh")
    if cx is not None:
        im = img.copy()
        d = ImageDraw.Draw(im)
        if cx == "band":
            d.rectangle([x + 0.33 * w, y + 0.55 * h, x + 0.67 * w, y + 0.95 * h], fill=FILL)
        else:
            d.rectangle([x + (cx - BOX_W / 2) * w, y + (cy - BOX_H / 2) * h,
                         x + (cx + BOX_W / 2) * w, y + (cy + BOX_H / 2) * h], fill=FILL)
        img = im
    return crop_with_pad(img, x, y, w, h, 0.10)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ev", EVALUATOR)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    train = {r["image_id"]: r for r in rows(D / "train.csv")}
    split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
    seen, hg, hq = set(), [], []
    for r in split:
        key = (r["vehicle_id"], r["camera_id"])
        (hq if key in seen else hg).append(r)
        seen.add(key)
    tgal = rows(D / "test_gallery.csv")
    items = [(r["image_id"], train[r["image_id"]]) for r in hq + hg] + [(r["image_id"], r) for r in tgal]
    frames = {}
    for iid, _ in items:
        with Image.open(D / "dataset" / "images" / f"{iid}.jpg") as im:
            frames[iid] = im.convert("RGB")
    m = ReidCheckpointModel("vit_base_patch16_siglip_224.webli", CKPT, 224)
    m.load()
    qids = [r["image_id"] for r in hq]; hgids = [r["image_id"] for r in hg]; tgids = [r["image_id"] for r in tgal]
    qdf = pd.DataFrame(hq).set_index("image_id")
    gal = {"361": hgids, "1111": hgids + tgids}
    gdfs = {k: pd.DataFrame([{"image_id": i, "vehicle_id": train[i]["vehicle_id"], "camera_id": train[i]["camera_id"]}
                             if i in train else {"image_id": i, "vehicle_id": "x_" + i, "camera_id": "x_" + i}
                             for i in v]).set_index("image_id") for k, v in gal.items()}
    base = {}
    print(f"{'occluder':48s} {'mAP@10 361':>11s} {'Δ':>7s} {'mAP@10 1111':>12s} {'Δ':>7s}")
    for name, cx, cy in CONFIGS:
        emb = {}
        for s in range(0, len(items), 64):
            chunk = items[s:s + 64]
            vecs = m.embed_batch([masked_crop(frames[i], b, cx, cy) for i, b in chunk])
            emb.update({i: v for (i, _), v in zip(chunk, vecs)})
        q = np.stack([emb[i] for i in qids])
        cells = []
        for k, gids in gal.items():
            g = np.stack([emb[i] for i in gids])
            order = np.argsort(-(q @ g.T), axis=1, kind="stable")[:, :10]
            r = ev.ranking_metrics(qdf, gdfs[k], {qid: [gids[j] for j in order[n]] for n, qid in enumerate(qids)}, top_k=10)
            base.setdefault(k, r["mAP@10"])
            cells.append((r["mAP@10"], r["mAP@10"] - base[k]))
        print(f"{name:48s} {cells[0][0]:11.4f} {cells[0][1]:+7.4f} {cells[1][0]:12.4f} {cells[1][1]:+7.4f}", flush=True)


if __name__ == "__main__":
    main()
