#!/usr/bin/env python3
"""Подбор демонстрационных случаев ЧЕРЕЗ ЖИВОЙ СЕРВИС.

Признаки, посчитанные офлайн, и ответ сервиса когда-то расходились
(косинус 0.57 на одном и том же кадре) — голова училась на нормированных
выходах бэкбона, а сервис подавал сырые. Баг исправлен, но правило
осталось: каждый кандидат проверяется тем же запросом, который сделает
оператор, а не выгрузкой признаков.
"""
import csv, json, subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path("/opt/case17/app/research")
CROPS = Path("/opt/case17/data/crops")
API = "http://localhost:8000"
THR = 0.5114

sizes = {r["image_id"]: (r["out_w"], r["out_h"])
         for r in csv.DictReader((CROPS / "crops_train.csv").open())}
gal_vid = {r["image_id"]: r["vehicle_id"]
           for r in csv.DictReader((CROPS / "crops_val_gallery.csv").open())}
vs = list(csv.DictReader((HERE / "val_split.csv").open()))
qry = [r for r in vs if r["role"] == "query"]

removed = set()
rf = HERE / "results" / "demo_refusal.json"
if rf.exists():
    removed = set(json.loads(rf.read_text()).get("removed_from_gallery", []))


def brightness(iid):
    im = Image.open(CROPS / "train" / f"{iid}.jpg").convert("L").resize((64, 64))
    return float(np.asarray(im).mean())


def search(r):
    w, h = sizes[r["image_id"]]
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/v1/search",
         "-F", f"image=@{CROPS/'train'/(r['image_id']+'.jpg')}",
         "-F", "x=0", "-F", "y=0", "-F", f"w={w}", "-F", f"h={h}",
         "-F", "top_n=5", "-F", "threshold=0.0"],
        capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


rng = np.random.default_rng(7)
probe = [qry[i] for i in rng.choice(len(qry), size=140, replace=False)]
rows = []
for r in probe:
    d = search(r)
    c = d.get("candidates") or []
    if not c:
        continue
    hit = None
    for n, cd in enumerate(c, 1):
        if gal_vid.get(Path(cd["image_path"]).stem) == r["vehicle_id"]:
            hit = n
            break
    rows.append({"image_id": r["image_id"], "vehicle_id": r["vehicle_id"],
                 "camera_id": r["camera_id"], "top1": c[0]["confidence"],
                 "hit_rank": hit,
                 "hit_conf": next((x["confidence"] for x in c
                                   if gal_vid.get(Path(x["image_path"]).stem) == r["vehicle_id"]), None)})
print(f"проверено запросов через сервис: {len(rows)}")

accepted = [x for x in rows if x["top1"] >= THR and x["hit_rank"]]
print(f"принято и найдено: {len(accepted)}")

easy = max((x for x in accepted if x["hit_rank"] == 1), key=lambda x: x["top1"])
# сложный: принят, правильный ответ найден, но уверенность в нём минимальна
hard = min((x for x in accepted if x["image_id"] != easy["image_id"]),
           key=lambda x: x["hit_conf"])
dark = sorted((x for x in accepted
               if x["image_id"] not in (easy["image_id"], hard["image_id"])),
              key=lambda x: brightness(x["image_id"]))
night = dark[0]
for c in (easy, hard, night):
    c["brightness"] = round(brightness(c["image_id"]), 1)

nomatch = json.loads(rf.read_text()) if rf.exists() else None

cases = [("лёгкий", easy), ("сложный ракурс", hard), ("ночь", night)]
out = []
for label, c in cases:
    print(f"\n=== {label} ===")
    print(f"  {c['image_id']}  машина {c['vehicle_id']}  яркость {c['brightness']}")
    print(f"  лучший кандидат {c['top1']:.4f}, правильный на месте {c['hit_rank']} "
          f"с уверенностью {c['hit_conf']:.4f}")
    out.append({"label": label, **c})
if nomatch:
    print(f"\n=== без пары (отказ) ===\n  {nomatch['image_id']}  машина {nomatch['vehicle_id']}")
    out.append({"label": "без пары (отказ)", **nomatch})

(HERE / "results" / "demo_cases.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\nзаписано: {HERE / 'results' / 'demo_cases.json'}")
