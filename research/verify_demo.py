#!/usr/bin/env python3
"""Прогон демонстрационных случаев через живой сервис — до защиты, а не на ней."""
import csv, json, subprocess
from pathlib import Path

HERE = Path("/opt/case17/app/research")
CROPS = Path("/opt/case17/data/crops")
API = "http://localhost:8000"

sizes = {r["image_id"]: (r["out_w"], r["out_h"])
         for r in csv.DictReader((CROPS / "crops_train.csv").open())}
gal_vid = {r["image_id"]: r["vehicle_id"]
           for r in csv.DictReader((CROPS / "crops_val_gallery.csv").open())}

cases = json.loads((HERE / "results" / "demo_cases.json").read_text())
report = []
for c in cases:
    iid = c["image_id"]
    w, h = sizes[iid]
    f = CROPS / "train" / f"{iid}.jpg"
    args = ["curl", "-s", "-X", "POST", f"{API}/v1/search",
            "-F", f"image=@{f}", "-F", "x=0", "-F", "y=0",
            "-F", f"w={w}", "-F", f"h={h}", "-F", "top_n=5"]
    d = json.loads(subprocess.run(args, capture_output=True, text=True).stdout)

    line = {"label": c["label"], "image_id": iid, "vehicle_id": c["vehicle_id"],
            "refusal": d.get("refusal"), "took_ms": d.get("took_ms")}
    print(f"\n=== {c['label']} === {iid}")
    if d.get("refusal"):
        line["message"] = d.get("message")
        print(f"  ОТКАЗ: {d.get('message')}")
    else:
        cands = d.get("candidates", [])
        hit_rank = None
        for n, cd in enumerate(cands, 1):
            gid = Path(cd["image_path"]).stem
            same = gal_vid.get(gid) == c["vehicle_id"]
            if same and hit_rank is None:
                hit_rank = n
            print(f"  {n}. {gid[:12]}…  уверенность {cd['confidence']:.4f}"
                  f"{'   <- ТА ЖЕ МАШИНА' if same else ''}")
        line["hit_rank"] = hit_rank
        line["top1_conf"] = cands[0]["confidence"] if cands else None
        print(f"  правильный ответ на месте: {hit_rank}, за {d.get('took_ms')} мс")

    # объяснение решения
    ex = ["curl", "-s", "-X", "POST", f"{API}/v1/explain",
          "-F", f"image=@{f}", "-F", "x=0", "-F", "y=0", "-F", f"w={w}", "-F", f"h={h}"]
    e = json.loads(subprocess.run(ex, capture_output=True, text=True).stdout)
    if "grid" in e:
        line["explain"] = {"grid": e["grid"], "took_ms": e["took_ms"]}
        print(f"  объяснение: карта {e['grid']}x{e['grid']} за {e['took_ms']} мс")
    else:
        line["explain"] = {"error": e.get("detail")}
        print(f"  объяснение НЕ РАБОТАЕТ: {e.get('detail')}")
    report.append(line)

p = HERE / "results" / "demo_verified.json"
p.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(f"\nзаписано: {p}")
ok = all((r["refusal"] if "отказ" in r["label"] else (not r["refusal"] and r.get("hit_rank")))
         for r in report)
print("ВСЕ ЧЕТЫРЕ ОТРАБОТАЛИ КАК ЗАДУМАНО" if ok else "ЕСТЬ РАСХОЖДЕНИЯ, смотри выше")
