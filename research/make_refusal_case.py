#!/usr/bin/env python3
"""Корректный пример отказа: машина, которой в галерее нет вовсе.

Первая попытка брала «посторонние» машины из обучающей выборки — и сервис
принимал их все подряд. Причина не в пороге: модель эти машины видела при
обучении, их признаки лежат ближе к выученному многообразию, и сходство
с чем-нибудь из галереи получается завышенным. Валидным отрицательным
запросом может быть только отложенная машина, которой в галерее нет —
ровно так строилась калибровка порога.

Скрипт убирает все кадры одной валидационной машины из галереи и проверяет,
что её запрос теперь получает честный отказ.
"""
import csv, json, subprocess
from collections import defaultdict
from pathlib import Path

CROPS = Path("/opt/case17/data/crops")
HERE = Path("/opt/case17/app/research")
API = "http://localhost:8000"


def api(path, *args):
    out = subprocess.run(["curl", "-s", *args, f"{API}{path}"],
                         capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else {}


def search(image_id, w, h, thr=None):
    f = CROPS / "train" / f"{image_id}.jpg"
    a = ["-X", "POST", "-F", f"image=@{f}", "-F", "x=0", "-F", "y=0",
         "-F", f"w={w}", "-F", f"h={h}", "-F", "top_n=5"]
    if thr is not None:
        a += ["-F", f"threshold={thr}"]
    return api("/v1/search", *a)


vs = list(csv.DictReader((HERE / "val_split.csv").open()))
sizes = {r["image_id"]: (r["out_w"], r["out_h"])
         for r in csv.DictReader((CROPS / "crops_train.csv").open())}
by_vid = defaultdict(lambda: {"gallery": [], "query": []})
for r in vs:
    by_vid[r["vehicle_id"]][r["role"]].append(r["image_id"])

# берём машину, у которой есть и галерея, и запросы
victim = next(v for v, d in sorted(by_vid.items()) if d["gallery"] and d["query"])
gal_ids, qry_ids = by_vid[victim]["gallery"], by_vid[victim]["query"]
print(f"убираю из галереи машину {victim}: {len(gal_ids)} кадров, "
      f"остаётся {len(qry_ids)} запросов без пары")

items = api("/v1/gallery?limit=500&offset=0").get("items", [])
to_del = [it["id"] for it in items if Path(it["image_path"]).stem in set(gal_ids)]
for i in to_del:
    subprocess.run(["curl", "-s", "-X", "DELETE", f"{API}/v1/gallery/{i}"],
                   capture_output=True)
print(f"удалено записей: {len(to_del)}, в галерее осталось "
      f"{api('/ready').get('gallery_size')}")

print("\n=== проверка: запросы этой машины должны получить отказ ===")
results = []
for q in qry_ids:
    w, h = sizes[q]
    d = search(q, w, h)
    ok = bool(d.get("refusal"))
    top = (d.get("candidates") or [{}])[0].get("confidence")
    results.append({"image_id": q, "refusal": ok, "top1": top})
    print(f"  {q[:16]}…  {'ОТКАЗ' if ok else 'принят'}"
          f"{'' if ok else f', лучший {top:.4f}'}")

good = [r for r in results if r["refusal"]]
print(f"\nотказов: {len(good)} из {len(results)}")
if good:
    pick = good[0]
    print(f"для демонстрации берём {pick['image_id']}")
    (HERE / "results" / "demo_refusal.json").write_text(json.dumps(
        {"image_id": pick["image_id"], "vehicle_id": victim,
         "removed_from_gallery": gal_ids,
         "note": "все кадры этой машины удалены из демо-галереи, пары нет по построению"},
        ensure_ascii=False, indent=2))
else:
    print("отказа не случилось — смотреть надо на порог, а не на пример")
