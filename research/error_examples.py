#!/usr/bin/env python3
"""Картинки ошибок и удачных совпадений для презентации.

Берёт research/honest_eval_results/error_breakdown_queries.csv (его пишет
research/error_breakdown.py) и рисует в honest_eval_results/error_examples/:

  * по одной ошибке top-1 на причину: запрос, верный ответ (место в выдаче,
    сходство), ошибочный top-1 (сходство и решение порога 0.8025);
  * удачные совпадения через смену ракурса: у запроса в галерее нет позитива
    того же ракурса, а верная машина всё равно первая.

Примеры не подбираются на глаз: в каждой причине берётся ошибка с самым
высоким сходством неверного top-1 (самая уверенная), у удачных — с самым
высоким сходством верного ответа; машины между примерами не повторяются.

Номера. В датасете размыт номер только целевой машины, у соседних машин в
рамке номера читаются (см. research/errors/error_сильная_смена_ракурса.jpg).
По ТЗ номер не используем, поэтому зона каждого номера на каждом кропе
отмечена вручную в error_examples/plate_masks.json (доли ширины и высоты
кропа) и закрывается заливкой. Кроп без разметки скрипт не рисует, а кладёт
лист с сеткой для разметки (--annotate).

    python research/error_examples.py --data-dir /opt/case17/data
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RES = HERE / "honest_eval_results"
OUT = RES / "error_examples"
THRESHOLD = 0.8025
MAX_BYTES = 1_000_000

# Палитра research/make_pitch_assets.py, чтобы картинки легли в ту же колоду.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
RIGHT, WRONG, MASK = "#2a78d6", "#eb6834", "#6f6e69"
FONT, FONT_SMALL = "PT Sans", "PT Sans Caption"
VIEW_RU = {"front": "спереди", "rear": "сзади", "side": "сбоку", "mixed": "разный", "unknown": "не размечен"}

FAILS = (
    ("twin", "Двойник: тот же цвет и та же модель",
     lambda r: r["same_color"] == "True" and r["similar_model"] == "True" and r["pos_same_view"] == "True"),
    ("front_rear", "Смена ракурса: спереди и сзади", lambda r: r["view_change"] == "перед↔зад"),
    ("side", "Смена ракурса: бок", lambda r: r["view_change"] == "перед/зад↔бок"),
    ("night", "Ночь и сумерки", lambda r: "ночь" in r["categories"].split(";")),
    ("glare", "Блики: засветка фар и солнца", lambda r: "блики" in r["categories"].split(";")),
    ("small", "Мелкая рамка", lambda r: "мелкий bbox" in r["categories"].split(";")),
    ("neighbor", "Соседняя машина в рамке", lambda r: r["neighbor"] == "True"),
    ("model_other_color", "Похожая модель, другой цвет",
     lambda r: r["same_color"] == "False" and r["similar_model"] == "True"),
)
# Подгруппа error_breakdown, по которой видно, перепредставлена ли причина среди ошибок.
QUERY_SHARE = {"front_rear": "— только другого: перед↔зад", "side": "— только другого: перед/зад↔бок",
               "night": "ночь или сумерки (глазами)", "glare": "блики (глазами)",
               "small": "мелкий bbox (< 120 000 пикс.)"}
WINS = (
    ("win_front_rear", "Нашёл машину через смену ракурса", lambda r: r["view_change"] == "перед↔зад"),
    ("win_side", "Нашёл машину сбоку по виду спереди или сзади", lambda r: r["view_change"] == "перед/зад↔бок"),
    ("win_night", "Нашёл машину ночью через смену ракурса",
     lambda r: r["view_change"] != "" and r["q_light"] in ("night", "dusk")),
)


def pick(rows: list, specs, key, used: set) -> list:
    out = []
    for name, title, pred in specs:
        cand = sorted((r for r in rows if pred(r) and r["vehicle_id"] not in used), key=key, reverse=True)
        if not cand:
            print(f"  {name}: подходящих нет, пропускаю", flush=True)
            continue
        used.add(cand[0]["vehicle_id"])
        out.append((name, title, cand[0]))
    return out


def crop_path(data_dir: Path, part: dict, image_id: str) -> Path:
    return data_dir / "crops" / part[image_id] / f"{image_id}.jpg"


def masked(path: Path, boxes: list) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in boxes:
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], fill=MASK)
    return im


def annotate_sheets(data_dir: Path, part: dict, ids: list, out_dir: Path) -> None:
    """Кропы с сеткой 10 % — по ним зоны номеров отмечаются в plate_masks.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 14)
    for i, iid in enumerate(ids):
        im = Image.open(crop_path(data_dir, part, iid)).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        for k in range(1, 10):
            d.line([(w * k / 10, 0), (w * k / 10, h)], fill=(255, 255, 0), width=1)
            d.line([(0, h * k / 10), (w, h * k / 10)], fill=(255, 255, 0), width=1)
            d.text((w * k / 10 + 2, 2), str(k), fill=(255, 255, 0), font=font)
            d.text((2, h * k / 10 + 2), str(k), fill=(255, 255, 0), font=font)
        im.save(out_dir / f"{i:02d}_{iid}.jpg", quality=90)
    print(f"  листов для разметки номеров: {len(ids)} -> {out_dir}")


def fit_png(path: Path) -> int:
    """PNG не больше 1 МБ: при перевесе — палитра 256 цветов с диффузией."""
    if path.stat().st_size > MAX_BYTES:
        im = Image.open(path).convert("RGB")
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(
            path, optimize=True)
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise SystemExit(f"{path.name}: {size} байт > 1 МБ")
    return size


def draw(panes: list, title: str, subtitle: str, footer: str, path: Path) -> int:
    """panes: (картинка, цвет рамки, подпись жирная, подпись обычная)."""
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    fig.text(0.035, 0.935, title, fontsize=32, color=INK, family=FONT, weight="bold", va="top")
    fig.text(0.035, 0.85, subtitle, fontsize=19, color=INK2, family=FONT, va="top")
    n = len(panes)
    gap, left, width = 0.025, 0.035, 0.93
    pw = (width - gap * (n - 1)) / n
    for k, (im, color, head, body) in enumerate(panes):
        ax = fig.add_axes([left + k * (pw + gap), 0.235, pw, 0.53])
        ax.imshow(im)
        ax.set_anchor("S")
        ax.axis("off")
        w, h = im.size
        ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, lw=7, edgecolor=color))
        x = left + k * (pw + gap)
        fig.text(x, 0.195, head, fontsize=21, color=color if color != MUTED else INK, family=FONT,
                 weight="bold", va="top")
        fig.text(x, 0.14, body, fontsize=17, color=INK2, family=FONT_SMALL, va="top", linespacing=1.35)
    fig.text(0.035, 0.03, footer, fontsize=13, color=MUTED, family=FONT_SMALL, va="bottom")
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    return fit_png(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=RES / "error_breakdown_queries.csv")
    ap.add_argument("--summary", type=Path, default=RES / "error_breakdown.json")
    ap.add_argument("--masks", type=Path, default=OUT / "plate_masks.json")
    ap.add_argument("--annotate", type=Path, default=None, help="куда класть листы разметки номеров")
    a = ap.parse_args()

    part = {}
    for p, name in (("train", "train.csv"), ("query", "test_query.csv"), ("gallery", "test_gallery.csv")):
        for r in csv.DictReader((a.data_dir / name).open()):
            part[r["image_id"]] = p
    rows = [r for r in csv.DictReader(a.queries.open(encoding="utf-8")) if r["openset"] == "False"]
    s = json.loads(a.summary.read_text(encoding="utf-8"))
    n_f = s["errors"]
    views = {}
    for r in rows:
        views[r["query"]] = r["q_view"]
        views[r["correct"]] = r["correct_view"]

    used = set()
    fails = pick([r for r in rows if r["hit1"] == "False"], FAILS, lambda r: float(r["top1_sim"]), used)
    wins = pick([r for r in rows if r["hit1"] == "True" and r["pos_same_view"] == "False"], WINS,
                lambda r: float(r["correct_sim"]), used)
    need = []
    for _, _, r in fails:
        need += [r["query"], r["correct"], r["top1"]]
    for _, _, r in wins:
        need += [r["query"], r["correct"]]
    masks = json.loads(a.masks.read_text(encoding="utf-8")) if a.masks.exists() else {}
    missing = [i for i in dict.fromkeys(need) if i not in masks]
    if missing:
        if a.annotate:
            annotate_sheets(a.data_dir, part, missing, a.annotate)
        raise SystemExit(f"нет разметки номеров для {len(missing)} кропов: {', '.join(missing[:3])}… "
                         "(--annotate DIR, затем дописать plate_masks.json)")

    share = {"twin": s["pair_labels"]["same_color_and_similar_model"],
             "front_rear": s["view_change_kinds"].get("перед↔зад", {}).get("n", 0),
             "side": s["view_change_kinds"].get("перед/зад↔бок", {}).get("n", 0),
             "night": s["errors_by_category"]["ночь"]["n"], "glare": s["errors_by_category"]["блики"]["n"],
             "small": s["errors_by_category"]["мелкий bbox"]["n"],
             "neighbor": s["errors_by_category"]["соседняя машина в рамке"]["n"],
             "model_other_color": s["pair_labels"]["similar_model"] - s["pair_labels"]["same_color_and_similar_model"]}
    footer = ("Студент из main, галерея 2.2k, протокол организаторов, сырой косинус. "
              "Номера закрыты: по ТЗ номер не используется.")
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for i, (name, title, r) in enumerate(fails, 1):
        t1 = float(r["top1_sim"])
        verdict = "выше порога 0.8025: сервис выдал бы её" if t1 >= THRESHOLD else "ниже порога 0.8025: сервис отказал бы"
        panes = [
            (masked(crop_path(a.data_dir, part, r["query"]), masks[r["query"]]), MUTED, "Запрос",
             f"вид {VIEW_RU[r['q_view']]}"),
            (masked(crop_path(a.data_dir, part, r["correct"]), masks[r["correct"]]), RIGHT, "Верный ответ",
             f"место {r['correct_rank']} в выдаче, сходство {float(r['correct_sim']):.3f}\n"
             f"вид {VIEW_RU[views.get(r['correct'], 'unknown')]}"),
            (masked(crop_path(a.data_dir, part, r["top1"]), masks[r["top1"]]), WRONG, "Выдана первой: чужая машина",
             f"сходство {t1:.3f}\n{verdict}"),
        ]
        sub = f"{share[name]} из {n_f} ошибок top-1 на 2.2k ({share[name] / n_f:.0%})"
        if name in QUERY_SHARE:
            g = next(x for x in s["subgroups"] if x["group"] == QUERY_SHARE[name])
            sub += f", а среди всех запросов таких {g['share_queries']:.0%}"
        path = OUT / f"fail_{i}_{name}.png"
        made.append((path.name, draw(panes, title, sub, footer, path), r["query"]))
    for i, (name, title, r) in enumerate(wins, 1):
        qv, cv = VIEW_RU[r["q_view"]], VIEW_RU[views.get(r["correct"], "unknown")]
        panes = [
            (masked(crop_path(a.data_dir, part, r["query"]), masks[r["query"]]), MUTED, "Запрос", f"вид {qv}"),
            (masked(crop_path(a.data_dir, part, r["correct"]), masks[r["correct"]]), RIGHT, "Выдана первой: верно",
             f"сходство {float(r['correct_sim']):.3f}, вид {cv}"),
        ]
        sub = f"В галерее эта машина есть только в другом ракурсе: запрос {qv}, ответ {cv}"
        path = OUT / f"win_{i}_{name.removeprefix('win_')}.png"
        made.append((path.name, draw(panes, title, sub, footer, path), r["query"]))
    for name, size, q in made:
        print(f"  {name}: {size / 1e6:.2f} МБ (запрос {q})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
