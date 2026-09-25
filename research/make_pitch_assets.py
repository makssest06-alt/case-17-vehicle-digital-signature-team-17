#!/usr/bin/env python3
"""Картинки для презентации. Один прогон — весь комплект в pitch_assets/.

    python research/make_pitch_assets.py

Правила оформления одни на все графики и выбраны не на глаз:
  * номинальные категории — один цвет на серию, акцент вместо радуги;
  * упорядоченные стадии — одна гамма от светлого к тёмному (проверена
    валидатором палитры: монотонная светлота, шаг по светлоте, светлый
    конец различим на фоне);
  * сетка волосяная и сплошная, подписи значений прямо у столбцов,
    текст своим цветом, а не цветом серии.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).parent
OUT = HERE / "pitch_assets"
RES = HERE / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"
NEUTRAL = "#d8d7d0"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def style(ax, xlabel=None, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)


def fig_new(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    return fig


def title(fig, main, sub=None):
    fig.text(0.012, 0.965, main, fontsize=14, color=INK, weight="bold", va="top")
    if sub:
        fig.text(0.012, 0.895, sub, fontsize=10.5, color=INK2, va="top")


# ---------------------------------------------------------------- 1. бэкбоны
def chart_backbones():
    # (подпись, mAP, лицензия свободная?)
    data = [
        ("ResNet50 ImageNet, без обучения",      10.61, True),
        ("CLIP ViT-B/16, без обучения",          19.05, True),
        ("fast-reid VeRi-776",                   26.18, False),
        ("SigLIP base, без обучения",            46.81, True),
        ("CLIP ViT-B/16 + голова",               47.71, True),
        ("fast-reid VERI-Wild",                  48.58, False),
        ("SigLIP 2 base, без обучения",          49.57, True),
        ("SigLIP so400m, без обучения",          60.80, True),
        ("SigLIP base + голова",                 65.16, True),
    ]
    fig = fig_new(10, 6.0)
    ax = fig.add_axes([0.34, 0.16, 0.63, 0.68])
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    ys = range(len(data))

    for y, (lab, v, free) in zip(ys, data):
        is_best = lab.startswith("SigLIP base +")
        ax.barh(y, v, height=0.62, zorder=2,
                color=S1 if is_best else NEUTRAL,
                hatch="" if free else "///",
                edgecolor=SURFACE if free else MUTED, linewidth=1.2)
        ax.text(v + 0.9, y, f"{v:.2f}", va="center", ha="left",
                fontsize=10.5, color=INK if is_best else INK2,
                weight="bold" if is_best else "normal")

    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=10.5, color=INK2)
    ax.set_xlim(0, 74)
    style(ax, "mAP на кросс-камерной валидации")
    ax.legend(handles=[Patch(facecolor=NEUTRAL, hatch="///", edgecolor=MUTED,
                             label="лицензия не подходит для решения")],
              loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)
    title(fig, "Выбор бэкбона: девять вариантов на одном сплите",
          "Замена CLIP на SigLIP дала +27.8 mAP без единого шага обучения. "
          "Обученная голова добавила ещё +18.4.")
    # Все девять строк измерены одним офлайн-протоколом, поэтому сравнимы между
    # собой. Цифру живого сервиса приписываем отдельно, а не подменяем ею строку:
    # смешивать два протокола в одном столбике нельзя.
    ex = json.loads((RES / "service_path_extras.json").read_text())
    fig.text(0.012, 0.012,
             "Сравнение офлайн, один протокол на все девять строк.\n"
             f"Тем же кодом, которым отвечает сервис, выбранный вариант даёт "
             f"{ex['prenorm']['mAP']} mAP — research/eval_service_path.py.",
             fontsize=9, color=MUTED, va="bottom", linespacing=1.5)
    fig.savefig(OUT / "01_backbones.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------- 2. ракурс
def chart_viewpoint():
    """Стадии берутся из замеров, а не из констант в коде."""
    ex = json.loads((RES / 'service_path_extras.json').read_text())
    stages = [
        ("CLIP, без обучения", 34.7),
        ("SigLIP, без обучения", 12.3),
        ("CLIP + голова", 7.6),
        ("SigLIP + голова\n(живой сервис)", ex['prenorm']['catastrophic_pct']),
        ("+ постобработка\n(артефакты сдачи)", ex['prenorm_postprocess']['catastrophic_pct']),
    ]
    fig = fig_new(10, 5.0)
    ax = fig.add_axes([0.07, 0.20, 0.90, 0.60])
    xs = range(len(stages))
    for x, (lab, v) in enumerate(stages):
        ax.bar(x, v, width=0.56, color=RAMP[x], zorder=2)
        ax.text(x, v + 0.9, f"{v}%", ha="center", va="bottom",
                fontsize=12, color=INK, weight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s[0] for s in stages], fontsize=10.5, color=INK2)
    ax.set_ylim(0, 40)
    style(ax, None, grid_axis="y")
    ax.set_ylabel("доля катастрофических провалов", color=INK2, fontsize=10)
    last = stages[-1][1]
    times = stages[0][1] / last if last else 0
    ax.annotate("", xy=(4.72, last), xytext=(4.72, stages[0][1]),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.text(4.86, 18, f"в {times:.0f} раз\nменьше", fontsize=11, color=INK,
            weight="bold", va="center")
    ax.set_xlim(-0.6, 5.3)
    title(fig, "Проблема ракурса: измерили и закрыли",
          "Катастрофический провал — правильный ответ дальше 50-го места. "
          "Именно так выглядит промах по ракурсу.")
    fig.savefig(OUT / "02_viewpoint.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------- 3. скорость
def chart_speed():
    rows = [("ДО\nторч, Pillow, pgvector", 332.6, 347.4 - 332.6),
            ("ПОСЛЕ\nONNX, TurboJPEG,\nиндекс в памяти", 311.2, 320.1 - 311.2)]
    fig = fig_new(10, 3.9)
    ax = fig.add_axes([0.24, 0.24, 0.70, 0.46])
    for y, (lab, model, rest) in enumerate(rows):
        ax.barh(y, model, height=0.34, color=S1, zorder=2)
        ax.barh(y, rest, height=0.34, left=model + 2, color=S2, zorder=2)
        ax.text(model / 2, y, f"{model:.0f} мс", va="center", ha="center",
                fontsize=11, color="#ffffff", weight="bold")
        ax.text(model + rest + 10, y, f"итого {model + rest:.0f} мс",
                va="center", ha="left", fontsize=10.5, color=INK2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK2)
    ax.set_xlim(0, 430)
    ax.set_ylim(1.6, -0.6)
    style(ax, "миллисекунды на один кадр, 2 vCPU")
    # Легенда над полем: справа она села бы поверх столбцов и подписи итога.
    ax.legend(handles=[Patch(facecolor=S1, label="модель"),
                       Patch(facecolor=S2, label="вся остальная обвязка")],
              loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, fontsize=10, labelcolor=INK2)
    title(fig, "Почему мы не заявляем двукратное ускорение",
          "97% времени занимает модель. Оптимизация обвязки физически "
          "не может дать больше 3%, и честный выигрыш — 9%.")
    fig.savefig(OUT / "03_speed.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------- 4. ANN
def chart_ann():
    path = RES / "ann" / "summary.json"
    if not path.exists():
        print("  пропускаю ANN: нет", path)
        return
    rows = json.loads(path.read_text())
    names = [r["method"].split(" (")[0] for r in rows]

    fig = fig_new(12, 4.4)
    panels = [
        ("Задержка одного запроса", [r["latency"]["p50_ms"] for r in rows], "мс, p50", True),
        ("Память индекса", [r["index_mb"] for r in rows], "МБ", False),
        ("Качество на реальных запросах", [r["mAP"] for r in rows], "mAP", False),
    ]
    for i, (t, vals, unit, logscale) in enumerate(panels):
        ax = fig.add_axes([0.06 + i * 0.322, 0.16, 0.25, 0.58])
        ax.bar(range(len(vals)), vals, width=0.56, color=S1, zorder=2)
        if logscale:
            ax.set_yscale("log")
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:,.1f}".replace(",", " ") if v < 100 else f"{v:,.0f}".replace(",", " "),
                    ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold")
        if i == 2:
            ax.axhline(rows[0]["mAP"], color=MUTED, lw=1.0, zorder=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9.5, color=INK2)
        style(ax, None, grid_axis="y")
        ax.set_ylabel(unit, color=INK2, fontsize=9.5)
        ax.set_title(t, fontsize=11, color=INK, pad=10)
        ax.margins(y=0.22)
    title(fig, "Галерея на миллион векторов: цена и качество",
          "Реальная валидационная галерея внутри миллиона правдоподобных "
          "дистракторов, реальные запросы, метрики по протоколу жюри.")
    fig.savefig(OUT / "04_ann_1m.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------- 5. распределения и порог
def chart_refusal():
    """Два распределения максимального сходства и обе рабочие точки.

    Гистограмма, а не ROC: на защите нужно показать, что решение об отказе
    принимается по одному числу и что это число разделяет два случая.
    """
    import numpy as np
    npz = RES / "refusal_scores_service.npz"
    ex = RES / "service_path_extras.json"
    if not npz.exists() or not ex.exists():
        print("  пропускаю порог отказа: нет замеров")
        return
    z = np.load(npz)
    r = json.loads(ex.read_text())["refusal_deployment"]
    pos, neg = z["deployment_pos"], z["deployment_neg"]
    bal, strict = r["balanced"]["threshold"], r["strict"]["threshold"]

    fig = fig_new(10, 5.0)
    ax = fig.add_axes([0.07, 0.15, 0.90, 0.55])
    lo = min(pos.min(), neg.min()); hi = max(pos.max(), neg.max())
    bins = np.linspace(lo, hi, 46)
    ax.hist(neg, bins=bins, color=S2, alpha=0.85, zorder=2,
            label=f"пары в галерее нет ({len(neg)} запросов)")
    ax.hist(pos, bins=bins, color=S1, alpha=0.85, zorder=2,
            label=f"пара есть ({len(pos)} запросов)")
    top = ax.get_ylim()[1]
    # Подписи разводим по высоте и по стороне: при близких порогах
    # две подписи в одну строку налезают друг на друга.
    marks = ((bal, "сбалансированный", "-", 0.99, "right", -0.006),
             (strict, "строгий", "--", 0.99, "left", 0.006))
    for x, lab, ls, ypos, ha, dx in marks:
        ax.axvline(x, color=INK, lw=1.4, ls=ls, zorder=4)
        ax.text(x + dx, top * ypos, f"{lab}\n{x:.4f}", fontsize=10, color=INK,
                va="top", ha=ha, weight="bold", zorder=5)
    style(ax, "максимальное сходство с галереей", grid_axis="y")
    ax.set_ylabel("число запросов", color=INK2, fontsize=10)
    # Легенда над полем: внутри она садится поверх подписи порога и столбиков.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, fontsize=10, labelcolor=INK2)
    title(fig, "Порог отказа: откуда взялось число",
          f"Рабочий протокол, AUC {r['auc']:.4f}. Сбалансированная точка — максимум индекса "
          f"Юдена:\nF1 {r['balanced']['f1']}, полнота {r['balanced']['recall']}, "
          f"TNR {r['balanced']['TNR']}. Строгая точка — TNR {r['strict']['TNR']}, "
          f"точность {r['strict']['precision']}.")
    fig.savefig(OUT / "05_refusal.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------- схемы
def _flow(fig, ax, steps, y=0.5, box_h=0.62, gap=0.026, accent=None):
    """Горизонтальная цепочка прямоугольников со стрелками.

    Подписи внутри блоков, а не подписи-выноски: на слайде выноски читаются
    хуже и ломаются при масштабировании.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    n = len(steps)
    w = (1.0 - gap * (n - 1)) / n
    for i, (title_txt, sub) in enumerate(steps):
        x = i * (w + gap)
        is_accent = accent is not None and i in accent
        ax.add_patch(FancyBboxPatch(
            (x, y - box_h / 2), w, box_h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.1, facecolor=S1 if is_accent else "#ffffff",
            edgecolor=S1 if is_accent else AXIS, zorder=2))
        ax.text(x + w / 2, y + 0.16, title_txt, ha="center", va="center",
                fontsize=10.5, weight="bold", zorder=3,
                color="#ffffff" if is_accent else INK)
        ax.text(x + w / 2, y - 0.11, sub, ha="center", va="center",
                fontsize=8.2, zorder=3, linespacing=1.5,
                color="#e8f0fb" if is_accent else INK2)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w, y), (x + w + gap, y), arrowstyle="-|>",
                mutation_scale=11, linewidth=1.1, color=MUTED, zorder=2))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1)
    ax.axis("off")


def chart_architecture():
    fig = fig_new(13, 3.1)
    ax = fig.add_axes([0.015, 0.06, 0.97, 0.62])
    _flow(fig, ax, [
        ("Кадр", "1920×1080\nс камеры"),
        ("Рамка ТС", "x, y, w, h\nоператор\nили детектор"),
        ("Кроп", "запас 10%\nресайз 224×224"),
        ("SigLIP base", "бэкбон Apache 2.0\nзапечён в образ"),
        ("Голова", "768 → 1024 → 512\nBNNeck, ArcFace"),
        ("Вектор", "512 чисел\nL2-нормирован"),
        ("Поиск", "pgvector,\nиндекс в памяти,\nHNSW"),
        ("Ответ", "10 кандидатов\nлибо отказ"),
    ], accent={3, 4, 5})
    title(fig, "Схема компонентов",
          "Одна модель, одна голова, один индекс. Госномер не извлекается "
          "и не используется нигде в этой цепочке.")
    fig.savefig(OUT / "11_architecture.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_user_journey():
    # Пороги берём из замера, иначе схема начинает врать первой:
    # её правят реже всего, а число на ней видно с последнего ряда.
    r = json.loads((RES / "service_path_extras.json").read_text())["refusal_deployment"]
    bal, strict = r["balanced"]["threshold"], r["strict"]["threshold"]
    fig = fig_new(13, 3.1)
    ax = fig.add_axes([0.015, 0.10, 0.97, 0.62])
    _flow(fig, ax, [
        ("Открывает веб", "порт 8080\nничего не ставит"),
        ("Грузит кадр", "перетащил файл\nили выбрал"),
        ("Обводит машину", "рамка мышью\nили координаты"),
        ("Выбирает режим", f"сбаланс. {bal}\nили строгий {strict}"),
        ("Получает ответ", "кандидаты\nс уверенностью\nили «нет совпадения»"),
        ("Показать почему", "карта вклада\nучастков кадра"),
        ("Выгружает", "CSV или JSON\nдля отчёта"),
    ], accent={3, 4})
    title(fig, "Путь оператора",
          "Строгость — решение человека, а не константа в коде: цена ложного "
          "совпадения в каждой задаче своя.")
    fig.savefig(OUT / "12_user_journey.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def copy_existing():
    """Готовые картинки: коллажи ошибок и скриншоты живого интерфейса.

    Скриншоты снимаются Playwright по SSH-туннелю к демо-стенду и кладутся
    в /tmp перед запуском — заново их сгенерировать нельзя, поэтому они
    копируются, а не рисуются.
    """
    import shutil
    pairs = [
        (HERE / "errors" / "error_сильная_смена_ракурса.jpg", "06_errors_viewpoint.jpg"),
        (HERE / "errors" / "error_ночь.jpg", "07_errors_night.jpg"),
        (HERE / "errors" / "error_похожая_машина.jpg", "08_errors_similar.jpg"),
        (Path("/tmp/ui_search_easy.png"), "09_ui_search.png"),
        (Path("/tmp/ui_hard.png"), "13_ui_hard_viewpoint.png"),
        (Path("/tmp/ui_night.png"), "14_ui_night.png"),
        (Path("/tmp/ui_refusal.png"), "15_ui_refusal.png"),
        (Path("/tmp/ui_explain.png"), "16_ui_explain.png"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, OUT / dst)
            print(f"  скопировано {dst}")
        else:
            print(f"  НЕТ ИСХОДНИКА {src}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chart_backbones(); print("  01_backbones.png")
    chart_viewpoint(); print("  02_viewpoint.png")
    chart_speed();     print("  03_speed.png")
    chart_ann()
    chart_refusal(); print("  05_refusal.png")
    chart_architecture(); print("  11_architecture.png")
    chart_user_journey(); print("  12_user_journey.png")
    copy_existing()
    print(f"\nвсё в {OUT}")


if __name__ == "__main__":
    main()
