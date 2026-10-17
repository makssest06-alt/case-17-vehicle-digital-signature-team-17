#!/usr/bin/env python3
"""Пакет для презентации: slides.md и две картинки 16:9, числа — только из прогонов.

Читает research/honest_eval_results/*.json и пишет туда же:
  * gallery_size_map.png — mAP@10 студента из main от размера галереи
    (honest_eval_ladder_*.json и honest_eval_student.json, 5 фолдов);
  * score_scale.png — шкала сырого косинуса против CSLS-скора и порог 0.8025
    (score_contract.json);
  * slides.md — 4 слайда, в каждом тезисе число подставлено из файла.

    python research/honest_eval_slides.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "honest_eval_results"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CURVE, NOW, BAND = "#2a78d6", "#eb6834", "#cfe0f6"
FONT = "PT Sans"
LADDER = (("honest_eval_ladder_none.json", "только машины\nholdout"),
          ("honest_eval_ladder_test_gallery.json", "+ test gallery"),
          ("honest_eval_student.json", "+ весь тест"),
          ("honest_eval_ladder_train_seen.json", "+ кадры\nобучающих машин"),
          ("honest_eval_ladder_test_all_train_seen.json", "+ тест и\nобучающие"))


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def ladder() -> list:
    out = []
    for name, label in LADDER:
        s = load(name)["summary"]["big/raw/official"]
        out.append({"label": label, "n": int(round(s["n_gallery"]["mean"])), "map": s["mAP@10"]["mean"],
                    "std": s["mAP@10"]["std"], "folds": s["mAP@10"]["n"], "r1": s["Rank-1"]["mean"]})
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=16, length=0)


def header(fig, title, sub):
    fig.text(0.04, 0.94, title, fontsize=32, color=INK, family=FONT, weight="bold", va="top")
    fig.text(0.04, 0.865, sub, fontsize=17, color=INK2, family=FONT, va="top", linespacing=1.4)


def gallery_png(pts: list, path: Path) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    header(fig, "Чем больше чужих машин в галерее, тем ниже mAP@10",
           "Студент из main, сырой косинус, протокол организаторов, 150 машин holdout, 5 фолдов по 30 машин.\n"
           "Полоса и усы: выборочный разброс по фолдам.")
    ax = fig.add_axes([0.08, 0.2, 0.86, 0.55])
    style(ax)
    x = [p["n"] for p in pts]
    y = [p["map"] for p in pts]
    lo = [p["map"] - p["std"] for p in pts]
    hi = [p["map"] + p["std"] for p in pts]
    ax.fill_between(x, lo, hi, color=BAND, lw=0, zorder=1)
    ax.plot(x, y, color=CURVE, lw=4, zorder=3)
    ax.errorbar(x, y, yerr=[p["std"] for p in pts], fmt="none", ecolor=CURVE, elinewidth=2, capsize=6, zorder=3)
    ax.scatter(x, y, s=180, color=CURVE, edgecolor=SURFACE, linewidth=2.5, zorder=4)
    # две последние точки близко на логарифмической шкале: подписи разводим влево и вправо
    place = [((0, 18), "center")] * (len(pts) - 2) + [((-14, 14), "right"), ((14, -34), "left")]
    for p, (off, ha) in zip(pts, place):
        ax.annotate(f"{p['map']:.3f}", (p["n"], p["map"]), xytext=off, textcoords="offset points",
                    ha=ha, fontsize=22, color=INK, family=FONT, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9), zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(230, 16000)
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p['n']:,}".replace(",", " ") + f"\n{p['label']}" for p in pts], fontsize=15,
                       family=FONT, color=INK2, linespacing=1.25)
    ax.minorticks_off()
    ticks = ax.get_xticklabels()
    ticks[-2].set_ha("right")
    ticks[-1].set_ha("left")
    for lab in ax.get_yticklabels():
        lab.set_family(FONT)
    ax.set_ylabel("mAP@10", fontsize=19, color=INK2, family=FONT, labelpad=10)
    fig.text(0.94, 0.035, "Кадров в галерее (логарифмическая шкала)", fontsize=17, color=INK2, family=FONT,
             ha="right")
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def score_png(sc: dict, path: Path) -> None:
    pc = sc["modes"]["official"]["score_percentiles"]
    art = sc["submitted_test_artifacts"]
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    header(fig, "Порог 0.8025 и CSLS-скор живут на разных шкалах",
           "Top-1 скор запросов студента на 2.2k, режим эталона: полоса от 1-го до 99-го процентиля, точки: 10, 50 и 90.\n"
           "Порог сервиса задан для сырого косинуса; на CSLS-скоре candidates.csv он отказывает всем.")
    ax = fig.add_axes([0.2, 0.22, 0.74, 0.5])
    style(ax)
    ax.grid(axis="y", visible=False)
    rows = (("сырой косинус\n(/v1/search)", pc["raw"], CURVE, 1.0), ("CSLS-скор\n(candidates.csv)", pc["csls"], MUTED, 0.0))
    for label, p, color, y in rows:
        ax.plot([p["1"], p["99"]], [y, y], color=color, lw=10, solid_capstyle="round", alpha=0.35)
        ax.plot([p["10"], p["90"]], [y, y], color=color, lw=10, solid_capstyle="round")
        for q in ("10", "50", "90"):
            ax.scatter([p[q]], [y], s=160, color=SURFACE, edgecolor=color, linewidth=3, zorder=4)
            ax.annotate(f"{p[q]:.2f}", (p[q], y), xytext=(0, 18), textcoords="offset points", ha="center",
                        fontsize=16, color=INK2, family=FONT)
    ax.axvline(0.8025, color=NOW, lw=3)
    ax.text(0.8085, 0.5, "порог 0.8025", fontsize=19, color=NOW, family=FONT, weight="bold", va="center")
    ax.axvline(art["csls_top1_max"], color=INK, lw=1.5, ls=(0, (4, 4)))
    ax.text(art["csls_top1_max"] - 0.008, -0.32, f"максимум CSLS на сданном тесте {art['csls_top1_max']:.3f}",
            fontsize=15, color=INK, family=FONT, va="center", ha="right")
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=18, family=FONT, color=INK, linespacing=1.3)
    ax.set_ylim(-0.55, 1.45)
    ax.set_xlim(-0.05, 1.02)
    for lab in ax.get_xticklabels():
        lab.set_family(FONT)
    ax.set_xlabel("Скор top-1", fontsize=19, color=INK2, family=FONT, labelpad=10)
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def slides(pts: list) -> str:
    st = load("honest_eval_student.json")
    suite = load("honest_eval_student_rk_suite.json")["suite"]
    sc = load("score_contract.json")
    leak = load("leak_audit.json")
    err = load("error_breakdown.json")
    rm = load("refusal_mode_a10.json")                                 # режим отказа, a10, как сервис
    sub = {g["group"]: g for g in err["subgroups"]}
    means = lambda c: {k: v["mean"] for k, v in c.items()}           # среднее по фолдам
    svc_off = means(sc["modes"]["official"]["contracts"]["service (raw, 0.8025)"])
    svc_nod = means(sc["modes"]["nodup"]["contracts"]["service (raw, 0.8025)"])
    mixed = means(sc["modes"]["nodup"]["contracts"]["mixed (CSLS, 0.8025)"])
    art = means(sc["modes"]["nodup"]["contracts"]["artifacts (CSLS, 10-й процентиль)"])
    fin = rm.get("final", "dba1+camera")
    now, fin_z, fin_r = rm["none"]["current"], rm[fin]["z100"], rm[fin]["raw"]
    rm_grid = rm[fin]["grid"]
    tz = [v["TNR"]["mean"] for k, v in fin_z["sizes"].items() if k.isdigit()]
    old = st["old_protocol"]
    ec = err["errors_by_category"]
    n_f = err["errors"]
    big = suite["2.2k (+ весь тест)"]["metrics"]
    first, last = pts[0], pts[-1]
    nd = leak["near_duplicates_same_vehicle"]
    f2 = lambda x: f"{x:.2f}"
    f3 = lambda x: f"{x:.3f}"
    pct = lambda a, b: f"{a / b:.0%}".replace("%", " %")
    num = lambda n: f"{n:,}".replace(",", " ")

    return "\n".join([
        "# Слайды: честная валидация", "",
        "Числа подставлены из файлов `research/honest_eval_results/` скриптом "
        "`research/honest_eval_slides.py`; слайды 1–3 — студент из `main` (его мерили эксперты), слайд 4 — "
        "прод-модель a10; сырой косинус, если не сказано иначе.", "",
        "## 1. Честная валидация", "",
        f"- Метрики считает эталон организаторов без правок: junk — та же машина с той же камеры, "
        f"AP@10 нормирован на min(позитивов, 10).",
        f"- {st['holdout_vehicles']} машин holdout, {st['folds']} фолдов по {st['holdout_vehicles'] // st['folds']} машин; "
        f"у {st['openset_frac']:.0%} машин фолда "
        f"пары в галерее нет, по ним меряется отказ. Модель строится кодом сервиса, скор — как у `/v1/search`.",
        f"- mAP@10 падает с размером галереи: {f3(first['map'])} на {num(first['n'])} кадрах → "
        + " → ".join(f"{f3(p['map'])}" for p in pts[1:]) + f" на {num(last['n'])}.",
        f"- Утечек нет: пересечение машин holdout и обучения "
        f"{leak['split_audit']['train']['vehicle_id_overlap_with_holdout']}, ArcFace-классов студента "
        f"{leak['student_training_identities']['arcface_classes']} = машины обучения; "
        f"{leak['exact_full_frame_duplicates']['groups_by_kind']['holdout↔train1391']} дублей полных кадров "
        f"holdout ↔ обучение исключены.", "",
        "Картинка: `honest_eval_results/gallery_size_map.png`", "",
        "## 2. Почему падала метрика", "",
        f"- Размер галереи: старый протокол — одна галерея на 361 кадр, raw {f3(old['raw']['mAP@10'])}, "
        f"с постобработкой {f3(old['full']['mAP@10'])}; честный протокол на 2.2k — raw "
        f"{f3(big['raw']['official']['mAP@10']['mean'])}, на {num(last['n'])} кадрах — {f3(last['map'])}.",
        f"- Шкала скора: порог 0.8025 задан для сырого косинуса. На CSLS-скоре `candidates.csv` максимум на сданном "
        f"тесте {f3(sc['submitted_test_artifacts']['csls_top1_max'])}, и порог отказывает всем: recall "
        f"{f2(mixed['recall'])}, TNR {f2(mixed['TNR'])}. Правило артефактов (порог — 10-й процентиль скора) без "
        f"почти-дублей даёт TNR {f2(art['TNR'])}.",
        f"- Порог держится на соседних кадрах одного проезда: в режиме эталона recall {f2(svc_off['recall'])}, "
        f"TNR {f2(svc_off['TNR'])}; без них recall {f2(svc_nod['recall'])} при том же TNR. У "
        f"{nd['holdout_query_same_camera_gallery_cos>=0.95']} из {nd['holdout_query']} запросов holdout такой "
        f"соседний кадр в галерее с косинусом ≥ 0.95 (сплит make_split.py).",
        f"- Постобработка (расширение запроса + k-reciprocal) на 2.2k даёт "
        f"{f3(big['rerank']['official']['mAP@10']['mean'])} против {f3(big['raw']['official']['mAP@10']['mean'])}, "
        "но опирается на другие запросы, и `/v1/search` так не отвечает.", "",
        "Картинки: `honest_eval_results/score_scale.png`, `honest_eval_results/gallery_size_map.png`", "",
        "## 3. Разбор ошибок", "",
        f"- {n_f} ошибок top-1 на {err['scored_queries']} запросах (2.2k). "
        f"{pct(err['pair_labels']['same_color_and_similar_model'], n_f)} — двойник того же цвета и той же модели "
        f"или кузова; чёрные машины {f3(sub['цвет запроса: чёрный']['mAP@10'])} против "
        f"{f3(sub['цвет запроса: цветной']['mAP@10'])} у цветных.",
        f"- {ec['смена ракурса']['share_errors']:.0%}".replace("%", " %")
        + f" ошибок — в галерее нет кадра того же ракурса (перед ↔ зад "
        f"{err['view_change_kinds']['перед↔зад']['share_errors']:.0%}".replace("%", " %") + "); mAP@10 "
        f"{f3(sub['позитивы только другого ракурса']['mAP@10'])} против "
        f"{f3(sub['в галерее есть позитив того же ракурса']['mAP@10'])}.",
        f"- Ночь и сумерки {f3(sub['ночь или сумерки (глазами)']['mAP@10'])} против "
        f"{f3(sub['день (глазами)']['mAP@10'])} днём, блики {f3(sub['блики (глазами)']['mAP@10'])} против "
        f"{f3(sub['без бликов (глазами)']['mAP@10'])}.",
        f"- Промахи близкие: у {err['correct_rank_on_error']['in_top10']} из {n_f} ошибок верный ответ в топ-10 "
        f"(медиана места {err['correct_rank_on_error']['median']}); {err['confident_wrong_any']} неверных top-1 "
        "выше порога 0.8025.", "",
        "Картинки (номера закрыты): `honest_eval_results/error_examples/fail_1_twin.png`, "
        "`fail_2_front_rear.png`, `fail_4_night.png`, `fail_5_glare.png`", "",
        "## 4. Режим отказа (предварительно, прод-модель a10)", "",
        f"- Отказ — оценка лучшего кандидата ниже порога. В галерее сервиса у запроса без пары остаётся кадр его "
        f"машины с той же камеры, и сейчас (raw 0.8025, без постобработки) TNR {f3(now['2.2k']['TNR']['mean'])} на "
        f"2.2k и {f3(now['10.8k']['TNR']['mean'])} на 10.8k: чужим сервис почти не отказывает.",
        f"- Итог, предварительно: {rm[fin]['label']}, оценка z100 = (s1 − mean(s2..s100)) / "
        f"std(s2..s100), порог {fin_z['threshold']['mean']:.2f}. F1 / TNR {f3(fin_z['sizes']['2.2k']['F1']['mean'])} / "
        f"{f3(fin_z['sizes']['2.2k']['TNR']['mean'])} на 2.2k, {f3(fin_z['sizes']['10.8k']['F1']['mean'])} / "
        f"{f3(fin_z['sizes']['10.8k']['TNR']['mean'])} на 10.8k; TNR {f3(min(tz))}–{f3(max(tz))} на галереях от "
        f"{num(rm_grid[0])} до {num(rm_grid[-1])} кадров.",
        f"- Почему не сырой косинус: максимум сходства с чужими машинами растёт с размером галереи "
        f"({f3(fin_r['sizes'][str(rm_grid[0])]['open_s1'])} → {f3(fin_r['sizes'][str(rm_grid[-1])]['open_s1'])}), "
        f"лучший порог raw — {f3(fin_r['sizes'][str(rm_grid[0])]['own_threshold']['mean'])} → "
        f"{f3(fin_r['sizes'][str(rm_grid[-1])]['own_threshold']['mean'])}; один порог raw на 10.8k даёт TNR "
        f"{f3(fin_r['sizes']['10.8k']['TNR']['mean'])}. Порог от N держит TNR, но на 10.8k F1 "
        f"{f3(rm[fin]['thr_n']['sizes']['10.8k']['F1']['mean'])}.",
        "- Порог выбран вне фолда (150 машин holdout, 5 фолдов по 30 машин) по всем размерам галереи сразу, тест — только "
        "чужие кадры. Финал — после решения по постобработке, пересчёт одной командой `research/refusal_mode.py`, в "
        "конфиге меняется одно число.", "",
        "Подробно: `docs/refusal.md`; картинка — `honest_eval_results/threshold_vs_gallery_a10_service.png`", ""])


def main() -> int:
    pts = ladder()
    gallery_png(pts, RES / "gallery_size_map.png")
    score_png(load("score_contract.json"), RES / "score_scale.png")
    text = slides(pts)
    (RES / "slides.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
