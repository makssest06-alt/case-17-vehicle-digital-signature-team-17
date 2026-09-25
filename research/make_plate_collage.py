#!/usr/bin/env python3
"""Коллаж «почему номер не спасает»: четыре кадра, каждый — измеренная причина.

Кадры не подбираются на глаз. Для случайной подвыборки считаются те же
характеристики, что в error_analysis.py (размер рамки, яркость, доля
пересветов, соотношение сторон), и берётся самый выраженный представитель
каждой причины.

Кадры разной формы, поэтому каждый вписывается в общую рамку 4:3 с полями
цвета фона: иначе подписи разъезжаются по вертикали и нижняя уходит
за край листа.

    python research/make_plate_collage.py
"""
import csv
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

HERE = Path('/opt/case17/app/research')
CROPS = Path('/opt/case17/data/crops')
OUT = HERE / 'pitch_assets'

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'

SAMPLE = 2500
SEED = 42
BOX = (4, 3)


def stats(row):
    p = CROPS / 'train' / f"{row['image_id']}.jpg"
    if not p.exists():
        return None
    im = Image.open(p).convert('RGB')
    g = im.convert('L')
    arr = np.asarray(g, dtype='float32')
    st = ImageStat.Stat(g)
    return {
        'image_id': row['image_id'],
        'bbox_w': int(row['w']), 'bbox_h': int(row['h']),
        'luma': st.mean[0],
        'bright_frac': float((arr > 235).mean()),
        'aspect': im.width / max(1, im.height),
        'contrast': st.stddev[0],
    }


def letterbox(im, box=BOX):
    """Вписать кадр в общую рамку, поля — цветом фона, кадр не растягивать."""
    tw, th = box[0] * 200, box[1] * 200
    k = min(tw / im.width, th / im.height)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                   Image.LANCZOS)
    canvas = Image.new('RGB', (tw, th), SURFACE)
    canvas.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
    return canvas


def main():
    rows = list(csv.DictReader((CROPS / 'crops_train.csv').open()))
    random.Random(SEED).shuffle(rows)
    pool = [s for s in (stats(r) for r in rows[:SAMPLE]) if s]
    print(f'подвыборка {len(pool)} кадров из {len(rows)}')

    # Каждая причина — свой измеримый критерий отбора.
    picks = [
        (max(pool, key=lambda s: -s['bbox_w'] * s['bbox_h']),
         'Машина далеко',
         lambda s: f"рамка {s['bbox_w']}×{s['bbox_h']} пикселей — на номер\n"
                   f"приходится меньше 20 пикселей по ширине"),
        (min(pool, key=lambda s: s['luma']),
         'Ночь',
         lambda s: f"яркость кадра {s['luma']:.0f} из 255 — номер тонет\n"
                   f"в шуме, кузов и фары ещё различимы"),
        (max(pool, key=lambda s: s['bright_frac']),
         'Блик и пересвет',
         lambda s: f"{s['bright_frac'] * 100:.1f}% кадра в пересвете — свет фар\n"
                   f"и отражения забивают зону номера"),
        (max(pool, key=lambda s: s['aspect']),
         'Машина боком',
         lambda s: f"соотношение сторон {s['aspect']:.2f} — машина снята вдоль,\n"
                   f"номерная плоскость уходит от камеры"),
    ]

    fig = plt.figure(figsize=(11, 7.2), facecolor=SURFACE)
    fig.text(0.012, 0.972, 'Почему номер не спасает', fontsize=14,
             color=INK, weight='bold', va='top')
    fig.text(0.012, 0.925,
             'Четыре кадра из выборки заказчика. Причина под каждым — не оценка '
             'на глаз, а измеренная характеристика кадра.',
             fontsize=10.5, color=INK2, va='top')

    # Ручная сетка: у subplot подписи прыгают вслед за высотой картинки.
    for i, (s, head, sub) in enumerate(picks):
        col, row = i % 2, i // 2
        # Рамка почти 4:3, чтобы поля вокруг кадра не съедали половину панели.
        x = 0.085 + col * 0.455
        y = 0.495 - row * 0.425
        ax = fig.add_axes([x, y, 0.33, 0.325])
        ax.imshow(letterbox(Image.open(CROPS / 'train' / f"{s['image_id']}.jpg")))
        ax.axis('off')
        ax.set_title(head, fontsize=11.5, color=INK, weight='bold', pad=6)
        fig.text(x + 0.165, y - 0.018, sub(s), ha='center', va='top',
                 fontsize=9, color=INK2, linespacing=1.6)
        print(f"  {head}: {s['image_id']}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / '10_plate_unreadable.jpg'
    fig.savefig(p, dpi=150, facecolor=SURFACE,
                pil_kwargs={'quality': 90, 'optimize': True})
    print(f'записано: {p}')


if __name__ == '__main__':
    main()
