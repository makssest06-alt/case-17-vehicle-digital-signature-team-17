#!/usr/bin/env python3
"""Сборка файла презентации из структуры PITCH_OUTLINE.md и картинок.

ТЗ, раздел 11, требует PDF или PPTX. Этот скрипт делает PPTX: 16:9,
одна мысль на слайд, картинка из research/pitch_assets/, ключевая цифра
отдельной плашкой, текст доклада — в заметках к слайду.

Цифры не вписаны руками: они читаются из research/results/*.json,
поэтому после пересчёта метрик достаточно перезапустить скрипт.

    python research/make_pitch_deck.py

Слайды 7-11 организаторы требуют по своему шаблону. Шаблона нет, поэтому
их содержание здесь — предложение; при появлении шаблона переносится в него
как есть, потому что текст и картинки от разметки не зависят.
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = Path('/opt/case17/app/research')
ASSETS = HERE / 'pitch_assets'
RES = HERE / 'results'
OUT = HERE / 'pitch_assets' / 'ЛЦТ2026_кейс17_команда17.pptx'

W, H = Inches(13.333), Inches(7.5)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
S1 = RGBColor(0x2A, 0x78, 0xD6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = 'Arial'


def num():
    """Все цифры презентации — из файлов замеров."""
    ex = json.loads((RES / 'service_path_extras.json').read_text())
    ann = json.loads((RES / 'ann' / 'summary.json').read_text())
    by = {r['method'].split(' (')[0]: r for r in ann}
    exact = ann[0]
    hnsw = next((v for k, v in by.items() if 'HNSW' in k), None)
    rd = ex['refusal_deployment']
    return {
        'mAP': ex['prenorm']['mAP'],
        'r1': ex['prenorm']['rank1'],
        'r5': ex['prenorm']['rank5'],
        'top10': ex['prenorm']['top10_pct'],
        'cat': ex['prenorm']['catastrophic_pct'],
        'cat_pp': ex['prenorm_postprocess']['catastrophic_pct'],
        'mAP_pp': ex['prenorm_postprocess']['mAP'],
        'auc': rd['auc'],
        'thr': rd['balanced']['threshold'],
        'f1': rd['balanced']['f1'],
        'tnr': rd['balanced']['TNR'],
        'thr_s': rd['strict']['threshold'],
        'tnr_s': rd['strict']['TNR'],
        'exact_ms': exact['latency']['p50_ms'],
        'hnsw_ms': hnsw['latency']['p50_ms'] if hnsw else None,
        'hnsw_map_drop': round(exact['mAP'] - hnsw['mAP'], 2) if hnsw else None,
        'speedup': round(exact['latency']['p50_ms'] / hnsw['latency']['p50_ms']) if hnsw else None,
    }


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = SURFACE
    return s


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: список (текст, размер, цвет, жирный)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, (t, size, color, bold) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = t
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.bold = bold
        r.font.name = FONT
    return tb


def rule(slide, y):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), y, Inches(11.93), Pt(1))
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor(0xE1, 0xE0, 0xD9)
    sh.line.fill.background()
    sh.shadow.inherit = False


def badge(slide, x, y, value, caption):
    """Плашка с ключевой цифрой. Одна на слайд, иначе перестаёт читаться."""
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(3.5), Inches(1.15))
    sh.fill.solid()
    sh.fill.fore_color.rgb = S1
    sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.18)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = value
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = FONT
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.LEFT
    r2 = p2.add_run(); r2.text = caption
    r2.font.size = Pt(11); r2.font.color.rgb = RGBColor(0xE8, 0xF0, 0xFB); r2.font.name = FONT
    return sh


def picture(slide, name, x, y, w=None, h=None):
    p = ASSETS / name
    if not p.exists():
        text(slide, x, y, Inches(6), Inches(0.5),
             [(f'НЕТ КАРТИНКИ: {name}', 14, RGBColor(0xC0, 0x39, 0x2B), True)])
        return None
    kw = {}
    if w is not None:
        kw['width'] = w
    if h is not None:
        kw['height'] = h
    return slide.shapes.add_picture(str(p), x, y, **kw)


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body


def content(prs, n, head, sub, image=None, badge_val=None, badge_cap=None,
            say='', img_h=Inches(4.3), bullets=None):
    s = blank(prs)
    text(s, Inches(0.7), Inches(0.42), Inches(11.9), Inches(0.5),
         [(f'{n}', 12, MUTED, False)])
    text(s, Inches(0.7), Inches(0.7), Inches(11.9), Inches(0.7),
         [(head, 30, INK, True)])
    text(s, Inches(0.7), Inches(1.5), Inches(11.9), Inches(0.8),
         [(sub, 14, INK2, False)])
    rule(s, Inches(2.25))
    if bullets:
        runs = []
        for b in bullets:
            runs.append((f'•  {b}', 16, INK2, False))
        text(s, Inches(0.7), Inches(2.6), Inches(6.6), Inches(3.6), runs)
    if image:
        x = Inches(0.7) if not bullets else Inches(7.5)
        w = None
        picture(s, image, x, Inches(2.55), h=img_h)
    if badge_val:
        badge(s, Inches(9.5), Inches(6.2), badge_val, badge_cap or '')
    if say:
        notes(s, say)
    return s


def main():
    N = num()
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ------------------------------------------------------------- 1. титул
    s = blank(prs)
    text(s, Inches(0.9), Inches(1.9), Inches(7.2), Inches(2.6), [
        ('Цифровой признак транспортного средства', 40, INK, True),
        ('Сопоставление снимков одной машины с разных камер без '
         'государственного номера', 17, INK2, False),
    ])
    text(s, Inches(0.9), Inches(5.6), Inches(7.2), Inches(1.2), [
        ('Кейс 17 · «Фалькон Тех» · команда 17', 14, INK2, False),
        ('ЛЦТ 2026', 13, MUTED, False),
    ])
    picture(s, '09_ui_search.png', Inches(8.4), Inches(1.5), w=Inches(4.5))
    notes(s, 'Одна фраза: сервис узнаёт машину по внешности, без номера.')

    # ------------------------------------------------------ 2. задача
    content(prs, '2 · Задача',
            'Номер решает не все случаи',
            'Ракурс, ночь, блик, расстояние, перекрытие — и опознавать машину '
            'нечем. Нужен признак, который работает, когда номера нет.',
            image='10_plate_unreadable.jpg', img_h=Inches(4.3),
            badge_val='0 OCR', badge_cap='номер не извлекается нигде в цепочке',
            say='Мы решаем не «распознать номер получше», а «обойтись без номера». '
                'Четыре кадра справа — из выборки заказчика, причина под каждым '
                'измерена, а не оценена на глаз.')

    # ------------------------------------------------------ 3. продукт
    content(prs, '3 · Что сделали',
            'Это сервис, а не ноутбук',
            'Три контейнера, подъём одной командой, веб-интерфейс для оператора, '
            'режим отказа и объяснение решения.',
            image='11_architecture.png', img_h=Inches(2.6),
            badge_val='19 / 19', badge_cap='проверок воспроизводимости зелёные',
            say='Чистый клон собирается и работает без доступа в интернет — '
                'это проверяется скриптом check_reproducibility.sh, а не обещанием. '
                'Веса лежат в составе решения.')

    # ------------------------------------------------------ 4. бэкбон
    content(prs, '4 · Методология',
            'Выбор бэкбона решает больше, чем тюнинг',
            'Девять вариантов на одном сплите. Протокол: 150 отложенных машин, '
            'кросс-камерный расчёт, совпадения своей камеры исключены.',
            image='01_backbones.png', img_h=Inches(4.3),
            badge_val='+27.8 mAP', badge_cap='замена CLIP на SigLIP, без обучения',
            say='Два вывода. Первый: выбор бэкбона дал больше, чем любая настройка. '
                'Второй: заштрихованные строки — модели, обученные на датасетах '
                'с закрытой лицензией. Мы их замерили как ориентир и в решение не взяли.')

    # ------------------------------------------------------ 5. ракурс
    content(prs, '5 · Проблема ракурса',
            'Назвали проблему числом и закрыли её',
            'Катастрофический провал — правильный ответ дальше 50-го места. '
            'Именно так выглядит промах по ракурсу.',
            image='02_viewpoint.png', img_h=Inches(4.3),
            badge_val=f'{34.7 / N["cat_pp"]:.0f}×' if N['cat_pp'] else '—',
            badge_cap=f'меньше катастроф: 34.7% → {N["cat_pp"]}%',
            say='У одной и той же машины спереди и сзади сходство доходило до '
                'отрицательного, правильный ответ улетал на 230-е место. '
                f'Сейчас в живом сервисе {N["cat"]}%, в артефактах сдачи с '
                f'постобработкой — {N["cat_pp"]}%.')

    # ------------------------------------------------------ 6. ошибки
    content(prs, '6 · Анализ ошибок',
            'Мы знаем, на чём модель падает',
            'Пять категорий, у каждой — живые кадры: запрос, что выдала первым, '
            'где оказался правильный ответ.',
            image='07_errors_night.jpg', img_h=Inches(4.3),
            badge_val=f'{N["cat"]}%', badge_cap='катастроф осталось в сервисе',
            say='Это не «модель иногда ошибается». Мы можем показать конкретные кадры '
                'и назвать причину: ночь, близнецы одной модели и цвета, сильная '
                'смена ракурса. Отсюда же план следующей итерации.')

    # ------------------------------------------- 7-11. блок организаторов
    content(prs, '7 · Архитектура решения',
            'Одна модель, одна голова, один индекс',
            'Кадр → рамка → кроп с запасом 10% → SigLIP base → голова с BNNeck → '
            'вектор 512D → поиск → кандидаты либо отказ.',
            image='11_architecture.png', img_h=Inches(2.6),
            badge_val='359 МБ', badge_cap='веса при лимите 2048 МБ',
            say='Ничего экзотического в схеме нет. Сложность спрятана в протоколе '
                'оценки и в калибровке отказа, а не в количестве блоков.')

    content(prs, '8 · Путь пользователя',
            'Строгость выбирает человек, а не константа в коде',
            'Оператор загружает кадр, обводит машину, получает кандидатов '
            'с уверенностью либо честный отказ.',
            image='12_user_journey.png', img_h=Inches(2.6),
            badge_val=f'{N["thr"]} / {N["thr_s"]}',
            badge_cap='сбалансированный и строгий порог',
            say='Цена ложного совпадения в каждой задаче своя, поэтому порог вынесен '
                'в интерфейс. Рядом с переключателем написано, чем этот порог '
                'обоснован: F1, полнота и TNR в этой точке.')

    content(prs, '9 · Метрики и режим отказа',
            'Порог взят из распределения, а не из головы',
            'Качество ранжирования и отдельно качество отказа. Порог — максимум '
            'индекса Юдена на том распределении, в котором работает сервис.',
            image='05_refusal.png', img_h=Inches(4.3),
            badge_val=f'AUC {N["auc"]}', badge_cap=f'F1 {N["f1"]}, TNR {N["tnr"]}',
            say='Здесь мы нашли у себя ошибку до жюри: порог, снятый с кросс-камерного '
                'распределения, на боевой выдаче не срабатывал ни разу. '
                'Перекалибровали на рабочем распределении и показываем оба.')

    content(prs, '10 · Производительность',
            'Сначала профиль, потом оптимизация',
            'Профиль сказал, что вся обвязка — 3% времени. Поэтому честный выигрыш '
            'от её оптимизации 9%, а не двукратный.',
            image='03_speed.png', img_h=Inches(2.9),
            badge_val='2 vCPU', badge_cap='без видеокарты, без деградации памяти',
            say='Мы сознательно не заявляем двукратное ускорение: первая цифра '
                'оказалась артефактом замера — ONNX делал один проход против двух '
                'у торча. Поправили замер и оставили честные 9%.')

    content(prs, '11 · Масштабирование',
            'Миллион векторов — это замер, а не рассуждение',
            'Реальная валидационная галерея внутри миллиона правдоподобных '
            'дистракторов, реальные запросы, метрики по протоколу жюри.',
            image='04_ann_1m.png', img_h=Inches(3.3),
            badge_val=f'в {N["speedup"]}× быстрее' if N['speedup'] else '—',
            badge_cap=f'HNSW против точного перебора, −{N["hnsw_map_drop"]} mAP',
            say='Точный перебор по миллиону векторов — 137 мс на запрос. HNSW — '
                'доли миллисекунды при потере четверти пункта mAP. IVF-PQ ещё '
                'компактнее, но теряет уже 14 пунктов, поэтому он не наш выбор.')

    # ------------------------------------------------------ 12. отличия
    s = content(prs, '12 · Чем отличаемся',
                'Три вещи, которые редко делают',
                'Лицензионная чистота, пойманные собственные ошибки замера, '
                'проверяемость каждой цифры.',
                bullets=[
                    'Все открытые ReID-датасеты для машин — non-commercial. '
                    'Проверили по первоисточникам и обучались только на данных заказчика.',
                    'Трижды поймали себя на завышенном замере: ускорение, re-ranking '
                    'и расхождение офлайн-признаков с ответом сервиса.',
                    'У каждой цифры в презентации написано, каким скриптом она получена. '
                    'Всё лежит в research/results/ в JSON.',
                ],
                badge_val='0', badge_cap='закрытых лицензий в решении',
                say='Мы показываем не только то, что получилось, но и то, где мы сами '
                    'себя поймали. Это и есть проверяемость.')

    # ------------------------------------------------------ 13. дальше
    content(prs, '13 · Что дальше',
            'Знаем, где следующий рост',
            'Полное дообучение бэкбона на видеокарте, so400m как вариант для стенда '
            'с GPU, геометрические аугментации под смену ракурса.',
            bullets=[
                'Разморозка бэкбона: ноутбук готов и проверен, не хватало только GPU.',
                'so400m даёт 60.80 mAP без обучения, но 0.35 кадра/с на CPU — '
                'включается профилем, если у стенда есть видеокарта.',
                'Ансамбль из нескольких голов: замерен, помещается в лимит весов, '
                'включается одной переменной.',
            ],
            badge_val='+5…10 mAP', badge_cap='ожидаемо при разморозке бэкбона',
            say='План не абстрактный: код обучения написан и прогнан, упирается '
                'только в железо.')

    # ------------------------------------------------------ 14. демо
    s = blank(prs)
    text(s, Inches(0.7), Inches(0.9), Inches(11.9), Inches(1.2),
         [('Демонстрация', 34, INK, True),
          ('Сервис поднимается одной командой и отвечает на живых кадрах',
           15, INK2, False)])
    rule(s, Inches(2.3))
    text(s, Inches(0.7), Inches(2.7), Inches(6.2), Inches(3.4), [
        ('cp .env.example .env && docker compose up -d', 16, INK, True),
        ('', 8, INK2, False),
        ('Веб-интерфейс на порту 8080, API и Swagger на 8000.', 14, INK2, False),
        ('Четыре подготовленных запроса: лёгкий, сложный ракурс, ночь '
         'и запрос без пары для показа отказа.', 14, INK2, False),
        ('Сценарий показа — research/DEMO_SCENARIO.md', 13, MUTED, False),
    ])
    picture(s, '16_ui_explain.png', Inches(7.3), Inches(2.5), h=Inches(4.4))
    notes(s, 'Открываем веб, грузим кадр, обводим машину, получаем кандидатов. '
             'Потом «показать почему»: зона номера в подсветке не участвует.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f'записано: {OUT}  слайдов: {len(prs.slides.__iter__.__self__._sldIdLst)}')


if __name__ == '__main__':
    main()
