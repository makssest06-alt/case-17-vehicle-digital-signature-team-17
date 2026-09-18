# research — замеры и обучение

Всё, что не идёт в продакшн-сервис: замеры готовых моделей, обучение головы,
подготовка полного дообучения. Результаты — в [results.md](results.md),
лицензии — в [LICENSES.md](LICENSES.md).

## Окружения

```bash
cd /opt/case17/app/research

# основное (timm, torch CPU)
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  torch==2.5.1+cpu torchvision==0.20.1+cpu timm==1.0.12 numpy==2.1.3 pillow==11.1.0 matplotlib

# отдельное для fast-reid (ему нужен numpy<2)
uv venv --python 3.11 .venv_fastreid && . .venv_fastreid/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  torch==2.5.1+cpu torchvision==0.20.1+cpu 'numpy<2' yacs termcolor tabulate scikit-learn pillow opencv-python-headless
```

## Пайплайн

```bash
python make_split.py                          # val_ids.txt + val_split.csv, seed=42
python evaluate.py --model <timm-имя> --tag X # zero-shot замер любой timm-модели
python extract_features.py --backbone clip_vitb16   # признаки в /opt/case17/data/features
python train_head.py --backbone clip_vitb16 --tag v1 --out-dim 512 --loss arcface
python eval_refusal.py --backbone clip_vitb16 --heads v1,v2,v3
```

Бэкбон везде выбирается параметром `--backbone` из реестра [backbones.py](backbones.py).
Добавить новый — дописать одну строку в `REGISTRY`, код трогать не нужно.

Доступные ключи: `clip_vitb16`, `dinov2_vits14`, `siglip2_base`, `resnet50_in1k`,
а также `veriwild_r50ibn` и `veri_r50ibn_sbs` — **эти два только для исследования**,
их датасеты non-commercial, см. LICENSES.md.

## Полное дообучение на GPU

`train_head.py` учит только голову поверх замороженных признаков — это дёшево
и работает на CPU. `finetune_clip.py` размораживает весь бэкбон, и это уже
задача для видеокарты.

Проверка, что код живой (работает и на CPU):

```bash
python finetune_clip.py --smoke --steps 2 --P 4 --K 2
```

Полный запуск на GPU-машине:

```bash
python finetune_clip.py \
  --backbone clip_vitb16 \
  --out-dim 512 --hidden 1024 --loss arcface \
  --epochs 30 --P 16 --K 4 \
  --lr-backbone 1e-5 --lr-head 3e-4 \
  --warmup 3 --workers 8 --amp \
  --tag finetune_clip_v1
```

Чекпоинт каждой эпохи пишется в `/opt/case17/models/finetune_clip_v1_ep*.pt`.

### Сколько это займёт

Обучающая часть: **8 644 кадра, 1 391 машина** (150 машин отложены в валидацию).
При `P=16, K=4` это батч 64 и **86 шагов на эпоху**, то есть 2 580 шагов на 30 эпох
и около 165 тысяч обработанных изображений.

| GPU | 30 эпох | 60 эпох | Чем ограничено |
|---|---|---|---|
| T4 (16 ГБ) | ~50–60 мин | ~1,5–2 ч | поровну вычисления и загрузка данных |
| A10 / A10G (24 ГБ) | ~25–35 мин | ~50–70 мин | в основном загрузка данных |
| A100 (40–80 ГБ) | ~15–25 мин | ~30–50 мин | почти целиком загрузка данных |

Оценки считались так: ViT-B/16 при входе 224 — примерно 17,6 GFLOPs на кадр вперёд
и втрое больше с обратным проходом, то есть около 8,7 PFLOPs на 30 эпох. На T4 при
реалистичных 25% утилизации это меньше 10 минут чистых вычислений — **узкое место
не матричные операции, а декодирование 165 тысяч JPEG**. Поэтому `--workers 8`
и быстрый диск важнее, чем класс карты, и разница между A10 и A100 почти не видна.

Практический вывод: **одной T4 достаточно**, брать A100 под эту задачу смысла нет.
Запас по времени позволяет прогнать 3–4 конфигурации за вечер и выбрать лучшую
по mAP на валидации.

При батче 64 и `--amp` ViT-B/16 занимает около 8–10 ГБ видеопамяти, то есть
влезает и в T4. Если памяти не хватит — уменьшать `--P`, а не `--K`:
для batch-hard triplet важно, чтобы на каждую машину приходилось не меньше 4 кадров.

---

# Обучение на бесплатной GPU (Kaggle)

У Kaggle есть бесплатная T4 — примерно 30 часов в неделю на аккаунт. Нашего обучения
там хватает с большим запасом, поэтому GPU-машину арендовать не обязательно.

Ноутбук: [`kaggle/train_reid.ipynb`](../kaggle/train_reid.ipynb). Он самодостаточный —
внутри и аугментации, и модель, и оценка, и режим отказа. Ничего доустанавливать не нужно.

## Шаг 1. Собрать архив кропов

На сервере:

```bash
/opt/case17/app/scripts/make_kaggle_archive.sh
```

Кладёт `/opt/case17/data/case17_crops_kaggle.zip` — около 1,1 ГБ, 11 416 jpg и 3 csv.
Служебные файлы macOS (`._*`, `.DS_Store`) в архив не попадают: из-за них при прошлой
распаковке количество файлов удвоилось, и проверка не сошлась.

Скачать себе:

```bash
scp lct:/opt/case17/data/case17_crops_kaggle.zip ~/Downloads/
```

## Шаг 2. Залить датасет на Kaggle

1. [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
2. Перетащить `case17_crops_kaggle.zip`. Kaggle распакует его сам.
3. Название — например `case17-crops`. **Visibility: Private.**
4. **Create**. Заливка 1,1 ГБ идёт несколько минут.

Данные организаторов, поэтому датасет **обязательно приватный**. Публиковать нельзя.

## Шаг 3. Создать ноутбук

1. [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **File → Import Notebook**
   → загрузить `kaggle/train_reid.ipynb`.
2. Справа в панели Settings:
   - **Accelerator → GPU T4 x2** (ноутбук использует одну, но выбор такой);
   - **Internet → On** — нужен, чтобы скачались предобученные веса SigLIP;
   - **Add Input → Datasets** → выбрать свой `case17-crops`.
3. **Run All**.

Ноутбук сам найдёт папку с `crops_train.csv` внутри `/kaggle/input` — путь подставлять
руками не нужно.

## Шаг 4. Дождаться

| Что | Сколько |
|---|---|
| Скачивание весов бэкбона | 1–2 мин |
| Одна эпоха (8 644 кадра, батч 64) | ~1,5–2 мин |
| 30 эпох + оценки | **50–70 мин** |
| 60 эпох | 1,5–2 ч |

Узкое место — не матричные операции, а декодирование 165 тысяч JPEG за прогон.
Поэтому `workers=4` в конфиге важнее, чем класс карты.

**Kaggle убивает интерактивную сессию через 12 часов простоя, а вкладку — раньше.**
Для длинного прогона жать **Save Version → Save & Run All (Commit)**: тогда ноутбук
считается на серверах Kaggle и вкладку можно закрыть.

## Шаг 5. Забрать веса и поставить в прототип

В готовом ноутбуке справа панель **Output** → скачать `best.pt` и `metrics.json`.

Дальше одной командой:

```bash
scp ~/Downloads/best.pt lct:/tmp/best.pt
ssh lct '/opt/case17/app/scripts/install_trained_weights.sh /tmp/best.pt'
```

Скрипт сам: прочитает из чекпоинта, какой был бэкбон и размерность, положит веса в
`/opt/case17/models`, пропишет `EMBED_MODEL=reid:<бэкбон>` и `REID_WEIGHTS` в `.env`,
пересоберёт образ api, поднимет стек и дождётся healthy. Если размерность вектора
изменилась — пересоздаст том базы, иначе pgvector ругнётся на несовпадение.

После этого переимпортировать галерею и пересчитать артефакты:

```bash
curl -X POST localhost:8000/v1/gallery/bulk-import -H 'Content-Type: application/json' \
  -d '{"images_dir":"/data/crops/gallery","csv_path":"/data/crops/crops_gallery.csv","truncate":true}'
docker compose exec api python scripts/make_submission.py
```

## Что менять, если хочется попробовать другое

Всё в ячейке `CFG`:

- `backbone` — любое имя из timm. По умолчанию SigLIP base (Apache 2.0).
- `epochs` — 30 хватает, 60 даёт немного больше и вдвое дольше.
- `P` и `K` — если не хватит видеопамяти, уменьшать **`P`, а не `K`**: batch-hard triplet
  требует минимум 4 кадра на машину, иначе ему не из чего выбирать.
- `loss` — `arcface` или `cosface`.

## Про приватность

Датасет приватный, ноутбук приватный. Kaggle не использует приватные данные для обучения
своих моделей, но данные всё равно уезжают на чужие серверы. Это данные организаторов —
если есть сомнения, лучше спросить модератора кейса до заливки.
