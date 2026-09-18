#!/usr/bin/env python3
"""Генерирует два ноутбука из одного источника: Kaggle и Colab.

    python kaggle/build_notebooks.py

Отличаются только тремя вещами — откуда берутся данные, куда пишутся
чекпоинты и что доустанавливать. Вся логика обучения общая, поэтому
правка делается в одном месте.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(keepends=True)}


# ------------------------------------------------------------------ платформы

KAGGLE_INTRO = """# Кейс 17 — дообучение ReID-модели на Kaggle (бесплатная T4)

Обучает **весь бэкбон целиком**, не только голову.

## Что нажать перед запуском

1. Справа **Settings → Accelerator → GPU T4 x2**
2. Справа **Settings → Internet → On** (нужен для скачивания весов бэкбона)
3. Справа **Add Input → Datasets** → выбрать свой датасет с кропами
4. **Run All**

Для длинного прогона лучше **Save Version → Save & Run All (Commit)** — тогда
считает сервер Kaggle и вкладку можно закрыть.

На выходе в `/kaggle/working/`: `best.pt`, `last.pt`, `metrics.json`, `log.txt`.
"""

COLAB_INTRO = """# Кейс 17 — дообучение ReID-модели в Google Colab

Запасной вариант, если на Kaggle кончилась квота GPU.

## Что нажать перед запуском

1. **Среда выполнения → Сменить среду выполнения → Аппаратный ускоритель → T4 GPU**
2. Запустить первую ячейку — она подключит Google Диск
3. Положить архив `case17_crops_kaggle.zip` на Диск в папку `case17/`
4. **Среда выполнения → Выполнить все**

Чекпоинты пишутся **на Диск**, поэтому обрыв сессии не теряет прогресс:
повторный запуск продолжит с последней сохранённой эпохи.

Бесплатный Colab отключает сессию примерно через 90 минут простоя и
ограничивает время GPU в сутки. Обучение на 30 эпох укладывается, но
если оборвётся — просто запустите ноутбук заново.
"""

KAGGLE_SETUP = '''import subprocess, sys, os, json, time, math, random, io, csv
from pathlib import Path
from collections import defaultdict

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "GPU не найден!")

import torch
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
assert torch.cuda.is_available(), "Включи GPU: Settings -> Accelerator -> GPU T4"

# В Kaggle свой предустановленный timm — проверяем версию и доставляем при нужде.
NEED_TIMM = "1.0.12"
try:
    import timm
    from packaging.version import Version
    if Version(timm.__version__) < Version(NEED_TIMM):
        raise ImportError(f"timm {timm.__version__} старее {NEED_TIMM}")
except Exception as e:
    print("ставлю timm:", e)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"timm>={NEED_TIMM}"], check=True)
    import importlib, timm
    importlib.reload(timm)
print("timm", timm.__version__)

WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)

# Данные: ищем папку с crops_train.csv среди подключённых датасетов
DATA_DIR = None
for cand in Path("/kaggle/input").rglob("crops_train.csv"):
    DATA_DIR = cand.parent
    break
assert DATA_DIR is not None, (
    "crops_train.csv не найден. Подключи датасет: Add Input -> Datasets")
print("DATA_DIR =", DATA_DIR)
print("картинок в train:", len(list((DATA_DIR / "train").glob("*.jpg"))))'''

COLAB_SETUP = '''import subprocess, sys, os, json, time, math, random, io, csv, zipfile, shutil
from pathlib import Path
from collections import defaultdict

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "GPU не найден!")

import torch
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "Включи GPU: Среда выполнения -> Сменить среду выполнения -> T4 GPU")

# В Colab timm обычно не предустановлен
NEED_TIMM = "1.0.12"
try:
    import timm
    from packaging.version import Version
    if Version(timm.__version__) < Version(NEED_TIMM):
        raise ImportError(f"timm {timm.__version__} старее {NEED_TIMM}")
except Exception as e:
    print("ставлю timm:", e)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"timm>={NEED_TIMM}"], check=True)
    import importlib, timm
    importlib.reload(timm)
print("timm", timm.__version__)

# Диск нужен, чтобы чекпоинты пережили обрыв сессии
from google.colab import drive
drive.mount("/content/drive")

DRIVE = Path("/content/drive/MyDrive/case17")
DRIVE.mkdir(parents=True, exist_ok=True)
WORK = DRIVE                      # чекпоинты пишем на Диск
LOCAL = Path("/content/crops")    # данные распаковываем локально — так быстрее

ZIP = DRIVE / "case17_crops_kaggle.zip"
assert ZIP.exists(), f"Положи архив кропов на Диск: {ZIP}"

if not (LOCAL / "crops_train.csv").exists():
    print("распаковываю архив (пара минут)...")
    LOCAL.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(LOCAL)
DATA_DIR = LOCAL if (LOCAL / "crops_train.csv").exists() else next(
    LOCAL.rglob("crops_train.csv")).parent
print("DATA_DIR =", DATA_DIR)
print("картинок в train:", len(list((DATA_DIR / "train").glob("*.jpg"))))'''


CFG_INTRO = """## 2. Конфигурация: выберите пресет

Три готовых конфигурации. **Запускайте по очереди и сравнивайте `mAP` на val.**
Меняется ровно одна строка — `PRESET`.

| Пресет | Что пробует | Зачем |
|---|---|---|
| `baseline` | случайный P×K, обычные аугментации, ArcFace + batch-hard triplet | точка отсчёта, с ней сравниваем остальные |
| `viewpoint` | позитивы **обязательно с другой камеры**, геометрические аугментации, батчи из визуально похожих машин | бьёт в главную беду модели — смену ракурса |
| `centroid` | всё из `viewpoint` + притягивание к центроиду своей машины | усредняет ракурсы в одну точку, чтобы «спереди» и «сзади» сошлись |

### Почему именно это

Анализ ошибок показал: катастрофические промахи — почти все про ракурс.
У одного и того же автомобиля с другой стороны сходство **отрицательное**,
правильный ответ улетает на 230–250 место. Модель выучила «похожесть картинки»,
а не «тождество объекта».

- **Позитивы с другой камеры.** В обычном P×K сэмплере K кадров машины берутся
  случайно, и часто это соседние кадры с одной камеры — почти одинаковые
  картинки, на которых triplet ничему не учится. Здесь K кадров подбираются так,
  чтобы покрыть максимум разных камер: каждая пара «якорь-позитив» становится
  кросс-камерной, то есть ровно той задачей, которую считает жюри.
- **Геометрические аугментации.** Перспектива и случайный кроп куска машины
  заставляют узнавать автомобиль по деталям (диски, фары, наклейки),
  а не по общему силуэту, который при смене ракурса меняется полностью.
- **Батчи из похожих машин.** Один прогон замороженным бэкбоном даёт центроид
  каждой машины; батч набирается из машин с близкими центроидами. Негативы
  в батче становятся действительно трудными — тот же тип кузова и цвет,
  а не случайный грузовик. Это дешёвая замена ручному подбору «той же модели
  и цвета», для которого у нас нет меток.
- **Центроид.** У каждой машины считается средний вектор по всем её кадрам,
  и каждый кадр притягивается к нему. Центроид усредняет ракурсы, поэтому
  вид спереди и вид сзади тянутся к одной точке, а не друг к другу напрямую.
"""

CFG_CELL = '''PRESET = "baseline"   # baseline | viewpoint | centroid  <-- МЕНЯТЬ ЗДЕСЬ

BASE = dict(
    backbone      = "vit_base_patch16_siglip_224.webli",  # SigLIP base, Apache 2.0
    input_size    = 224,
    hidden        = 1024,
    out_dim       = 512,
    dropout       = 0.3,

    loss          = "arcface",   # arcface | cosface
    margin        = 0.25,
    scale         = 30.0,
    tri_margin    = 0.3,
    tri_weight    = 1.0,
    centroid_weight = 0.0,       # вес центроидного лосса

    P             = 16,          # машин в батче
    K             = 4,           # кадров на машину -> батч = P*K
    epochs        = 30,
    lr_backbone   = 1e-5,
    lr_head       = 3e-4,
    weight_decay  = 5e-4,
    warmup_epochs = 3,

    cross_camera_positives = False,  # K кадров с разных камер
    hard_batches           = False,  # батч из визуально похожих машин
    geometry_aug           = False,  # перспектива и кроп частей

    grad_checkpointing = "auto",  # auto | True | False — экономит память на крупных бэкбонах
    workers       = 4,
    amp           = True,
    seed          = 42,
    eval_every    = 5,
    save_every    = 2,
    n_val_ids     = 150,
)

PRESETS = {
    "baseline": {},
    "viewpoint": dict(
        cross_camera_positives = True,
        hard_batches           = True,
        geometry_aug           = True,
    ),
    "centroid": dict(
        cross_camera_positives = True,
        hard_batches           = True,
        geometry_aug           = True,
        centroid_weight        = 1.0,
        tri_weight             = 0.5,   # часть веса уходит центроиду
    ),
}

assert PRESET in PRESETS, f"неизвестный пресет {PRESET}, доступны: {list(PRESETS)}"
CFG = {**BASE, **PRESETS[PRESET]}
CFG["preset"] = PRESET

# --- автоподстройка под крупные бэкбоны и объём видеопамяти ---
# so400m это ~428M параметров: веса + градиенты + два момента AdamW уже около
# 7 ГБ, и на T4 с 16 ГБ батч 64 обычно не влезает. Поэтому для крупных моделей
# включаем gradient checkpointing (пересчёт активаций вместо их хранения:
# примерно на треть медленнее, но памяти нужно кратно меньше) и уменьшаем P.
# Уменьшаем именно P, а не K: batch-hard triplet требует минимум 4 кадра
# на машину, иначе ему не из чего выбирать трудные пары.
_BIG = any(t in CFG["backbone"] for t in ("so400m", "large", "giantopt", "huge"))
_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

if CFG["grad_checkpointing"] == "auto":
    CFG["grad_checkpointing"] = bool(_BIG)

if _BIG and _vram and _vram < 24 and CFG["P"] > 8:
    print(f"крупный бэкбон ({CFG['backbone']}) и {_vram:.0f} ГБ видеопамяти -> "
          f"P уменьшен с {CFG['P']} до 8 (батч {8 * CFG['K']})")
    CFG["P"] = 8

print(f"видеопамять: {_vram:.1f} ГБ | gradient checkpointing: {CFG['grad_checkpointing']}")

print(f"пресет: {PRESET}")
for k in ("cross_camera_positives", "hard_batches", "geometry_aug", "centroid_weight"):
    print(f"  {k:<24} {CFG[k]}")
print()
print(json.dumps(CFG, indent=2, ensure_ascii=False))'''


def build(platform: str):
    intro = KAGGLE_INTRO if platform == "kaggle" else COLAB_INTRO
    setup = KAGGLE_SETUP if platform == "kaggle" else COLAB_SETUP
    cells = [md(intro), md("## 1. Окружение и данные"), code(setup)]

    cells.append(md(CFG_INTRO))
    cells.append(code(CFG_CELL))

    cells.append(md('''## 3. Сплит

**Тот же, что на сервере:** 150 машин (`seed=42`) в валидацию, остальные в обучение.
Gallery — по одному кадру с каждой камеры, остальные кадры уходят в query.
Кадры с той же камеры, что у query, исключаются из выдачи — как считает жюри.'''))
    cells.append(code('''rows = list(csv.DictReader((DATA_DIR / "crops_train.csv").open()))
by_vehicle = defaultdict(list)
for r in rows:
    by_vehicle[r["vehicle_id"]].append(r)

vehicles = sorted(by_vehicle)
val_ids = set(random.Random(CFG["seed"]).sample(vehicles, CFG["n_val_ids"]))

train_rows = [r for r in rows if r["vehicle_id"] not in val_ids]
val_rows   = [r for r in rows if r["vehicle_id"] in val_ids]
train_vids = sorted({r["vehicle_id"] for r in train_rows})
vid2lab = {v: i for i, v in enumerate(train_vids)}
for r in train_rows:
    r["_label"] = vid2lab[r["vehicle_id"]]

by_v = defaultdict(lambda: defaultdict(list))
for r in val_rows:
    by_v[r["vehicle_id"]][r["camera_id"]].append(r)
val_gal, val_qry = [], []
for v in sorted(by_v):
    for cam in sorted(by_v[v], key=int):
        items = sorted(by_v[v][cam], key=lambda r: r["image_id"])
        val_gal.append(items[0]); val_qry.extend(items[1:])

print(f"train: {len(train_rows)} кадров / {len(train_vids)} машин")
print(f"val:   gallery {len(val_gal)}, query {len(val_qry)}")'''))

    cells.append(md('''## 4. Аугментации

Под наши условия, а не общие: каждый седьмой кадр в датасете ночной,
плюс блики фар, смаз от движения и компрессия камеры.'''))
    cells.append(code('''import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader

class NightAndGlare:
    """Ночь, блики, смаз, JPEG-компрессия, отражение."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        r = random.random
        if r() < self.p:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.25, 0.7))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.0))
        if r() < 0.3:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(1.2, 1.8))
        if r() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.6)))
        if r() < 0.3:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(25, 70))
            buf.seek(0); img = Image.open(buf).convert("RGB")
        if r() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class GeometryAug:
    """Перспектива и кроп куска машины — бьёт прямо в проблему ракурса.

    При смене ракурса общий силуэт меняется полностью, поэтому модель должна
    научиться узнавать автомобиль по деталям: дискам, фарам, наклейкам,
    форме стоек. Случайный кроп 60-100% площади заставляет её это делать.
    """

    def __init__(self, p_persp=0.5, p_crop=0.7, strength=0.25):
        self.p_persp, self.p_crop, self.strength = p_persp, p_crop, strength

    def __call__(self, img):
        w, h = img.size
        if random.random() < self.p_crop:
            scale = random.uniform(0.6, 1.0)
            ratio = random.uniform(0.85, 1.18)
            cw = min(w, int(round(w * scale * ratio)))
            ch = min(h, int(round(h * scale / ratio)))
            x = random.randint(0, max(0, w - cw))
            y = random.randint(0, max(0, h - ch))
            img = img.crop((x, y, x + cw, y + ch))
            w, h = img.size
        if random.random() < self.p_persp:
            d = self.strength
            jitter = lambda v, lim: v + random.uniform(-lim, lim)
            src = [(0, 0), (w, 0), (w, h), (0, h)]
            dst = [(jitter(x, w * d), jitter(y, h * d)) for x, y in src]
            coeffs = _perspective_coeffs(dst, src)
            if coeffs is not None:
                img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        return img


def _perspective_coeffs(src, dst):
    """Коэффициенты перспективного преобразования по четырём парам точек."""
    A, B = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    try:
        res = np.linalg.solve(np.array(A, dtype="float64"), np.array(B, dtype="float64"))
        return res.tolist()
    except np.linalg.LinAlgError:
        return None


class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.2)):
        self.p, self.scale = p, scale
    def __call__(self, t):
        if random.random() > self.p:
            return t
        c, h, w = t.shape
        for _ in range(10):
            area = h * w * random.uniform(*self.scale)
            ar = random.uniform(0.3, 3.3)
            eh, ew = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
            if eh < h and ew < w:
                y, x = random.randint(0, h - eh), random.randint(0, w - ew)
                t[:, y:y+eh, x:x+ew] = torch.randn(c, eh, ew) * 0.2
                return t
        return t'''))

    cells.append(md("## 5. Датасет и сэмплеры\\n\\nP машин × K кадров — обязательное условие для batch-hard triplet.\\nСам DataLoader создаётся ниже, после модели: трудным батчам нужен прогон бэкбоном."))
    cells.append(code('''_probe = timm.create_model(CFG["backbone"], pretrained=False, num_classes=0)
_dc = timm.data.resolve_data_config({}, model=_probe)
_dc["input_size"] = (3, CFG["input_size"], CFG["input_size"])
tf = timm.data.create_transform(**_dc, is_training=False)
del _probe
print("препроцессинг:", _dc)

class CropDataset(Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train
        self.aug = NightAndGlare() if train else None
        self.geom = GeometryAug() if (train and CFG["geometry_aug"]) else None
        self.erase = RandomErasing() if train else None
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(DATA_DIR / "train" / (r["image_id"] + ".jpg")).convert("RGB")
        if self.geom:
            img = self.geom(img)
        if self.aug:
            img = self.aug(img)
        x = tf(img)
        if self.erase:
            x = self.erase(x)
        return x, r.get("_label", -1)

class PKBatchSampler(torch.utils.data.Sampler):
    """P машин x K кадров.

    Два переключателя поверх обычного P x K:

    * cross_camera_positives — K кадров машины подбираются так, чтобы покрыть
      максимум РАЗНЫХ камер. В обычном сэмплере часто выпадают соседние кадры
      одной камеры, почти одинаковые: triplet на них ничему не учится.
      Здесь каждая пара «якорь-позитив» кросс-камерная — ровно та задача,
      которую считает жюри.

    * hard_batches — P машин берутся не случайно, а из визуально похожих
      (по центроидам, посчитанным замороженным бэкбоном). Негативы в батче
      становятся трудными: тот же тип кузова и цвет, а не случайный грузовик.
    """

    def __init__(self, rows, P, K, seed=42, cross_camera=False, neighbors=None):
        self.rows = rows
        self.by = defaultdict(list)
        self.by_cam = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(rows):
            self.by[r["vehicle_id"]].append(i)
            self.by_cam[r["vehicle_id"]][r["camera_id"]].append(i)
        self.vids = sorted(self.by)
        self.P, self.K = P, K
        self.cross_camera = cross_camera
        self.neighbors = neighbors          # vid -> список похожих vid
        self.rng = random.Random(seed)

    def _pick_frames(self, vid):
        if not self.cross_camera:
            pool = self.by[vid]
            return (self.rng.sample(pool, self.K) if len(pool) >= self.K
                    else [self.rng.choice(pool) for _ in range(self.K)])
        cams = list(self.by_cam[vid])
        self.rng.shuffle(cams)
        picked = []
        # сначала по одному кадру с каждой камеры — максимум разных точек съёмки
        for cam in cams:
            picked.append(self.rng.choice(self.by_cam[vid][cam]))
            if len(picked) == self.K:
                return picked
        # камер меньше K — добираем остальными кадрами
        rest = [i for i in self.by[vid] if i not in picked]
        self.rng.shuffle(rest)
        picked += rest[: self.K - len(picked)]
        while len(picked) < self.K:
            picked.append(self.rng.choice(self.by[vid]))
        return picked

    def _pick_vehicles(self):
        """Группы по P машин на эпоху."""
        vids = self.vids[:]
        self.rng.shuffle(vids)
        if not self.neighbors:
            return [vids[i:i+self.P] for i in range(0, len(vids) - self.P + 1, self.P)]
        groups, used = [], set()
        for anchor in vids:
            if anchor in used:
                continue
            group = [anchor]
            used.add(anchor)
            for cand in self.neighbors.get(anchor, []):
                if cand not in used:
                    group.append(cand); used.add(cand)
                if len(group) == self.P:
                    break
            if len(group) < self.P:          # добираем случайными
                for cand in vids:
                    if cand not in used:
                        group.append(cand); used.add(cand)
                    if len(group) == self.P:
                        break
            if len(group) == self.P:
                groups.append(group)
        return groups

    def __iter__(self):
        for group in self._pick_vehicles():
            batch = []
            for v in group:
                batch += self._pick_frames(v)
            yield batch

    def __len__(self):
        return len(self.vids) // self.P


@torch.no_grad()
def build_vehicle_neighbors(model, rows, top=None, bs=64):
    """Центроид каждой машины замороженным бэкбоном -> ближайшие соседи.

    Один прогон по train (около минуты на T4). Нужен только для hard_batches.
    """
    top = top or (CFG["P"] - 1)
    model.eval()
    ds = CropDataset(rows, train=False)
    feats = []
    for i in range(0, len(rows), bs):
        x = torch.stack([ds[j][0] for j in range(i, min(i + bs, len(rows)))]).to(device)
        with torch.autocast("cuda", enabled=CFG["amp"]):
            f = model(x)[1]
        feats.append(F.normalize(f.float(), dim=1).cpu())
    model.train()
    feats = torch.cat(feats).numpy()

    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["vehicle_id"]].append(i)
    vids = sorted(by)
    cent = np.stack([feats[by[v]].mean(0) for v in vids])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    sim = cent @ cent.T
    np.fill_diagonal(sim, -2.0)
    order = np.argsort(-sim, axis=1)[:, :top]
    return {vids[i]: [vids[j] for j in order[i]] for i in range(len(vids))}'''))

    cells.append(md('''## 6. Модель

`бэкбон → Linear(dim→1024) → BN → ReLU → Dropout → Linear(1024→512) → BNNeck`

**BNNeck:** triplet тянет признак в одну сторону, классификационный лосс в другую.
BatchNorm между ними снимает конфликт — triplet берёт вектор **до** BN,
классификация и инференс — **после**.'''))
    cells.append(code('''import torch.nn as nn
import torch.nn.functional as F

class ReidHead(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.bnneck = nn.BatchNorm1d(out_dim)
        self.bnneck.bias.requires_grad_(False)
    def forward(self, x):
        h = self.drop(F.relu(self.bn1(self.fc1(x))))
        feat = self.fc2(h)
        return feat, self.bnneck(feat)

class MarginHead(nn.Module):
    def __init__(self, dim, n_cls, kind, scale, margin):
        super().__init__()
        self.kind, self.s, self.m = kind, scale, margin
        self.weight = nn.Parameter(torch.empty(n_cls, dim))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, feat_bn, labels):
        cos = F.linear(F.normalize(feat_bn), F.normalize(self.weight)).clamp(-1+1e-7, 1-1e-7)
        target = cos - self.m if self.kind == "cosface" else torch.cos(torch.acos(cos) + self.m)
        onehot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        return F.cross_entropy(self.s * (onehot * target + (1 - onehot) * cos), labels)

def centroid_loss(feat, labels, margin=0.3):
    """Притягивает каждый кадр к центроиду СВОЕЙ машины и отталкивает от чужих.

    Зачем это здесь: центроид усредняет ракурсы. Обычный triplet тянет
    «вид спереди» напрямую к «виду сзади» — а они визуально не похожи,
    и градиент получается противоречивым. Центроид даёт им общую цель:
    оба ракурса тянутся к одной точке, а не друг к другу.

    Центроиды считаются внутри батча, поэтому P x K сэмплер обязателен.
    """
    f = F.normalize(feat)
    uniq = torch.unique(labels)
    cent = torch.stack([f[labels == u].mean(0) for u in uniq])
    cent = F.normalize(cent)
    sim = f @ cent.t()                                  # (batch, машин в батче)
    own = (labels.view(-1, 1) == uniq.view(1, -1))
    pos = sim[own]                                       # сходство со своим центроидом
    neg = sim.masked_fill(own, -float("inf")).max(dim=1).values   # с ближайшим чужим
    return F.relu(neg - pos + margin).mean()


def batch_hard_triplet(feat, labels, margin):
    f = F.normalize(feat)
    dist = torch.cdist(f, f, p=2)
    same = labels.view(-1, 1) == labels.view(1, -1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=f.device)
    pos = dist.masked_fill(~same | eye, -float("inf")).max(dim=1).values
    neg = dist.masked_fill(same, float("inf")).min(dim=1).values
    return F.relu(pos - neg + margin).mean()

class FullModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = timm.create_model(cfg["backbone"], pretrained=True, num_classes=0)
        if cfg["grad_checkpointing"]:
            try:
                self.encoder.set_grad_checkpointing(True)
                print("gradient checkpointing включён")
            except Exception as e:
                print("gradient checkpointing не поддерживается этим бэкбоном:", e)
        self.head = ReidHead(self.encoder.num_features, cfg["hidden"], cfg["out_dim"], cfg["dropout"])
    def forward(self, x):
        return self.head(self.encoder(x))

torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
device = torch.device("cuda")
model = FullModel(CFG).to(device)
margin_head = MarginHead(CFG["out_dim"], len(train_vids), CFG["loss"],
                         CFG["scale"], CFG["margin"]).to(device)
print(f"бэкбон {CFG['backbone']}: "
      f"{sum(p.numel() for p in model.encoder.parameters())/1e6:.0f}M параметров")'''))

    cells.append(md("""### Сэмплер и загрузчик

Трудным батчам нужен один прогон замороженным бэкбоном, поэтому загрузчик
создаётся здесь, а не раньше."""))
    cells.append(code('''neighbors = None
if CFG["hard_batches"]:
    t = time.time()
    neighbors = build_vehicle_neighbors(model, train_rows)
    print(f"центроиды машин посчитаны за {time.time()-t:.0f} с, "
          f"соседей на машину: {len(next(iter(neighbors.values())))}")

sampler = PKBatchSampler(train_rows, CFG["P"], CFG["K"], CFG["seed"],
                         cross_camera=CFG["cross_camera_positives"],
                         neighbors=neighbors)
train_dl = DataLoader(CropDataset(train_rows, True), batch_sampler=sampler,
                      num_workers=CFG["workers"], pin_memory=True)
steps_per_epoch = len(sampler)
print(f"{steps_per_epoch} шагов на эпоху, батч {CFG['P'] * CFG['K']}")

# показать, что кросс-камерные позитивы действительно работают
probe = next(iter(sampler))
cams = defaultdict(set)
for i in probe:
    r = train_rows[i]
    cams[r["vehicle_id"]].add(r["camera_id"])
avg_cams = sum(len(c) for c in cams.values()) / len(cams)
print(f"в пробном батче {len(cams)} машин, в среднем {avg_cams:.2f} разных камер на машину "
      f"(из K={CFG['K']} кадров)")'''))

    cells.append(md("## 7. Оценка на валидации\\n\\nКросс-камерная. Эмбеддинг = среднее оригинала и отражения, затем L2."))
    cells.append(code('''@torch.no_grad()
def embed_rows(rows, bs=64):
    model.eval()
    ds = CropDataset(rows, train=False)
    out = []
    for i in range(0, len(rows), bs):
        x = torch.stack([ds[j][0] for j in range(i, min(i+bs, len(rows)))]).to(device)
        with torch.autocast("cuda", enabled=CFG["amp"]):
            a = model(x)[1]
            b = model(torch.flip(x, dims=[3]))[1]
        out.append(F.normalize((a.float() + b.float()) / 2, dim=1).cpu())
    model.train()
    return torch.cat(out).numpy()

def evaluate_val():
    q, g = embed_rows(val_qry), embed_rows(val_gal)
    g_vid = np.array([r["vehicle_id"] for r in val_gal])
    g_cam = np.array([r["camera_id"]  for r in val_gal])
    sim = q @ g.T

    aps, r1, r5, used = [], 0, 0, 0
    for i, qr in enumerate(val_qry):
        keep = g_cam != qr["camera_id"]
        if not keep.any(): continue
        rel = g_vid[keep] == qr["vehicle_id"]
        if not rel.any(): continue
        used += 1
        s = sim[i][keep]; order = np.argsort(-s); rs = rel[order]
        hits = np.flatnonzero(rs)
        aps.append(float(((np.arange(len(hits)) + 1) / (hits + 1)).mean()))
        r1 += int(rs[0]); r5 += int(rs[:5].any())

    res = {"mAP": round(float(np.mean(aps))*100, 2), "rank1": round(r1/used*100, 2),
           "rank5": round(r5/used*100, 2), "queries_used": used}

    # режим отказа: половине машин убираем их саму из галереи
    rng = np.random.default_rng(CFG["seed"])
    vids = sorted({r["vehicle_id"] for r in val_qry})
    unknown = set(rng.choice(vids, size=len(vids)//2, replace=False).tolist())
    pos, neg = [], []
    for i, qr in enumerate(val_qry):
        v = qr["vehicle_id"]
        keep = g_cam != qr["camera_id"]
        if v in unknown: keep = keep & (g_vid != v)
        if not keep.any(): continue
        (neg if v in unknown else pos).append(float(sim[i][keep].max()))
    pos, neg = np.array(pos), np.array(neg)
    allv = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(allv); ranks = np.empty(len(allv)); ranks[o] = np.arange(1, len(allv)+1)
    res["refusal_auc"] = round(float((ranks[lab==1].sum() - len(pos)*(len(pos)+1)/2)
                                     / (len(pos)*len(neg))), 3)
    best_j, best = -9, {}
    for t in np.arange(allv.min(), allv.max(), 0.002):
        tp, fp = int((pos>=t).sum()), int((neg>=t).sum())
        fn, tn = int((pos<t).sum()),  int((neg<t).sum())
        if tp == 0: continue
        pr, rc, tnr = tp/(tp+fp), tp/(tp+fn), tn/(tn+fp)
        j = rc + tnr - 1
        if j > best_j:
            best_j = j
            best = {"threshold": round(float(t),3), "f1": round(2*pr*rc/(pr+rc),3),
                    "recall": round(rc,3), "TNR": round(tnr,3)}
    res.update({"bal_" + k: v for k, v in best.items()})
    return res'''))

    cells.append(md('''## 8. Обучение с возобновлением

Чекпоинт пишется каждые `save_every` эпох. Если сессия оборвётся —
просто запустите ноутбук заново: он найдёт `last.pt` и продолжит
с той эпохи, на которой остановился, а не с нуля.'''))
    cells.append(code('''opt = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": CFG["lr_backbone"]},
    {"params": model.head.parameters(),    "lr": CFG["lr_head"]},
    {"params": margin_head.parameters(),   "lr": CFG["lr_head"]},
], weight_decay=CFG["weight_decay"])
scaler = torch.amp.GradScaler("cuda", enabled=CFG["amp"])

LAST = WORK / f"last_{PRESET}.pt"
BEST = WORK / f"best_{PRESET}.pt"
start_epoch, best_map, history = 0, -1.0, []

if LAST.exists():
    ck = torch.load(LAST, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); margin_head.load_state_dict(ck["margin"])
    opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    start_epoch = ck["epoch"]; best_map = ck.get("best_map", -1.0)
    history = ck.get("history", [])
    print(f"ВОЗОБНОВЛЯЮ с эпохи {start_epoch}, лучший mAP пока {best_map}")
else:
    print("начинаю с нуля")
    print("до обучения:", evaluate_val())

total_steps  = CFG["epochs"] * steps_per_epoch
warmup_steps = CFG["warmup_epochs"] * steps_per_epoch
base_lrs = [CFG["lr_backbone"], CFG["lr_head"], CFG["lr_head"]]
step = start_epoch * steps_per_epoch
t_start = time.time()
log = open(WORK / f"log_{PRESET}.txt", "a")

for ep in range(start_epoch, CFG["epochs"]):
    tot_c = tot_t = 0.0; nb = 0
    for x, y in train_dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        sc = ((step+1)/warmup_steps if step < warmup_steps else
              0.5*(1+math.cos(math.pi*(step-warmup_steps)/max(1, total_steps-warmup_steps))))
        for gp, base in zip(opt.param_groups, base_lrs):
            gp["lr"] = base * sc
        with torch.autocast("cuda", enabled=CFG["amp"]):
            feat, feat_bn = model(x)
            l_cls = margin_head(feat_bn, y)
            l_tri = batch_hard_triplet(feat.float(), y, CFG["tri_margin"])
            loss = l_cls + CFG["tri_weight"] * l_tri
            if CFG["centroid_weight"]:
                l_cen = centroid_loss(feat.float(), y, CFG["tri_margin"])
                loss = loss + CFG["centroid_weight"] * l_cen
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot_c += l_cls.item(); tot_t += l_tri.item(); nb += 1; step += 1

        # после прогрева говорим, сколько всё это займёт — чтобы не гадать
        if step == 20:
            sec_step = (time.time() - t_start) / 20
            eta_epoch = sec_step * steps_per_epoch / 60
            eta_all = eta_epoch * CFG["epochs"]
            print(f"  темп: {sec_step:.2f} с/шаг -> эпоха ~{eta_epoch:.1f} мин, "
                  f"все {CFG['epochs']} эпох ~{eta_all/60:.1f} ч "
                  f"(+ оценки на val каждые {CFG['eval_every']} эпох)", flush=True)
            if torch.cuda.is_available():
                print(f"  видеопамять занято: "
                      f"{torch.cuda.max_memory_allocated()/1024**3:.1f} ГБ", flush=True)

    msg = (f"эпоха {ep+1}/{CFG['epochs']}  cls {tot_c/nb:.3f}  tri {tot_t/nb:.3f}  "
           f"{(time.time()-t_start)/60:.1f} мин")

    if (ep+1) % CFG["eval_every"] == 0 or ep == CFG["epochs"]-1:
        m = evaluate_val(); m["epoch"] = ep+1; history.append(m)
        msg += f"  | mAP {m['mAP']}  R1 {m['rank1']}  AUC {m['refusal_auc']}"
        if m["mAP"] > best_map:
            best_map = m["mAP"]
            torch.save({"model": model.state_dict(), "cfg": CFG, "metrics": m}, BEST)
            msg += "  <- лучший"

    if (ep+1) % CFG["save_every"] == 0 or ep == CFG["epochs"]-1:
        torch.save({"model": model.state_dict(), "margin": margin_head.state_dict(),
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep+1, "best_map": best_map, "history": history,
                    "cfg": CFG}, LAST)

    print(msg, flush=True); log.write(msg + "\\n"); log.flush()

log.close()
print(f"\\nготово, лучший mAP {best_map}")'''))

    cells.append(md("## 9. Результаты"))
    cells.append(code('''final = {"config": CFG, "history": history, "best_mAP": best_map,
         "n_train_images": len(train_rows), "n_train_ids": len(train_vids)}
(WORK / f"metrics_{PRESET}.json").write_text(json.dumps(final, indent=2, ensure_ascii=False))
print(json.dumps(history[-1] if history else {}, indent=2, ensure_ascii=False))
print()
for f in sorted(WORK.glob("*")):
    if f.is_file():
        print(f"{f.stat().st_size/1024/1024:8.1f} МБ  {f.name}")
print(f"\\nЗабрать: best_{PRESET}.pt и metrics_{PRESET}.json")'''))

    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"},
                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


for platform, fname in [("kaggle", "train_reid.ipynb"), ("colab", "train_reid_colab.ipynb")]:
    nb = build(platform)
    p = HERE / fname
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"{fname}: {len(nb['cells'])} ячеек, {p.stat().st_size} байт")
