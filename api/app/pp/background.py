"""Признак фона кадра для штрафа за ту же камеру — без второго прогона сети.

Кадр, который сервис и так декодирует (imaging.load_image), ужимается до 128×72
(PIL reduce — среднее по блокам; для размеров не кратных 128×72 — BOX), машина
(bbox + 10 %, как у кропа) выбрасывается маской с запасом в пиксель. Части:
  grad64 / grad32  модуль градиента размытого серого кадра 64×36 / 32×18;
  tiny             цвет 32×18, каждый канал z-нормирован по пикселям фона;
  hist             гистограммы RGB 4×4×4 в зонах 4×4 по пикселям фона, корень + L2.
Сервис берёт SERVICE_PARTS = градиенты 64×36 + гистограммы (выбор и замер —
research/honest_eval_results/cheap_bg_a10.md): «та же камера» с точностью 0.92 и
полнотой 0.30 на парах обучающих машин, 1.1 мс CPU на запрос.

Только numpy и Pillow. research/cheap_bg.py импортирует отсюда: исследование
мерит этот же код.
"""
import numpy as np
from PIL import Image

from .dba import l2

PAD = 0.10                              # запас вокруг bbox — как BBOX_PAD кропа, признак калибровался с ним

SMALL = (128, 72)                       # 1920×1080 / 15
GRAY = np.array([0.299, 0.587, 0.114], "float32")
ZONES, BINS = 4, 4
PARTS = {"grad64": 64 * 36, "grad32": 32 * 18, "tiny": 32 * 18 * 3, "hist": ZONES * ZONES * BINS ** 3}


def small_frame(im: Image.Image, box: tuple) -> tuple:
    """Кадр 128×72 (float32, RGB) и маска фона: False — машина с запасом."""
    f = max(1, min(im.width // SMALL[0], im.height // SMALL[1]))
    s = im.reduce(f) if f > 1 else im
    if s.size != SMALL:                       # BOX: пиксель — среднее своего прямоугольника, маска с запасом его накрывает
        s = s.resize(SMALL, Image.BOX)
    a = np.asarray(s, dtype="float32")
    x, y, w, h = box
    sx, sy = SMALL[0] / im.width, SMALL[1] / im.height
    x0, x1 = int(np.floor((x - PAD * w) * sx)) - 1, int(np.ceil((x + (1 + PAD) * w) * sx)) + 1
    y0, y1 = int(np.floor((y - PAD * h) * sy)) - 1, int(np.ceil((y + (1 + PAD) * h) * sy)) + 1
    bg = np.ones(SMALL[::-1], bool)
    bg[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = False
    return a, bg


def pool(x: np.ndarray, f: int) -> np.ndarray:
    h, w = x.shape[0] // f * f, x.shape[1] // f * f
    return x[:h, :w].reshape(h // f, f, w // f, f, *x.shape[2:]).mean((1, 3))


def blur121(g: np.ndarray) -> np.ndarray:
    p = np.pad(g, 1, mode="edge")
    p = (p[:-2] + 2 * p[1:-1] + p[2:]) / 4
    return (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]) / 4


def erode(m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(m)
    out[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return out


def grad_part(a: np.ndarray, bg: np.ndarray, f: int) -> np.ndarray:
    g = a @ GRAY
    g = np.where(bg, g, g[bg].mean() if bg.any() else 0.0)       # машина — средним фона, не своими пикселями
    p = pool(blur121(g), f)
    gx, gy = np.zeros_like(p), np.zeros_like(p)
    gx[:, 1:-1] = p[:, 2:] - p[:, :-2]
    gy[1:-1] = p[2:] - p[:-2]
    ok = erode(pool(bg.astype("float32"), f) == 1.0)
    return l2(np.where(ok, np.sqrt(gx ** 2 + gy ** 2), 0.0).ravel())


def tiny_part(a: np.ndarray, bg: np.ndarray) -> np.ndarray:
    p = pool(a, 4)
    ok = pool(bg.astype("float32"), 4) == 1.0
    mu, sd = (p[ok].mean(0), p[ok].std(0) + 1e-6) if ok.any() else (0.0, 1.0)
    return l2(np.where(ok[..., None], (p - mu) / sd, 0.0).ravel())


def hist_part(a: np.ndarray, bg: np.ndarray) -> np.ndarray:
    q = np.minimum(a.astype(int) // (256 // BINS), BINS - 1)
    code = (q[..., 0] * BINS + q[..., 1]) * BINS + q[..., 2]
    H, W = bg.shape
    zone = (np.arange(H) * ZONES // H)[:, None] * ZONES + (np.arange(W) * ZONES // W)[None, :]
    h = np.bincount((zone * BINS ** 3 + code)[bg], minlength=PARTS["hist"]).reshape(ZONES * ZONES, -1)
    h = h / np.maximum(h.sum(1, keepdims=True), 1)
    return l2(np.sqrt(h).ravel().astype("float32"))


def describe(im: Image.Image, box: tuple, parts=tuple(PARTS)) -> dict:
    a, bg = small_frame(im, box)
    fn = {"grad64": lambda: grad_part(a, bg, 2), "grad32": lambda: grad_part(a, bg, 4),
          "tiny": lambda: tiny_part(a, bg), "hist": lambda: hist_part(a, bg)}
    return {p: fn[p]().astype("float32") for p in parts}


def combine(mats: list) -> np.ndarray:
    """Склейка частей (N, d_i): каждая L2, затем вместе L2 — веса частей равны."""
    return l2(np.concatenate([l2(m) for m in mats], axis=1))


SERVICE_PARTS = ("grad64", "hist")
BG_DIM = sum(PARTS[p] for p in SERVICE_PARTS)


def service_descriptor(im: Image.Image, box: tuple) -> np.ndarray:
    """Признак фона, который хранит и сравнивает сервис: (BG_DIM,), L2."""
    d = describe(im, box, SERVICE_PARTS)
    return combine([d[p][None] for p in SERVICE_PARTS])[0]
