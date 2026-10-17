"""Draft-mode JPEG decode + crop variants (experiment only)."""
import io
from PIL import Image

FACTORS = (8, 4, 2, 1)

def pad_box(W, H, x, y, w, h, pad):
    px, py = int(round(w * pad)), int(round(h * pad))
    return max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)

def choose_f(box, min_side):
    cw, ch = box[2] - box[0], box[3] - box[1]
    s = min(cw, ch)
    for f in FACTORS:
        if s / f >= min_side:
            return f
    return 1

def decode_full(src):
    """Current service path (load_image: verify + reopen + convert)."""
    raw = src
    probe = Image.open(io.BytesIO(raw)); probe.verify()
    return Image.open(io.BytesIO(raw)).convert("RGB")

def load_crop(raw, bbox, pad, min_side):
    """min_side None -> baseline. Returns (crop PIL RGB, f)."""
    x, y, w, h = bbox
    if min_side is None:
        im = decode_full(raw)
        l, t, r, b = pad_box(im.width, im.height, x, y, w, h, pad)
        return im.crop((l, t, r, b)), 1
    probe = Image.open(io.BytesIO(raw)); probe.verify()
    im = Image.open(io.BytesIO(raw))
    W, H = im.size
    box = pad_box(W, H, x, y, w, h, pad)
    f = choose_f(box, min_side)
    if f > 1:
        im.draft("RGB", (W // f, H // f))
    sx, sy = im.width / W, im.height / H
    im = im.convert("RGB")
    l, t, r, b = box
    cb = (int(round(l * sx)), int(round(t * sy)), int(round(r * sx)), int(round(b * sy)))
    return im.crop(cb), f
