"""Candidate extract(batch) implementations for the organizer protocol.
batch = list of (path, (x, y, w, h)); returns np.ndarray (B, 512) float32, L2-normalised.
Research only; imports the service code from SCR/tta/api read-only.
"""
import copy, io, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CKPT = r"C:\Users\user\Downloads\clean_student_a10_ep28.pt"
os.environ.update(REID_WEIGHTS=CKPT, REID_TF_MODE="squash", EMBED_TTA_FLIP="0",
                  EMBED_MODEL="reid:vit_base_patch16_siglip_224.webli",
                  OMP_NUM_THREADS="2", HF_HUB_OFFLINE="1")
sys.dont_write_bytecode = True
SCR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCR / "tta" / "api")); sys.path.insert(0, str(SCR / "decode"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.config import settings
from app.imaging import crop_with_pad, load_image
from app.models.reid_checkpoint import ReidCheckpointModel
from dlib import choose_f, pad_box

object.__setattr__(settings, "tta_flip", False)
PAD = settings.bbox_pad
S = 224
MIN_SIDE = 224
BUCKETS = (1, 2, 4, 8, 16, 32)

_MODEL = None


def get_model() -> ReidCheckpointModel:
    global _MODEL
    if _MODEL is None:
        m = ReidCheckpointModel("vit_base_patch16_siglip_224.webli", CKPT, 224)
        m.load()
        _MODEL = m
    return _MODEL


# ------------------------------------------------------------------ helpers
def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def decode_crop_resize_u8(path, bbox, draft, fused=False):
    """read -> decode (optionally draft) -> crop(10% pad) -> squash resize 224 bicubic -> uint8 HWC."""
    raw = read_bytes(path)
    im = Image.open(io.BytesIO(raw))
    W, H = im.size
    x, y, w, h = bbox
    box = pad_box(W, H, x, y, w, h, PAD)
    sx = sy = 1.0
    if draft:
        f = choose_f(box, MIN_SIDE)
        if f > 1:
            im.draft("RGB", (W // f, H // f))
            sx, sy = im.width / W, im.height / H
    if im.mode != "RGB":
        im = im.convert("RGB")
    else:
        im.load()
    l, t, r, b = box
    cb = (int(round(l * sx)), int(round(t * sy)), int(round(r * sx)), int(round(b * sy)))
    if fused:
        out = im.resize((S, S), Image.BICUBIC, box=cb)
    else:
        out = im.crop(cb).resize((S, S), Image.BICUBIC)
    return np.asarray(out)


# ------------------------------------------------------------------ (a)/(b)
class Baseline:
    """(a) exactly the current service path; (b) with draft decode."""

    def __init__(self, draft=False):
        self.m = get_model()
        self.draft = draft
        self.name = "draft" if draft else "baseline"

    def extract(self, batch):
        from dlib import load_crop
        imgs = []
        for path, (x, y, w, h) in batch:
            raw = read_bytes(path)
            if self.draft:
                im, _ = load_crop(raw, (x, y, w, h), PAD, MIN_SIDE)
            else:
                im = load_image(raw, settings.max_upload_bytes)
                im = crop_with_pad(im, x, y, w, h, PAD)
            imgs.append(im)
        return self.m.embed_batch(imgs)


# ------------------------------------------------------------------ (c)/(d)
class ParallelCPU:
    """Threaded read+decode+crop+resize+ToTensor+Normalize (as the service tf), one stacked fp16 forward,
    pinned float32 buffer + non_blocking copy."""

    def __init__(self, workers=16, draft=False):
        self.m = get_model()
        self.draft = draft
        self.pool = ThreadPoolExecutor(workers)
        self.name = f"par{workers}" + ("_draft" if draft else "")
        self.buf = torch.empty((32, 3, S, S), dtype=torch.float32).pin_memory()

    def _work(self, i, item):
        path, bbox = item
        u8 = decode_crop_resize_u8(path, bbox, self.draft)
        img = Image.fromarray(u8)
        t = self.m._tf.transforms[1](img)   # ToTensor (Resize is a no-op: already 224x224)
        t = self.m._tf.transforms[2](t)     # Normalize
        self.buf[i].copy_(t)

    @torch.inference_mode()
    def extract(self, batch):
        B = len(batch)
        if B > 32:
            return np.concatenate([self.extract(batch[i:i + 32]) for i in range(0, B, 32)])
        if B == 1:
            self._work(0, batch[0])
        else:
            list(self.pool.map(lambda a: self._work(*a), enumerate(batch)))
        x = self.buf[:B].to(self.m.device, non_blocking=True)
        z = self.m._encode(x)
        return F.normalize(z, p=2, dim=1).cpu().numpy().astype("float32")


# ------------------------------------------------------------------ (e)
def precast_linear_conv(model):
    """Cast Linear/Conv2d weights to fp16 once (what autocast does on every call). Norm layers stay fp32."""
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            mod.half()
    return model


class ParallelGPU:
    """Threaded uint8 prep (optionally draft / fused crop+resize), pinned uint8 buffer, GPU-side
    ToTensor/Normalize, fp16 forward, optional: cudnn.benchmark, channels_last, pre-cast weights, CUDA graph."""

    def __init__(self, workers=16, draft=True, fused=False, cudnn_bench=False, channels_last=False,
                 precast=False, graph=False):
        self.m = get_model()
        self.dev = self.m.device
        self.draft, self.fused, self.graph = draft, fused, graph
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self.name = (f"gpu{workers}" + ("_draft" if draft else "") + ("_fused" if fused else "") +
                     ("_cb" if cudnn_bench else "") + ("_cl" if channels_last else "") +
                     ("_pre" if precast else "") + ("_graph" if graph else ""))
        torch.backends.cudnn.benchmark = cudnn_bench
        if precast or channels_last:
            self.model = copy.deepcopy(self.m._model)
            self.head = copy.deepcopy(self.m._head)
            if channels_last:
                self.model = self.model.to(memory_format=torch.channels_last)
            if precast:
                precast_linear_conv(self.model); precast_linear_conv(self.head)
        else:
            self.model, self.head = self.m._model, self.m._head
        self.channels_last = channels_last
        tf = self.m._tf.transforms[2]
        self.mean = torch.tensor(tf.mean, dtype=torch.float32, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(tf.std, dtype=torch.float32, device=self.dev).view(1, 3, 1, 1)
        self.buf = torch.empty((32, S, S, 3), dtype=torch.uint8).pin_memory()
        self.buf_np = self.buf.numpy()
        self.graphs = {}
        if graph:
            self._capture()

    def _fwd(self, u8):  # u8: (B,224,224,3) uint8 on device
        x = u8.permute(0, 3, 1, 2).float().div_(255.0).sub_(self.mean).div_(self.std)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            x = x.contiguous()
        with torch.autocast("cuda", dtype=torch.float16):
            z = self.model(x)
            z = self.head(z)[1].float()
        return F.normalize(z, p=2, dim=1)

    @torch.inference_mode()
    def _capture(self):
        for B in BUCKETS:
            sin = torch.zeros((B, S, S, 3), dtype=torch.uint8, device=self.dev)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._fwd(sin)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sout = self._fwd(sin)
            self.graphs[B] = (g, sin, sout)

    def _work(self, i, item):
        path, bbox = item
        self.buf_np[i] = decode_crop_resize_u8(path, bbox, self.draft, self.fused)

    @torch.inference_mode()
    def extract(self, batch):
        B = len(batch)
        if B > 32:
            return np.concatenate([self.extract(batch[i:i + 32]) for i in range(0, B, 32)])
        if self.pool is None or B == 1:
            for i, it in enumerate(batch):
                self._work(i, it)
        else:
            list(self.pool.map(lambda a: self._work(*a), enumerate(batch)))
        if self.graph:
            bk = next(b for b in BUCKETS if b >= B)
            g, sin, sout = self.graphs[bk]
            sin[:B].copy_(self.buf[:B], non_blocking=True)
            g.replay()
            return sout[:B].cpu().numpy().astype("float32")
        u8 = self.buf[:B].to(self.dev, non_blocking=True)
        return self._fwd(u8).cpu().numpy().astype("float32")


# ------------------------------------------------------------------ (f) nvjpeg experiment
class NvJpeg:
    """GPU JPEG decode (nvjpeg) + GPU crop + GPU bicubic-antialias resize. Not bit-compatible by design."""

    def __init__(self):
        from torchvision.io import ImageReadMode, decode_jpeg
        self.dj, self.mode = decode_jpeg, ImageReadMode.RGB
        self.m = get_model(); self.dev = self.m.device
        self.name = "nvjpeg"
        tf = self.m._tf.transforms[2]
        self.mean = torch.tensor(tf.mean, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(tf.std, device=self.dev).view(1, 3, 1, 1)

    @torch.inference_mode()
    def extract(self, batch):
        datas = [torch.frombuffer(bytearray(read_bytes(p)), dtype=torch.uint8) for p, _ in batch]
        imgs = self.dj(datas, device=self.dev, mode=self.mode)
        outs = []
        for img, (_, (x, y, w, h)) in zip(imgs, batch):
            H, W = img.shape[1:]
            l, t, r, b = pad_box(W, H, x, y, w, h, PAD)
            c = img[:, t:b, l:r].unsqueeze(0).float()
            c = F.interpolate(c, size=(S, S), mode="bicubic", antialias=True, align_corners=False)
            outs.append(c.round_().clamp_(0, 255))
        x = torch.cat(outs).div_(255.0).sub_(self.mean).div_(self.std)
        with torch.autocast("cuda", dtype=torch.float16):
            z = self.m._model(x)
            z = self.m._head(z)[1].float()
        return F.normalize(z, p=2, dim=1).cpu().numpy().astype("float32")


def make(name):
    """Factory by variant name used by bench_q31.py."""
    if name == "baseline": return Baseline(False)
    if name == "draft": return Baseline(True)
    if name.startswith("par"):  # par16, par16_draft
        parts = name[3:].split("_")
        return ParallelCPU(int(parts[0]), draft="draft" in parts)
    if name.startswith("gpu"):  # gpu16_draft_fused_cb_cl_pre_graph
        parts = name[3:].split("_")
        return ParallelGPU(int(parts[0]), draft="draft" in parts, fused="fused" in parts,
                           cudnn_bench="cb" in parts, channels_last="cl" in parts,
                           precast="pre" in parts, graph="graph" in parts)
    if name == "nvjpeg": return NvJpeg()
    raise ValueError(name)
