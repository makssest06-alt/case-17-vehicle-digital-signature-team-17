"""embed_ft.py with a rectangular squash input (H x W) instead of 224 x 224.
usage: embed_rect.py <ckpt> <out.npz> <H> <W>
Same frames and crop as embed_ft.py (pad 0.10, no flip); only the resize target changes."""
import runpy
import sys
from pathlib import Path

from torchvision import transforms as T

CKPT, OUT, H, W = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
TTA = Path(__file__).resolve().parent.parent / "tta"
sys.path.insert(0, str(TTA / "api"))
from app.models import reid_checkpoint as rc  # noqa: E402

_orig_build_tf = rc.ReidCheckpointModel._build_tf


def _rect_tf(self, timm, backbone):
    tf = _orig_build_tf(self, timm, backbone)
    assert self.tf_mode == "squash"
    return T.Compose([T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                      *tf.transforms[1:]])


rc.ReidCheckpointModel._build_tf = _rect_tf
sys.argv = [str(Path(__file__).with_name("embed_ft.py")), CKPT, OUT]
runpy.run_path(sys.argv[0], run_name="__main__")
