#!/usr/bin/env python3
"""Build a case17-format checkpoint from downloaded timm DINOv2 weights."""
import sys
from pathlib import Path

import torch

SRC = r"C:\Users\Colorful\AppData\Local\Temp\opencode\dinov2_small.safetensors"
OUT = r"D:\bm\improve_models\dinov2_vits14_init.pt"
NAME = "vit_small_patch14_dinov2.lvd142m"


def main():
    import timm
    from safetensors.torch import load_file

    sd = load_file(SRC)
    # native resolution for lvd142m is 518; build there, then resample pos_embed
    model = timm.create_model(NAME, pretrained=False, num_classes=0,
                              img_size=518, dynamic_img_size=True)
    model.load_state_dict(sd, strict=True)
    print("loaded strict, pos_embed", tuple(model.pos_embed.shape))

    # resample pos_embed to 224 (16x16 grid) for our squash-224 training
    from timm.layers import resample_abs_pos_embed
    pe = model.pos_embed
    new = resample_abs_pos_embed(pe, new_size=(224 // 14, 224 // 14),
                                 num_prefix_tokens=1, antialias=True)
    model.pos_embed = torch.nn.Parameter(new)
    print("resampled pos_embed", tuple(model.pos_embed.shape))

    enc_sd = {f"encoder.{k}": v for k, v in model.state_dict().items()}
    ck = {
        "model": enc_sd,
        "margin": {"weight": torch.zeros(1, 512)},
        "args": {"backbone": "dinov2_vits14", "tf_mode": "squash",
                 "input_size": 224, "source": "timm/" + NAME},
    }
    torch.save(ck, OUT)
    print("saved", OUT, "tensors", len(enc_sd))


if __name__ == "__main__":
    main()
