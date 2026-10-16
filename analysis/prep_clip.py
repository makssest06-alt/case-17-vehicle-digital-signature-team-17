#!/usr/bin/env python3
"""Convert OpenAI CLIP ViT-B/16 ([OI] format) to a timm ViT state_dict."""
import re
import torch

SRC = r"C:\Users\Colorful\AppData\Local\Temp\opencode\clip_vitb16.bin"
OUT = r"D:\bm\improve_models\clip_vitb16_init.pt"
NAME = "vit_base_patch16_clip_224.openai"


def convert(sd):
    out = {}
    for k, v in sd.items():
        if not k.startswith("visual."):
            continue
        k = k[len("visual."):]
        m = re.match(r"transformer\.resblocks\.(\d+)\.(.+)", k)
        if m:
            i, rest = m.group(1), m.group(2)
            mapping = {
                "ln_1.weight": f"blocks.{i}.norm1.weight",
                "ln_1.bias": f"blocks.{i}.norm1.bias",
                "ln_2.weight": f"blocks.{i}.norm2.weight",
                "ln_2.bias": f"blocks.{i}.norm2.bias",
                "attn.in_proj_weight": f"blocks.{i}.attn.qkv.weight",
                "attn.in_proj_bias": f"blocks.{i}.attn.qkv.bias",
                "attn.out_proj.weight": f"blocks.{i}.attn.proj.weight",
                "attn.out_proj.bias": f"blocks.{i}.attn.proj.bias",
                "mlp.c_fc.weight": f"blocks.{i}.mlp.fc1.weight",
                "mlp.c_fc.bias": f"blocks.{i}.mlp.fc1.bias",
                "mlp.c_proj.weight": f"blocks.{i}.mlp.fc2.weight",
                "mlp.c_proj.bias": f"blocks.{i}.mlp.fc2.bias",
            }
            if rest in mapping:
                out[mapping[rest]] = v
            continue
        head = {
            "class_embedding": "cls_token",
            "positional_embedding": "pos_embed",
            "conv1.weight": "patch_embed.proj.weight",
            "ln_pre.weight": "norm_pre.weight",
            "ln_pre.bias": "norm_pre.bias",
            "ln_post.weight": "norm.weight",
            "ln_post.bias": "norm.bias",
        }
        if k in head:
            out[head[k]] = v
    # reshape to timm layout
    if "cls_token" in out:
        out["cls_token"] = out["cls_token"].reshape(1, 1, -1)
    if "pos_embed" in out:
        out["pos_embed"] = out["pos_embed"].reshape(1, *out["pos_embed"].shape)
    return out


def main():
    import timm

    sd = torch.load(SRC, map_location="cpu", weights_only=True)
    conv = convert(sd)
    model = timm.create_model(NAME, pretrained=False, num_classes=0, img_size=224)
    msd = model.state_dict()
    missing = [k for k in msd if k not in conv]
    print("converted", len(conv), "model keys", len(msd))
    print("missing after convert:", len(missing), missing[:10])
    model.load_state_dict(conv, strict=True)
    print("strict load OK")

    enc_sd = {f"encoder.{k}": v for k, v in model.state_dict().items()}
    ck = {"model": enc_sd, "margin": {"weight": torch.zeros(1, 512)},
          "args": {"backbone": "clip_vitb16", "tf_mode": "squash", "input_size": 224}}
    torch.save(ck, OUT)
    print("saved", OUT, "tensors", len(enc_sd))


if __name__ == "__main__":
    main()
