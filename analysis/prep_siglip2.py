#!/usr/bin/env python3
"""Build a case17-format checkpoint from downloaded SigLIP 2 weights."""
import torch

SRC = r"C:\Users\Colorful\AppData\Local\Temp\opencode\siglip2.safetensors"
OUT = r"D:\bm\improve_models\siglip2_init.pt"
NAME = "vit_base_patch16_siglip_224.v2_webli"


def main():
    import timm
    from safetensors.torch import load_file

    sd = load_file(SRC)
    model = timm.create_model(NAME, pretrained=False, num_classes=0,
                              img_size=224, dynamic_img_size=True)
    msd = model.state_dict()
    clean = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    missing = [k for k in msd if k not in clean]
    print("ckpt keys", len(sd), "model keys", len(msd), "matched", len(clean))
    print("missing", len(missing), missing[:6])
    model.load_state_dict(clean, strict=False)
    enc_sd = {f"encoder.{k}": v for k, v in model.state_dict().items()}
    ck = {"model": enc_sd, "margin": {"weight": torch.zeros(1, 512)},
          "args": {"backbone": "siglip2_base", "tf_mode": "squash", "input_size": 224}}
    torch.save(ck, OUT)
    print("saved", OUT, "tensors", len(enc_sd))


if __name__ == "__main__":
    main()
