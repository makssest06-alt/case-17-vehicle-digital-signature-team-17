#!/usr/bin/env python3
"""Build case17-format init checkpoints for small backbones from downloaded weights."""
import torch

SPECS = [
    ("mobilenetv3", "mobilenetv3_large_100.ra_in1k", r"C:\Users\Colorful\AppData\Local\Temp\opencode\mnv3.bin", "bin"),
    ("efficientnet_b0", "efficientnet_b0.ra_in1k", r"C:\Users\Colorful\AppData\Local\Temp\opencode\effb0.safetensors", "sf"),
    ("resnet18", "resnet18.a1_in1k", r"C:\Users\Colorful\AppData\Local\Temp\opencode\r18.safetensors", "sf"),
]


def load_sd(path, kind):
    if kind == "sf":
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            return sd[key]
    return sd


def main():
    import timm
    for key, name, path, kind in SPECS:
        sd = load_sd(path, kind)
        clean = {}
        for k, v in sd.items():
            for pfx in ("module.", "model.", "backbone."):
                if k.startswith(pfx):
                    k = k[len(pfx):]
            clean[k] = v
        model = timm.create_model(name, pretrained=False, num_classes=0)
        msd = model.state_dict()
        load = {k: clean[k] for k in msd if k in clean and clean[k].shape == msd[k].shape}
        miss = [k for k in msd if k not in load]
        model.load_state_dict(load, strict=False)
        enc = {f"encoder.{k}": v for k, v in model.state_dict().items()}
        out = rf"D:\bm\improve_models\{key}_init.pt"
        torch.save({"model": enc, "margin": {"weight": torch.zeros(1, 512)},
                    "args": {"backbone": key, "tf_mode": "squash", "input_size": 224}}, out)
        print(f"{key}: matched {len(load)}/{len(msd)} (missing {len(miss)}) -> {out}")


if __name__ == "__main__":
    main()
