import torch, hashlib, json, copy
from pathlib import Path
SCR = Path(r"C:\Users\user\AppData\Local\Temp\claude\c--Users-user-PycharmProjects-AIOS-BM\b9767fdc-a44d-4e89-91ed-82d3dff36f71\scratchpad")
P = {"A10": Path(r"C:\Users\user\Downloads\clean_student_a10_ep28.pt"),
     "ctrl": SCR/"ft/train/runs/ctrl_ep6.pt", "hm": SCR/"ft/train/runs/hm_ep6.pt"}
CK = {k: torch.load(v, map_location="cpu", weights_only=False) for k, v in P.items()}
for k, c in CK.items(): print(k, list(c.keys()), len(c["model"]))
SOUPS = {"S1": ["A10", "ctrl"], "S2": ["A10", "hm"], "S3": ["A10", "ctrl", "hm"], "S4": ["ctrl", "hm"]}
base = CK["A10"]
for name, mix in SOUPS.items():
    sd = {}
    for k, v in base["model"].items():
        if v.is_floating_point():
            for m in mix: assert CK[m]["model"][k].shape == v.shape and CK[m]["model"][k].dtype == v.dtype, (m, k)
            acc = sum(CK[m]["model"][k].double() for m in mix) / len(mix)
            sd[k] = acc.to(v.dtype)
        else:
            sd[k] = v.clone()
    for m in mix: assert set(CK[m]["model"]) == set(base["model"])
    args = copy.deepcopy(base["args"]); 
    if not isinstance(args, dict): args = vars(args) if hasattr(args, "__dict__") else args
    args = dict(args) if isinstance(args, dict) else args
    desc = {"mix": mix, "weights": [1/len(mix)]*len(mix), "method": "uniform mean of float tensors; int buffers from A10"}
    if isinstance(args, dict): args["soup"] = desc
    else: args.soup = desc
    out = {"model": sd, "margin": base["margin"], "args": args}
    p = SCR/"soup"/f"{name}.pt"; torch.save(out, p)
    h = hashlib.sha256(p.read_bytes()).hexdigest(); print(name, mix, h, flush=True)
    (SCR/"soup"/f"{name}.sha256").write_text(h)
