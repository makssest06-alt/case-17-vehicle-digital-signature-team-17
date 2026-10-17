import sys, time, dataclasses, torch
sys.path.insert(0, r"C:\Users\user\AppData\Local\Temp\claude\c--Users-user-PycharmProjects-AIOS-BM\b9767fdc-a44d-4e89-91ed-82d3dff36f71\scratchpad\ft\train")
import backbones
from train_distill import FullModel
ck = torch.load(r"C:\Users\user\Downloads\clean_student_a10_ep28.pt", map_location="cpu", weights_only=False)
cfg = dict(ck.get("args") or ck.get("cfg") or {})
spec = dataclasses.replace(backbones.get(str(cfg.get("backbone", "siglip2_base"))), input_size=224)
_, _, enc = backbones.build_encoder(spec, pretrained=False, input_size=224)
m = FullModel(enc, spec.dim, 1024, 512, 0.3); m.load_state_dict(ck["model"]); m.cuda().eval().half()
res = {}
for hw in [(224, 224), (224, 320), (224, 224), (224, 320)]:
    x = torch.randn(32, 3, *hw, device="cuda", dtype=torch.half)
    with torch.no_grad():
        for _ in range(20): m(x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50): m(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / 50 * 1000
    print(hw, f"{ms:.2f} ms/batch32", flush=True); res.setdefault(hw, []).append(ms)
a = min(res[(224, 224)]); b = min(res[(224, 320)])
print(f"ratio rect/square = {b/a:.3f}  (square {a:.2f} ms, rect {b:.2f} ms; tokens 196 vs 280)")
