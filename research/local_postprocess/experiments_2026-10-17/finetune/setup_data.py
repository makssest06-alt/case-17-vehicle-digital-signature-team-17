import csv, hashlib, numpy as np, torch
from pathlib import Path
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
IM = D/"dataset"/"images"
def rows(p):
    with open(p, newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))
tr = rows(D/"train.csv"); vs = rows(D/"val_split.csv")
val_v = {r["vehicle_id"] for r in vs}
print("train", len(tr), "val rows", len(vs), "val vehicles", len(val_v))
sha = lambda i: hashlib.sha256((IM/f"{i}.jpg").read_bytes()).hexdigest()
val_sha = {sha(r["image_id"]) for r in vs}
cand = [r for r in tr if r["vehicle_id"] not in val_v]
print("after vehicle filter", len(cand))
removed = [r["image_id"] for r in cand if sha(r["image_id"]) in val_sha]
print("removed byte-identical:", removed)
clean = [r for r in cand if r["image_id"] not in set(removed)]
print("clean rows", len(clean), "vehicles", len({r["vehicle_id"] for r in clean}))
with open("clean_train_rows.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["image_id","x","y","w","h","vehicle_id","camera_id"]); w.writeheader()
    for r in clean: w.writerow({k:r[k] for k in w.fieldnames})
z = np.load(D/"a10_clean_expanded.npz"); m = dict(zip(z["ids"].astype(str), z["emb"]))
missing = [r["image_id"] for r in clean if r["image_id"] not in m]; assert not missing, missing[:5]
a = np.stack([m[r["image_id"]] for r in clean]).astype("float32")
a /= np.linalg.norm(a,axis=1,keepdims=True)
np.save("a10_anchor.npy", a); print("anchor", a.shape, a.dtype)
ck = torch.load(r"C:\Users\user\Downloads\clean_student_a10_ep28.pt", map_location="cpu", weights_only=False)
print("ckpt keys", list(ck.keys())); print("margin" in ck)
print("args", ck.get("args") or ck.get("cfg"))
if "margin" in ck: print({k:tuple(v.shape) for k,v in ck["margin"].items()})
