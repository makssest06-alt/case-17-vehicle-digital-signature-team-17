import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
D = Path(r"C:\Users\user\PycharmProjects\AIOS\BM\case17_data")
spec = importlib.util.spec_from_file_location("ev", r"C:\Users\user\Downloads\evaluate.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
REMOVED = {"1b4ab30599ea41178b775b81f5f6265c", "25fe4ed7ba5b433b9ae84795ea909729",
           "4b90daade9334addb76bf838887e1430", "64a9bec33f0e4b21b1f9a0c8f8504741",
           "ba950758834d4f26b7df5dd4832b6797"}
def rows(p):
    with p.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))
def norm(x): return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
def hub(g, k=5):
    s = g @ g.T; np.fill_diagonal(s, -np.inf)
    return np.sort(s, axis=1)[:, -k:].mean(axis=1)

split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
seen, hold_g, hold_q = set(), [], []
for r in split:
    key = (r["vehicle_id"], r["camera_id"])
    (hold_q if key in seen else hold_g).append(r); seen.add(key)
qids = [r["image_id"] for r in hold_q]; hg = [r["image_id"] for r in hold_g]
tg = [r["image_id"] for r in rows(D / "dataset" / "test_gallery.csv")]
assert (len(qids), len(hg), len(tg)) == (546, 361, 750)
train = {r["image_id"]: r for r in rows(D / "train.csv")}
qdf = pd.DataFrame(hold_q).set_index("image_id")
def gdf(ids):
    return pd.DataFrame([{"image_id": i, "vehicle_id": train[i]["vehicle_id"], "camera_id": train[i]["camera_id"]}
                         if i in train else {"image_id": i, "vehicle_id": "x_" + i, "camera_id": "x_" + i}
                         for i in ids]).set_index("image_id")
def score(s, gids):
    order = np.argsort(-s, axis=1, kind="stable")[:, :10]
    ranked = {q: [gids[j] for j in order[i]] for i, q in enumerate(qids)}
    m = ev.ranking_metrics(qdf, gdf(gids), ranked, top_k=10)
    return m["mAP@10"], m["Rank-1"], m["Rank-5"]

def load(pad, flip):
    z = np.load(HERE / f"tta_emb_pad{pad:.2f}_flip{flip}.npz")
    e = {i: v for i, v in zip(z["ids"].astype(str), z["emb"])}
    return norm(np.stack([e[i] for i in qids])), norm(np.stack([e[i] for i in hg])), norm(np.stack([e[i] for i in tg]))

# parity
clean = np.load(D / "a10_clean_expanded.npz"); test = np.load(D / "a10_test.npz")
ce = {str(i): v for i, v in zip(clean["ids"].astype(str), clean["emb"])}
tids = test["ids"].astype(str); te = norm(test["emb"])
assert set(tids[1110:]) == set(tg)
tmap = {i: v for i, v in zip(tids[1110:], te[1110:])}
q, g, t = load(0.10, 0)
cq = norm(np.stack([ce[i] for i in qids])); cg = norm(np.stack([ce[i] for i in hg])); ct = norm(np.stack([tmap[i] for i in tg]))
par = {"holdout_q": (q * cq).sum(1), "holdout_g": (g * cg).sum(1), "test_g": (t * ct).sum(1)}
for k, v in par.items(): print(f"parity {k}: min cos {v.min():.6f} mean {v.mean():.6f} n={len(v)}")
# score cached embeddings too
for name, (Q, G, T) in {"cached": (cq, cg, ct), "fresh": (q, g, t)}.items():
    a = score(Q @ G.T, hg); b = score(Q @ np.vstack([G, T]).T, hg + tg)
    print(f"{name} no-TTA pad0.10: (a) mAP {a[0]:.4f} R1 {a[1]:.4f} R5 {a[2]:.4f} | (b) mAP {b[0]:.4f} R1 {b[1]:.4f} R5 {b[2]:.4f}")

results = []
for pad in (0.10, 0.05, 0.15):
    for flip in (0, 1):
        Q, G, T = load(pad, flip)
        for case, (GG, ids) in {"a_hold361": (G, hg), "b_hold+test1111": (np.vstack([G, T]), hg + tg)}.items():
            raw = Q @ GG.T; h = hub(GG)
            for post, s in (("raw", raw), ("hub0.5", raw - 0.5 * h[None]), ("hub1.0", raw - 1.0 * h[None])):
                m, r1, r5 = score(s, ids)
                results.append(dict(pad=pad, flip=flip, case=case, post=post, mAP10=m, R1=r1, R5=r5))
df = pd.DataFrame(results)
df.to_csv(HERE / "tta_results.csv", index=False)
pd.set_option("display.width", 200)
for case in df.case.unique():
    print("\n==", case)
    print(df[df.case == case].pivot_table(index=["pad", "flip"], columns="post", values=["mAP10", "R1", "R5"]).round(4).to_string())
