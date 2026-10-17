import csv, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent; TTA = HERE.parent / "tta"
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
    s = g @ g.T; np.fill_diagonal(s, -np.inf); return np.sort(s, axis=1)[:, -k:].mean(axis=1)
split = [r for r in rows(D / "val_split.csv") if r["image_id"] not in REMOVED]
seen, hold_g, hold_q = set(), [], []
for r in split:
    key = (r["vehicle_id"], r["camera_id"]); (hold_q if key in seen else hold_g).append(r); seen.add(key)
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
def top1_correct(s, gids):
    """top-1 after removing junk (same vehicle & same camera) from FULL ranking."""
    gd = gdf(gids); gv = gd.vehicle_id.values; gc = gd.camera_id.values
    out = []
    for i, q in enumerate(qids):
        qv, qc = qdf.loc[q, "vehicle_id"], qdf.loc[q, "camera_id"]
        junk = (gv == qv) & (gc == qc)
        ss = np.where(junk, -np.inf, s[i]); j = int(np.argmax(ss))
        out.append(gv[j] == qv)
    return np.array(out)
def load(p):
    z = np.load(p); e = {i: v for i, v in zip(z["ids"].astype(str), z["emb"])}
    return [norm(np.stack([e[i] for i in L])) for L in (qids, hg, tg)]
M = {"A10": load(TTA / "tta_emb_pad0.10_flip0.npz"), "old": load(HERE / "emb_old_student.npz")}
if (HERE / "emb_siglip2_scratch30_ep28.npz").exists(): M["scratch30"] = load(HERE / "emb_siglip2_scratch30_ep28.npz")
def combo(kind, a, b, w=0.5):
    if kind == "avg": return [norm(w * x + (1 - w) * y) for x, y in zip(a, b)]
    if kind == "cat": return [norm(np.hstack([x, y])) for x, y in zip(a, b)]
cases = {"a": lambda E: (E[1], hg), "b": lambda E: (np.vstack([E[1], E[2]]), hg + tg)}
def run(name, E, res, store):
    for c, f in cases.items():
        G, ids = f(E); raw = E[0] @ G.T; h = hub(G)
        for post, s in (("raw", raw), ("hub0.5", raw - 0.5 * h[None])):
            m, r1, r5 = score(s, ids)
            res.append(dict(model=name, case=c, post=post, mAP10=m, R1=r1, R5=r5))
            store[(name, c, post)] = top1_correct(s, ids)
res, store = [], {}
for k, E in M.items(): run(f"solo_{k}", E, res, store)
for other in [k for k in M if k != "A10"]:
    run(f"A10+{other} avg", combo("avg", M["A10"], M[other]), res, store)
    run(f"A10+{other} concat", combo("cat", M["A10"], M[other]), res, store)
    for w in (0.5, 0.6, 0.7, 0.8):
        run(f"A10+{other} w{w}", combo("avg", M["A10"], M[other], w), res, store)
df = pd.DataFrame(res); df.to_csv(HERE / "ens_results.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
for other in ["old"] + (["scratch30"] if "scratch30" in M else []):
    sub = df[df.model.str.contains(other) | (df.model == "solo_A10")]
    print(f"\n===== A10 vs {other}")
    print(sub.pivot_table(index="model", columns=["case", "post"], values=["mAP10", "R1", "R5"], sort=False).round(4).to_string())
# error correlation
print("\n===== error correlation (top-1 after junk removal)")
for other in [k for k in M if k != "A10"]:
    for c in "ab":
        for post in ("raw", "hub0.5"):
            a = store[("solo_A10", c, post)]; o = store[(f"solo_{other}", c, post)]
            print(f"{other} case({c}) {post}: n={len(a)} A10 acc {a.mean():.4f} other acc {o.mean():.4f} | "
                  f"A10 wrong & other right {(~a & o).sum()} ({(~a & o).mean():.3f}) | A10 right & other wrong {(a & ~o).sum()} ({(a & ~o).mean():.3f}) | "
                  f"both wrong {(~a & ~o).sum()} | both right {(a & o).sum()} | oracle {(a | o).mean():.4f}")
for k, E in M.items():
    if k != "A10":
        print("mean cos(A10,%s) holdout q: %.3f" % (k, (M["A10"][0] * E[0]).sum(1).mean()))
