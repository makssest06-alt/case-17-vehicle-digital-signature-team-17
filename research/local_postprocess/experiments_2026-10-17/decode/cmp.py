import json, numpy as np
b = np.load("emb/full.npz"); ref = np.load("../ft/emb/a10.npz")
print("full vs existing a10.npz: ids equal", (b["ids"]==ref["ids"]).all(), "min cos", (b["emb"]*ref["emb"]).sum(1).min())
n=546+361
for v in ("d224","d160"):
    z=np.load(f"emb/{v}.npz"); assert (z["ids"]==b["ids"]).all()
    c=(z["emb"]*b["emb"]).sum(1)/np.linalg.norm(z["emb"],axis=1)/np.linalg.norm(b["emb"],axis=1)
    for nm,s in (("all",slice(None)),("holdout q",slice(0,546)),("holdout g",slice(546,907)),("test gal",slice(907,None))):
        x=c[s]; print(v,nm,f"mean {x.mean():.4f} min {x.min():.4f} p1 {np.percentile(x,1):.4f} p5 {np.percentile(x,5):.4f}")
base=json.load(open("../ft/res_a10.json"))
for tag in ("full","d224","d160"):
    r=json.load(open(f"res_{tag}.json"))
    for g in r:
        for vn in r[g]:
            for k in ("mAP@10","R1","R5","F1","TNR","PR-AUC"):
                pass
print()
ks=("mAP@10","R1","R5","F1","TNR","PR-AUC")
for g in base:
    for vn in base[g]:
        row=[f"{g[:4]:4s} {vn:22s}"]
        row.append("base "+" ".join(f"{base[g][vn][k]:.3f}" for k in ks))
        for tag in ("full","d224","d160"):
            r=json.load(open(f"res_{tag}.json"))
            row.append(f"{tag} d: "+" ".join(f"{r[g][vn][k]-base[g][vn][k]:+.3f}" for k in ks))
        print(" | ".join(row))
