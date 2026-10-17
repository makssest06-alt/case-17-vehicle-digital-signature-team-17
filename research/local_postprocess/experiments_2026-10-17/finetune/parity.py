import numpy as np
r=np.load("../tta/tta_emb_pad0.10_flip0.npz"); a=np.load("emb/a10.npz")
m=dict(zip(r["ids"].astype(str),r["emb"])); n=dict(zip(a["ids"].astype(str),a["emb"]))
assert set(m)==set(n)
c=np.array([float(m[i]@n[i]/np.linalg.norm(m[i])/np.linalg.norm(n[i])) for i in m]); print("A10 parity min cos",c.min(),"n",len(c))
for t in ["ctrl_ep3","ctrl_ep6","hm_ep3","hm_ep6"]:
    b=np.load(f"emb/{t}.npz"); k=dict(zip(b["ids"].astype(str),b["emb"]))
    c=np.array([float(n[i]@k[i]/np.linalg.norm(n[i])/np.linalg.norm(k[i])) for i in n]); print(t,"cos to A10 mean",c.mean(),"min",c.min())
