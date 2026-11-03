"""Generate the data for Figure 3 (recovered curves vs truth) on one d=100 dataset.
Uses the replication code (research/replication, read only). Output: data/example_curves.npz.
Run with python3.13 (needs torch, sklearn). About 1-2 minutes on one core.
Usage: python3.13 make_example_curves.py [seed=2000]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
REP = os.path.join(REPO, "research/replication")
sys.path.insert(0, REP)
os.chdir(REP)                      # bench.py resolves its own paths from __file__
import numpy as np, torch
import bench as B
from spade_variants import zscore
from basil_s import fit_basil_s
import scale_instantaneous as SI
torch.set_num_threads(1)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
d, n = 100, 1000
X, A, f, sd, sim = B.gen_full(n, d, seed, "unif", "m1")
Xz = zscore(X); Zs = SI.zscore(X); Xt = torch.tensor(Zs)
mk = SI.screen_mask(Zs, 15, seed)
mm = SI.fit(Xt, seed, SI.lg_rule(d, 0.2), ep=400, mask=mk); S0 = SI.chunked_importance(mm, Xt)
Bm = mm._bases(Xt).detach(); cf = (mm.coef * mm.selfmask).detach()
inw = S0.sum(0).copy(); order0 = []
for _ in range(d):
    i = int(np.argmin(inw)); order0.append(i); inw[i] = np.inf; inw -= np.where(np.isinf(inw), 0, S0[i, :])
o = fit_basil_s(Xz, seed, m=20, iters=200 * d, chains=2, burn_frac=0.3, select=True, cand_score=S0, order0=order0)
edges = list(zip(*np.nonzero(A)))
rows = []
for c, e in edges:
    x = Xz[:, c].astype(float)
    gt = f[(c, e)](x) / sd[e]; gt = gt - gt.mean()
    ss = (Bm[:, c, :] @ cf[e, c]).numpy().astype(float); ss = ss - ss.mean()
    bw = np.asarray(o["mean_curve"](c, e), float); bw = bw - bw.mean()
    lo, hi = o["band"](c, e)
    nr = lambda g: float(np.sqrt(((g - gt) ** 2).sum() / (gt ** 2).sum()))
    rows.append((c, e, x, gt, ss, bw, np.asarray(lo, float), np.asarray(hi, float), nr(ss), nr(bw), float(o["P"][c, e])))
# keep 6 true edges drawn at random (RandomState(0)); no selection on any result
sel = np.random.RandomState(0).choice(len(rows), 6, replace=False)
out = dict(seed=seed, d=d, n=n, nrmse_ssc_all=float(np.mean([r[8] for r in rows])), nrmse_basil_all=float(np.mean([r[9] for r in rows])))
for k, i in enumerate(sel):
    c, e, x, gt, ss, bw, lo, hi, n1, n2, P = rows[i]
    j = np.argsort(x)
    for nm, v in dict(x=x, truth=gt, ssc=ss, basilw=bw, lo=lo, hi=hi).items(): out[f"e{k}_{nm}"] = v[j]
    out[f"e{k}_meta"] = np.array([c, e, n1, n2, P])
np.savez_compressed(os.path.join(HERE, "data/example_curves.npz"), **out)
print("saved; mean NRMSE over all true edges: SSC %.3f BASIL-W %.3f" % (out["nrmse_ssc_all"], out["nrmse_basil_all"]))
