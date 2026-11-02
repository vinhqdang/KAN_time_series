import sys, os
R = "/home/user/KAN_time_series"; sys.path[:0] = [R, R + "/scripts", R + "/research/spire"]
import numpy as np, torch
import bench as B
from bench import gen_full, zscore
from docalc import make_learned, topo_order, pick_pairs
import scale_instantaneous as SI
from basil_s import fit_basil_s
X, A, f, sd, sim = gen_full(500, 100, 1001, "unif", "m1"); Xz = zscore(X); d = 100
Xt = torch.tensor(Xz); mk = SI.screen_mask(Xz, 15, 1001); mm = SI.fit(Xt, 1001, SI.lg_rule(d, 0.2), ep=400, mask=mk); S0 = SI.chunked_importance(mm, Xt)
inw = S0.sum(0).copy(); order0 = []
for _ in range(d):
    i = int(np.argmin(inw)); order0.append(i); inw[i] = np.inf; inw -= np.where(np.isinf(inw), 0, S0[i, :])
o = fit_basil_s(Xz, 1001, m=20, iters=200 * d, chains=2, burn_frac=0.3, select=True, cand_score=S0, order0=order0)
coef = o["coef"](); W = np.linalg.norm(coef, axis=2).T; order = topo_order(W); pos = np.empty(d, int); pos[order] = np.arange(d)
tr = list(zip(*np.nonzero(A))); viol = sum(pos[c] > pos[e] for c, e in tr); print("true edges", len(tr), "against learned order:", viol)
big = (W > 0)
print("learned edges with positive weight:", int(big.sum()), "max |coef|:", np.abs(coef).max(), "max W:", W.max())
do_curve = make_learned(coef, Xz); rng = np.random.RandomState(1001 + 4242); dp, ip = pick_pairs(A, rng)
rows = []
for c, e in ip:
    vals = np.quantile(Xz[:, c], np.linspace(0.05, 0.95, 15)); tau = np.array([sim(1000, c, v)[:, e].mean() for v in vals]); tau -= tau.mean()
    if tau.std() < 0.05: continue
    hat = do_curve(c, vals)[:, :, e].mean(1); hat -= hat.mean(); rows.append((np.linalg.norm(hat - tau) / np.linalg.norm(tau), np.linalg.norm(tau), np.linalg.norm(hat), c, e))
rows.sort(reverse=True)
for r in rows[:4]: print("nrmse %.1f |tau| %.2f |hat| %.2f pair %d->%d  order pos c,e = %d,%d" % (r[0], r[1], r[2], r[3], r[4], pos[r[3]], pos[r[4]]))
c, e = int(rows[0][3]), int(rows[0][4])
vals = np.quantile(Xz[:, c], np.linspace(0.05, 0.95, 15)); full = do_curve(c, vals)
print("range of x values after propagation per node (max abs over grid/rows) for top 5 nodes:", np.sort(np.abs(full).max((0, 1)))[-5:].round(1), "true data max abs:", np.abs(Xz).max().round(1))
