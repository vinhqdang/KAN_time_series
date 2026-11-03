"""Interventional-effect error (research/07). Learned SEM = coef tensor [effect, cause, K] over the shared B-spline basis."""
import numpy as np, torch
from src.cdkan.causal_kan import CausalKANInstant


def topo_order(W):
    """source-peeling on strength W[cause, effect]: place first the node with the least in-weight from unplaced nodes."""
    d = W.shape[0]; inw = W.sum(0).astype(float).copy(); order = []
    for _ in range(d):
        i = int(np.argmin(inw)); order.append(i); inw[i] = np.inf; inw -= np.where(np.isinf(inw), 0, W[i, :])
    return order


def make_learned(coef, Xz):
    d = Xz.shape[1]; helper = CausalKANInstant(d, grid_size=8); coef = np.asarray(coef, np.float64)
    W = np.linalg.norm(coef, axis=2).T                                 # [cause, effect]
    order = topo_order(W); pos = np.empty(d, int); pos[order] = np.arange(d)
    par = [[c for c in range(d) if W[c, e] > 0 and pos[c] < pos[e]] for e in range(d)]
    def bases(x): return helper._bases(torch.as_tensor(np.asarray(x, np.float32))).double().numpy()
    def node_pred(x, e):
        if not par[e]: return np.zeros(len(x))
        return np.einsum("njk,jk->n", bases(x[:, par[e]]), coef[e, par[e]])
    resid = np.stack([Xz[:, e] - node_pred(Xz, e) for e in range(d)], 1)
    def do_curve(c, vals, nrows=400):
        rows = np.arange(min(nrows, len(Xz))); out = []
        for v in vals:
            x = Xz[rows].copy(); x[:, c] = v
            for e in order[pos[c] + 1:]: x[:, e] = node_pred(x, e) + resid[rows, e]
            out.append(x)
        return np.stack(out)                                             # [grid, rows, d]
    return do_curve


def pick_pairs(A, rng, k=20):
    d = A.shape[0]; R = (A > 0).astype(int); P = R.copy()
    for _ in range(d):
        P2 = ((P + P @ R) > 0).astype(int)
        if (P2 == P).all(): break
        P = P2
    direct = list(zip(*np.nonzero(A))); ind = [(c, e) for c, e in zip(*np.nonzero(P)) if not A[c, e]]
    pick = lambda L: [L[i] for i in rng.permutation(len(L))[:k]] if L else []
    return pick(direct), pick(ind)


def do_error(do_curve, truth_sim, Xz, A, seed=0, grid=15, nsim_true=1000):
    rng = np.random.RandomState(seed + 4242); dpairs, ipairs = pick_pairs(A, rng); res = {}
    for name, pairs in (("direct", dpairs), ("indirect", ipairs)):
        errs = []; skipped = 0
        for c, e in pairs:
            vals = np.quantile(Xz[:, c], np.linspace(0.05, 0.95, grid))
            tau = np.array([truth_sim(nsim_true, c, v)[:, e].mean() for v in vals]); tau = tau - tau.mean()
            if tau.std() < 0.05: skipped += 1; continue
            hat = do_curve(c, vals)[:, :, e].mean(1); hat = hat - hat.mean()
            errs.append(np.linalg.norm(hat - tau) / np.linalg.norm(tau))
        res[name] = (float(np.mean(errs)) if errs else np.nan, len(errs), skipped)
    return res
