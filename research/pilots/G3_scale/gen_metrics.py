"""Generators and metrics for the G3 pilot."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def gen_std(n, d, seed, noise=0.5):
    """Sortability-controlled ANM: internally standardized SCM (Ormaniec et al. 2025 idea). Each non-root node is
    x_e = standardize(sum_c f_ce(x_c)) + N(0, noise^2); roots are N(0,1). The signal part has unit variance at EVERY node,
    so marginal variance does not grow along the order (no varsortability); Gaussian noise (SCORE's theory setting).
    Same graph density as the existing generator (expected d edges, random permutation)."""
    r = np.random.RandomState(seed); perm = r.permutation(d); dens = 2.0 / (d - 1)
    A = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a):
            if r.rand() < dens: A[perm[b], perm[a]] = 1
    fns = [lambda x: np.tanh(1.5 * x), lambda x: np.sin(1.5 * x), lambda x: (x ** 2 - 1.0), lambda x: x / (1 + np.abs(x))]
    X = np.zeros((n, d))
    for e in perm:
        pa = np.where(A[:, e])[0]
        if len(pa) == 0: X[:, e] = r.randn(n); continue
        s = sum(r.choice([-1, 1]) * fns[r.randint(4)](X[:, c]) for c in pa)
        s = (s - s.mean()) / (s.std() + 1e-8); X[:, e] = s + noise * r.randn(n)
    return X.astype(np.float32), A


def gen_old(n, d, seed):
    from src.synthetic import generate_instantaneous_anm
    return generate_instantaneous_anm(n, d, seed=seed)


def topo_order(A):
    d = len(A); indeg = A.sum(0).copy(); out = []; stack = [i for i in range(d) if indeg[i] == 0]
    while stack:
        u = stack.pop(); out.append(u)
        for v in np.where(A[u])[0]:
            indeg[v] -= 1
            if indeg[v] == 0: stack.append(v)
    return out


def order_agree(order, A):
    pos = {v: k for k, v in enumerate(order)}; c, e = np.where(A)
    return float(np.mean([pos[a] < pos[b] for a, b in zip(c, e)]))


def varsortability(X, A):
    v = X.var(0); c, e = np.where(A); return float(np.mean(v[c] < v[e]))


def metrics(S, A, binary=None):
    """S, A indexed [cause, effect]. binary: optional boolean predicted graph for a label-free F1."""
    d = A.shape[0]; m = ~np.eye(d, dtype=bool); p = np.nan_to_num(np.abs(S)[m].astype(float)); t = A[m].astype(int)
    if p.max() > 0: p = p / p.max()
    out = dict(auroc=roc_auc_score(t, p), auprc=average_precision_score(t, p))
    pr, rc, _ = precision_recall_curve(t, p)
    out["f1_best"] = float(np.max(2 * pr * rc / np.maximum(pr + rc, 1e-12)))
    for q in (0.5, 0.8): out[f"rec@p{q}"] = float(rc[pr >= q].max()) if (pr >= q).any() else 0.0
    if binary is not None:
        b = binary[m].astype(bool); tp = (b & (t == 1)).sum(); pp = tp / max(b.sum(), 1); rr = tp / t.sum()
        out["f1_fixed"] = float(2 * pp * rr / max(pp + rr, 1e-12)); out["n_pred"] = int(b.sum())
    return out
