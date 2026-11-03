"""Semi-synthetic Sachs SEM with ground-truth edge functions (construction: research/semisyn/PLAN.md).
fit_sem(): consensus DAG + real preprocessed Sachs (log, z-score, all rows) -> per-node additive cubic-B-spline fit (ridge by CV).
make_gen(sem, copies): returns gen(n, seed, noise) -> X, A, f, sd, sim, raw_sd for `copies` independent blocks."""
import os, numpy as np, pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
NKNOTS = 7  # cubic, 7 quantile knots -> 9 basis functions per parent
ALPHAS = np.logspace(-2, 4, 13)


def load_sachs():
    X = pd.read_csv(os.path.join(ROOT, "data/sachs/cyto_full_data.csv")); T = pd.read_csv(os.path.join(ROOT, "data/sachs/cyto_full_target.csv"))
    names = list(X.columns); idx = {n: i for i, n in enumerate(names)}
    A = np.zeros((11, 11), int)
    for c, e in zip(T.Cause, T.Effect): A[idx[c], idx[e]] = 1
    assert A.sum() == 18
    L = np.log(X.to_numpy(float)); Z = (L - L.mean(0)) / L.std(0)   # same preprocessing as scripts/sachs_benchmark.py (+ z-score)
    return Z, A, names


def topo(A):
    d = len(A); rem = set(range(d)); order = []
    while rem:
        n = [i for i in rem if not any(A[j, i] and j in rem for j in range(d))]
        assert n, "cycle"; order += sorted(n); rem -= set(n)
    return order


def break_cycles(A, Z, names):
    """The 18-edge consensus graph contains ONE directed cycle (PIP2->PIP3->plcg->PIP2). A SEM needs a DAG: while a cycle exists, drop the cycle edge
    with the smallest |Pearson correlation| of its endpoints in the real z-scored data (rule fixed before any method was run). Returns A (17 edges), removed."""
    import networkx as nx
    A = A.copy(); removed = []
    while True:
        g = nx.DiGraph([(c, e) for c, e in zip(*np.nonzero(A))])
        try: cyc = nx.find_cycle(g)
        except nx.NetworkXNoCycle: return A, removed
        c, e = min(cyc, key=lambda ce: abs(np.corrcoef(Z[:, ce[0]], Z[:, ce[1]])[0, 1])); A[c, e] = 0; removed.append((names[c], names[e]))


def fit_sem(seed=0):
    Z, A, names = load_sachs(); A, removed = break_cycles(A, Z, names); d = 11; order = topo(A); sem = dict(removed=removed, names=names, A=A, order=order, node={}, resid_sd={}, r2={})
    for e in range(d):
        pa = list(np.nonzero(A[:, e])[0])
        if not pa:
            sem["node"][e] = dict(pa=[], b0=0.0, tr={}, coef={}, pool=Z[:, e].copy()); sem["resid_sd"][e] = float(Z[:, e].std()); sem["r2"][e] = 0.0; continue
        trs = {c: SplineTransformer(n_knots=NKNOTS, degree=3, knots="quantile", extrapolation="constant").fit(Z[:, [c]]) for c in pa}
        Bs = {c: trs[c].transform(Z[:, [c]]) for c in pa}; mu = {c: Bs[c].mean(0) for c in pa}
        M = np.hstack([Bs[c] - mu[c] for c in pa]); r = RidgeCV(alphas=ALPHAS, cv=5).fit(M, Z[:, e]); nb = Bs[pa[0]].shape[1]
        coef = {c: r.coef_[k * nb:(k + 1) * nb] for k, c in enumerate(pa)}
        pred = r.predict(M); res = Z[:, e] - pred
        sem["node"][e] = dict(pa=pa, b0=float(r.intercept_), tr=trs, mu=mu, coef=coef, pool=res.copy(), alpha=float(r.alpha_))
        sem["resid_sd"][e] = float(res.std()); sem["r2"][e] = float(1 - res.var() / Z[:, e].var())
    return sem


def fcurve(sem, c, e, x):
    nd = sem["node"][e]; return (nd["tr"][c].transform(np.asarray(x, float).reshape(-1, 1)) - nd["mu"][c]) @ nd["coef"][c]


def make_gen(sem, copies=1, noise_pop=200000):
    A1 = sem["A"]; order = sem["order"]; d1 = 11; d = d1 * copies
    A = np.kron(np.eye(copies, dtype=int), A1)

    def draw_block(rr, m, noise, mu, sd, do=None):
        Xs = np.zeros((m, d1))
        for e in order:
            nd = sem["node"][e]
            if do is not None and e == do[0]: Xs[:, e] = do[1]; continue
            u = rr.choice(nd["pool"], m, replace=True) if noise == "boot" else rr.randn(m) * sem["resid_sd"][e]
            v = u + nd["b0"]
            for c in nd["pa"]: v = v + fcurve(sem, c, e, Xs[:, c])
            Xs[:, e] = (v - mu[e]) / sd[e]
        return Xs
    cons = {}
    for noise in ("boot", "gauss"):   # population standardisation constants (fixed SEM scale) per noise variant
        rr = np.random.RandomState(777)
        Xs = np.zeros((noise_pop, d1)); mu = np.zeros(d1); sd = np.ones(d1)
        for e in order:
            nd = sem["node"][e]; u = rr.choice(nd["pool"], noise_pop) if noise == "boot" else rr.randn(noise_pop) * sem["resid_sd"][e]
            v = u + nd["b0"]
            for c in nd["pa"]: v = v + fcurve(sem, c, e, Xs[:, c])
            mu[e] = v.mean(); sd[e] = v.std(); Xs[:, e] = (v - mu[e]) / sd[e]
        cons[noise] = (mu, sd)

    def gen(n, seed, noise="boot"):
        mu, sd1 = cons[noise]; rr = np.random.RandomState(seed)
        X0 = np.hstack([draw_block(rr, n, noise, mu, sd1) for _ in range(copies)]); sdv = np.tile(sd1, copies)
        perm = rr.permutation(d) if copies > 1 else np.arange(d)    # column k holds original node perm[k]
        inv = np.empty(d, int); inv[perm] = np.arange(d)            # original node j sits at column inv[j]
        X = X0[:, perm]; Ap = A[np.ix_(perm, perm)]
        f = {}
        for k in range(copies):
            for c in range(d1):
                for e in range(d1):
                    if A1[c, e]: f[(inv[k * d1 + c], inv[k * d1 + e])] = (lambda x, c=c, e=e: fcurve(sem, c, e, x))
        sdd = np.ones(d)
        for j in range(d): sdd[inv[j]] = sdv[j]     # true curve in standardised units = f(x_c)/sdd[e]
        def sim(nsim, c=None, val=None, seed_=12345):
            r2 = np.random.RandomState(seed_); out = np.zeros((nsim, d))
            if c is None:
                for k in range(copies): out[:, inv[k * d1:(k + 1) * d1]] = draw_block(r2, nsim, noise, mu, sd1)
                return out
            oc = perm[c]; k = oc // d1
            out[:, inv[k * d1:(k + 1) * d1]] = draw_block(r2, nsim, noise, mu, sd1, do=(oc % d1, val))   # only the intervened block (blocks independent)
            return out
        return X.astype(np.float32), Ap, f, sdd, sim, sdd.copy()
    return gen
