"""Gap B: models, grids, datasets, per-setting metrics. See PLAN.md. usage: run.py val|test"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
import numpy as np, json, warnings
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, average_precision_score
import methods0 as M0
warnings.filterwarnings("ignore")

GRIDS = {
    "lasso": [0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4],
    "thrvar": [0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35],
    "dynotears": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "pcmci+": [0.001, 0.01, 0.05, 0.1, 0.25, 0.5],
    "varlingam": [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
}
FAMS = list(GRIDS)
SPLIT = 0.7

# ---------- restricted VAR(1) (own lag always kept) ----------
def restricted_var(Xtr, P):
    """P[i,j]=1 if lag-1 value of i is a regressor for target j (off-diagonal). Returns B (d,d; B[i,j] i->j incl. diagonal), b."""
    d = Xtr.shape[1]; Xp, Y = Xtr[:-1], Xtr[1:]
    B = np.zeros((d, d)); b = np.zeros(d)
    for j in range(d):
        cols = [j] + [i for i in range(d) if i != j and P[i, j]]
        D = np.c_[Xp[:, cols], np.ones(len(Xp))]
        beta, *_ = np.linalg.lstsq(D, Y[:, j], rcond=None)
        B[cols, j] = beta[:-1]; b[j] = beta[-1]
    return B, b

def one_step_mse(Xseg, pred):
    """per-target MSE of predictions pred (aligned with Xseg[1:])"""
    return ((Xseg[1:] - pred) ** 2).mean(0)

# ---------- family fits: return list of settings dicts with S, predict(Xprev)->Xhat ----------
def _setting(S, predict):
    S = np.array(S, float); np.fill_diagonal(S, 0)
    return dict(S=S, predict=predict)

def fit_lasso(Xtr, grid):
    d = Xtr.shape[1]; out = []
    for a in grid:
        B = np.zeros((d, d)); b = np.zeros(d)
        for j in range(d):
            m = Lasso(alpha=a, max_iter=20000, tol=1e-6).fit(Xtr[:-1], Xtr[1:, j])
            B[:, j] = m.coef_; b[j] = m.intercept_
        S = np.abs(B); S[np.abs(S) < 1e-10] = 0
        out.append(_setting(S, lambda Xp, B=B, b=b: Xp @ B + b))
    return out

def fit_thrvar(Xtr, grid):
    d = Xtr.shape[1]
    A, _ = M0.var_fit(Xtr, 1); A = A[0]
    off = ~np.eye(d, dtype=bool); out = []
    for tau in grid:
        P = (np.abs(A) >= tau) & off if tau > 0 else off.copy()
        B, b = restricted_var(Xtr, P)
        S = np.abs(B) * P
        out.append(_setting(S, lambda Xp, B=B, b=b: Xp @ B + b))
    return out

def fit_dynotears(Xtr, grid):
    d = Xtr.shape[1]; Z, Ys = M0.lagmats(Xtr, 1)
    mz, my = Z.mean(0), Ys[0].mean(0); Zc, Yc = Z - mz, [Ys[0] - my]; out = []
    for lam in grid:
        W, As = M0._solve(Zc, Yc, lam); A = As[0]
        R = A @ np.linalg.inv(np.eye(d) - W)
        S = np.maximum(np.abs(W), np.abs(A)); S[S <= 0.01] = 0
        out.append(_setting(S, lambda Xp, R=R: (Xp - my) @ R + mz))
    return out

def fit_pcmci(Xtr, grid):
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    d = Xtr.shape[1]; out = []
    for a in grid:
        pc = PCMCI(dataframe=pp.DataFrame(Xtr), cond_ind_test=ParCorr(), verbosity=0)
        r = pc.run_pcmciplus(tau_min=0, tau_max=1, pc_alpha=a)
        g, v = r["graph"], np.abs(r["val_matrix"])
        S = np.zeros((d, d)); P = np.zeros((d, d), bool)
        for i in range(d):
            for j in range(d):
                if i == j: continue
                linked = g[i, j, 0] in ("-->", "o-o", "x-x") or g[i, j, 1] == "-->"
                if linked:
                    S[i, j] = 2.0 + max(v[i, j, 0], v[i, j, 1]); P[i, j] = True
        B, b = restricted_var(Xtr, P)
        out.append(_setting(S, lambda Xp, B=B, b=b: Xp @ B + b))
    return out

def fit_varlingam(Xtr, grid):
    import lingam
    d = Xtr.shape[1]
    m = lingam.VARLiNGAM(lags=1, prune=False, random_state=0).fit(Xtr)
    B0, B1 = np.array(m.adjacency_matrices_[0]), np.array(m.adjacency_matrices_[1])
    off = ~np.eye(d, dtype=bool); out = []
    for tb in grid:
        b0 = np.where(off & (np.abs(B0) >= max(tb, 1e-12)), B0, 0.0)
        b1 = np.where(off & (np.abs(B1) >= max(tb, 1e-12)), B1, np.where(off, 0.0, B1))
        R = (np.linalg.inv(np.eye(d) - b0) @ b1).T      # row convention: xhat = x_prev @ R
        S = np.maximum(np.abs(b0), np.abs(np.where(off, b1, 0))).T
        out.append(_setting(S, lambda Xp, R=R: Xp @ R))
    return out

FIT = {"lasso": fit_lasso, "thrvar": fit_thrvar, "dynotears": fit_dynotears, "pcmci+": fit_pcmci, "varlingam": fit_varlingam}

# ---------- per dataset ----------
def split_std(X):
    n = int(round(SPLIT * len(X)))
    mu, sd = X[:n].mean(0), X[:n].std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    return Z[:n], Z[n - 1:]        # validation segment includes the last train row as the first "previous" observation

def choose(vals, nedges, rule, target=None, aux=None):
    """index among candidates (nedges>=1, finite vals) according to rule; ties: fewer edges then grid order"""
    idx = [i for i in range(len(vals)) if nedges[i] >= 1 and np.isfinite(vals[i])]
    if not idx: return None
    if rule == "sp": key = lambda i: (abs(nedges[i] - target), nedges[i], i)
    else: key = lambda i: (vals[i], nedges[i], i)
    return min(idx, key=key)

def necessity(Xtr, Xva, G):
    d = Xtr.shape[1]
    B, b = restricted_var(Xtr, G)
    base = one_step_mse(Xva, Xva[:-1] @ B + b)
    N = np.zeros((d, d))
    for i, j in zip(*np.where(G)):
        P = G.copy(); P[i, j] = False
        B2, b2 = restricted_var(Xtr, P)
        m2 = one_step_mse(Xva, Xva[:-1] @ B2 + b2)
        N[i, j] = m2[j] - base[j]
    return N

def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

def run_dataset(meta, X, Y, families=FAMS):
    """X (T,d) raw, Y (d,d) 0/1 ground truth i->j. Returns (setting rows, necessity rows)."""
    d = X.shape[1]; Xtr, Xva = split_std(X)
    off = ~np.eye(d, dtype=bool); y = Y[off]
    rows, nrows = [], []
    for fam in families:
        try:
            sets = FIT[fam](Xtr, GRIDS[fam])
        except Exception as ex:
            rows.append(dict(meta, family=fam, setting=-1, err=repr(ex)[:100])); continue
        vm, bic, ne, au, ap = [], [], [], [], []
        n = len(Xtr) - 1
        for k, s in enumerate(sets):
            S = s["S"]
            try:
                pv = s["predict"](Xva[:-1]); mv = one_step_mse(Xva, pv).mean()
                rss = ((Xtr[1:] - s["predict"](Xtr[:-1])) ** 2).mean(0)
                nedge = int((S > 0).sum())
                bi = n * np.log(np.maximum(rss, 1e-12)).sum() + (nedge + d) * np.log(n)
            except Exception:
                mv, bi, nedge = np.nan, np.nan, int((S > 0).sum())
            if not np.isfinite(mv): mv = np.nan
            a1 = safe_auc(y, S[off]); a2 = average_precision_score(y, S[off]) if 0 < y.sum() < len(y) else np.nan
            vm.append(mv); bic.append(bi); ne.append(nedge); au.append(a1); ap.append(a2)
            rows.append(dict(meta, family=fam, setting=k, grid=GRIDS[fam][k], val_mse=mv, bic=bi, nedges=nedge,
                             auroc=a1, auprc=a2, true_edges=int(y.sum()), err=""))
        # necessity at the forecast-selected setting
        c = choose(vm, ne, "fc")
        if c is not None:
            S = sets[c]["S"]; G = S > 0
            N = necessity(Xtr, Xva, G)
            aN = safe_auc(y, N[off]); aS = safe_auc(y, S[off])
            gm = G & off
            yG = Y[gm]
            aNG = safe_auc(yG, N[gm]) if 0 < yG.sum() < len(yG) else np.nan
            aSG = safe_auc(yG, S[gm]) if 0 < yG.sum() < len(yG) else np.nan
            nrows.append(dict(meta, family=fam, setting=c, auc_nec=aN, auc_S=aS, auc_nec_G=aNG, auc_S_G=aSG,
                              nedges=int(gm.sum()), prec=float(yG.mean()) if len(yG) else np.nan))
    return rows, nrows

# ---------- datasets ----------
def rivers_datasets(seeds):
    import lib0
    from common import RES
    sub = [s for s in json.load(open("subgraphs.json"))["subgraphs"] if s["seed"] in seeds]
    for sg in sub:
        nodes = sg["nodes"]; idx = {n: i for i, n in enumerate(nodes)}; d = len(nodes)
        Y = np.zeros((d, d), int)
        for u, v in sg["edges"]: Y[idx[u], idx[v]] = 1
        for rn in ("12h", "24h"):
            X = lib0.get_series(sg, rn, standardize=False)
            yield dict(data="rivers", cell=f"{sg['region']}_d{d}_{rn}", cluster=f"{sg['region']}_{d}_{sg['seed']}",
                       region=sg["region"], d=d, res=rn, seed=sg["seed"]), X, Y

def netsim_datasets(phase, nsub=20):
    import scipy.io as sio, glob, re
    files = sorted(glob.glob("/tmp/claude-0/dl/ns/sim*.mat"), key=lambda f: int(re.findall(r"sim(\d+)", f)[0]))
    elig = []
    for f in files:
        m = sio.loadmat(f)
        if int(m["Nnodes"][0, 0]) <= 15: elig.append(int(re.findall(r"sim(\d+)", f)[0]))
    perm = np.random.default_rng(6000).permutation(elig)
    nval = int(np.ceil(0.2 * len(elig)))
    chosen = sorted(perm[:nval]) if phase == "val" else sorted(perm[nval:])
    for s in chosen:
        m = sio.loadmat(f"/tmp/claude-0/dl/ns/sim{s}.mat")
        d = int(m["Nnodes"][0, 0]); ns = int(m["Nsubjects"][0, 0]); T = int(m["Ntimepoints"][0, 0])
        for k in range(min(ns, nsub)):
            X = np.asarray(m["ts"][k * T:(k + 1) * T], float)
            Y = (np.abs(m["net"][k]) > 0).astype(int); np.fill_diagonal(Y, 0)
            yield dict(data="netsim", cell=f"sim{s}", cluster=f"sim{s}", d=d, sim=s, subject=k, seed=-1), X, Y

def synth_datasets(seeds):
    for seed in seeds:
        for d in (5, 10):
            rng = np.random.default_rng(seed * 100 + d)
            p = 1.5 / (d - 1)
            order = rng.permutation(d)
            def mat(strict_lower):
                C = np.zeros((d, d))
                for a in range(d):
                    for b_ in range(d):
                        if a == b_: continue
                        if strict_lower and not (order[a] < order[b_]): continue
                        if rng.random() < p: C[a, b_] = rng.uniform(0.2, 0.6) * rng.choice([-1, 1])
                return C
            W = mat(True); A = mat(False); A[np.diag_indices(d)] = 0.3     # W[i,j]: i->j contemporaneous, A[i,j]: lag
            rad = max(abs(np.linalg.eigvals(A @ np.linalg.inv(np.eye(d) - W))))
            if rad > 0.8: A = A * (0.8 / rad)
            T = 700; E = rng.laplace(size=(T, d)); X = np.zeros((T, d)); inv = np.linalg.inv(np.eye(d) - W)
            for t in range(1, T): X[t] = (X[t - 1] @ A + E[t]) @ inv
            X = X[200:]
            Y = ((np.abs(W) > 0) | (np.abs(A) > 0)).astype(int); np.fill_diagonal(Y, 0)
            yield dict(data="synth", cell=f"synth_d{d}", cluster=f"synth_{d}_{seed}", d=d, seed=seed), X, Y
