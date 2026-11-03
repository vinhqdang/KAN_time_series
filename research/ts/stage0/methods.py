import numpy as np, scipy.linalg as sla, scipy.optimize as sopt, warnings
warnings.filterwarnings("ignore")

def lagmats(X, p):
    T = X.shape[0]
    Z = X[p:]
    Ys = [X[p - k: T - k] for k in range(1, p + 1)]
    return Z, Ys

def var_fit(X, p):
    """OLS VAR(p) with intercept. A[k][i,j] = effect of variable i at lag k+1 on variable j."""
    Z, Ys = lagmats(X, p)
    D = np.c_[np.hstack(Ys), np.ones(len(Z))]
    B, *_ = np.linalg.lstsq(D, Z, rcond=None)
    d = X.shape[1]
    A = [B[k * d:(k + 1) * d] for k in range(p)]
    R = Z - D @ B
    return A, R

def score_var(X, p):
    A, _ = var_fit(X, p)
    S = np.max([np.abs(a) for a in A], 0)
    np.fill_diagonal(S, 0)
    return S

def score_corr(X):
    S = np.abs(np.corrcoef(X.T)); np.fill_diagonal(S, 0); return S

def score_pcmciplus(X, p, alpha):
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    d = X.shape[1]
    pc = PCMCI(dataframe=pp.DataFrame(X), cond_ind_test=ParCorr(), verbosity=0)
    r = pc.run_pcmciplus(tau_min=0, tau_max=p, pc_alpha=alpha)
    g, v = r["graph"], np.abs(r["val_matrix"])
    S = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j: continue
            linked = False; val = 0.0
            for tau in range(0, p + 1):
                gl = g[i, j, tau]
                if tau == 0:
                    val = max(val, v[i, j, 0])
                    if gl in ("-->", "o-o", "x-x"): linked = True
                else:
                    val = max(val, v[i, j, tau])
                    if gl == "-->": linked = True
            S[i, j] = 2.0 * linked + val
    return S

def score_varlingam(X, p, prune):
    import lingam
    m = lingam.VARLiNGAM(lags=p, prune=prune, random_state=0).fit(X)
    Bs = [np.abs(b) for b in m.adjacency_matrices_]      # B[j,i]: effect of i on j
    S = np.max(Bs, 0).T.copy()
    np.fill_diagonal(S, 0)
    return S

# ---- DYNOTEARS (own implementation: Pamfil et al. 2020 objective, Zheng et al. 2018 augmented Lagrangian) ----
def _solve(Z, Ys, lam, max_outer=30, h_tol=1e-8, rho_max=1e16):
    n, d = Z.shape; p = len(Ys); nb = 1 + p
    def unpack(v):
        m = (nb) * d * d
        w = v[:m] - v[m:]
        return w[:d * d].reshape(d, d), [w[(k + 1) * d * d:(k + 2) * d * d].reshape(d, d) for k in range(p)]
    def f_and_g(v, rho, alpha):
        W, As = unpack(v)
        R = Z - Z @ W - sum(Y @ A for Y, A in zip(Ys, As))
        loss = 0.5 / n * (R ** 2).sum()
        E = sla.expm(W * W); h = np.trace(E) - d
        gW = -1.0 / n * Z.T @ R + (rho * h + alpha) * E.T * 2 * W
        gA = [-1.0 / n * Y.T @ R for Y in Ys]
        g = np.concatenate([gW.ravel()] + [x.ravel() for x in gA])
        obj = loss + 0.5 * rho * h * h + alpha * h + lam * np.abs(v).sum()
        return obj, np.concatenate([g + lam, -g + lam]), h
    m = nb * d * d
    bnds = []
    for half in range(2):
        for k in range(nb):
            for i in range(d):
                for j in range(d):
                    bnds.append((0, 0) if (k == 0 and i == j) else (0, None))
    v = np.zeros(2 * m); rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_outer):
        while rho < rho_max:
            res = sopt.minimize(lambda x: f_and_g(x, rho, alpha)[:2], v, jac=True, method="L-BFGS-B", bounds=bnds)
            vn = res.x; hn = f_and_g(vn, rho, alpha)[2]
            if hn > 0.25 * h: rho *= 10
            else: break
        v, h = vn, hn; alpha += rho * h
        if h <= h_tol or rho >= rho_max: break
    return unpack(v)

def score_dynotears(X, p, lam):
    Z, Ys = lagmats(X, p)
    Z = Z - Z.mean(0); Ys = [Y - Y.mean(0) for Y in Ys]
    W, As = _solve(Z, Ys, lam)
    S = np.max([np.abs(W)] + [np.abs(a) for a in As], 0); np.fill_diagonal(S, 0)
    return S

def score_notears_svar2(X, p, lam):
    """reference: OLS VAR, then linear NOTEARS on reduced-form residuals; lagged structural A_k = A_red_k (I - W)"""
    A, R = var_fit(X, p); d = X.shape[1]
    W, _ = _solve(R - R.mean(0), [], lam)
    As = [a @ (np.eye(d) - W) for a in A]
    S = np.max([np.abs(W)] + [np.abs(a) for a in As], 0); np.fill_diagonal(S, 0)
    return S

# ---- controls ----
def _sortnregress(X, order_score, rng):
    d = X.shape[1]
    o = np.lexsort((rng.random(d), order_score))       # increasing score, random tie-break
    Xc = X - X.mean(0); S = np.zeros((d, d))
    for pos, j in enumerate(o):
        pred = o[:pos]
        if len(pred) == 0: continue
        b, *_ = np.linalg.lstsq(Xc[:, pred], Xc[:, j], rcond=None)
        S[pred, j] = np.abs(b)
    return S

def score_var_sortnregress(X, rng):
    return _sortnregress(X, X.var(0), rng)

def score_r2_sortnregress(X, rng):
    from lib import r2_order_score
    return _sortnregress(X, r2_order_score(X), rng)

def score_random(X, rng):
    S = rng.random((X.shape[1],) * 2); np.fill_diagonal(S, 0); return S
