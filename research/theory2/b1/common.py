"""B1 common code: summary graphs, weights, generators G1/G2/G2', exact s_rho (Thm 1.3/1.5 and Thm 1.9),
enumeration check, methods (time labels hidden), metrics.

Conventions. A summary graph on p nodes is a boolean matrix S (S[i,j] = edge i -> j, S[i,i] = True for all i).
Weights A (A[i,j] != 0 iff S[i,j]); one step is x_{t+1} = A^T x_t + e_{t+1} (column vectors).
Window variable index k = t*p + v (t = 0..T-1 is the paper's slice t+1). True window DAG: (t,i) -> (t+1,j) for S[i,j].
"""
import numpy as np
from scipy import linalg, stats
INF = 10 ** 9

# ----------------------------------------------------------------------------- summary graphs
FAMILIES = ["symER", "ring", "chain", "outtree", "intree", "ER1", "ER2"]
SC_FAMILIES = ["symER", "ring"]            # every weak component strongly connected (E_x empty)
NONSC_FAMILIES = ["chain", "outtree", "intree", "ER1", "ER2"]


def summary_graph(family, p, rng):
    S = np.zeros((p, p), dtype=bool)
    while True:
        S[:] = False
        if family == "symER":                       # undirected G(p, 2/(p-1)) symmetrised: expected degree 2
            q = 2.0 / (p - 1)
            U = np.triu(rng.random((p, p)) < q, 1)
            S |= U | U.T
        elif family == "ring":
            for i in range(p):
                S[i, (i + 1) % p] = True
        elif family == "chain":
            for i in range(p - 1):
                S[i, i + 1] = True
        elif family in ("outtree", "intree"):       # random recursive tree
            for k in range(1, p):
                par = rng.integers(0, k)
                if family == "outtree":
                    S[par, k] = True
                else:
                    S[k, par] = True
        elif family in ("ER1", "ER2"):              # directed G(p, c/p), cycles allowed
            c = 1.0 if family == "ER1" else 2.0
            S |= rng.random((p, p)) < c / p
            np.fill_diagonal(S, False)
        else:
            raise ValueError(family)
        np.fill_diagonal(S, False)
        if S.any():                                 # resample if no non-self-loop edge (only ER/symER can hit this)
            break
    np.fill_diagonal(S, True)
    return S


RHO_CAP = 0.8


def weights(S, rng, var_regime="het"):
    """Positive weights Unif(0.5,1) on every summary edge (self-loops included), rescaled so that the spectral
    radius is <= RHO_CAP. Positive weights => every trek-connected pair has nonzero correlation (faithfulness of
    relatives). Noise s.d.: het = Unif(0.5,1) per summary node; eq = 1."""
    p = S.shape[0]
    A = np.where(S, rng.uniform(0.5, 1.0, (p, p)), 0.0)
    r = max(abs(np.linalg.eigvals(A)))
    if r > RHO_CAP:
        A *= RHO_CAP / r
    sd_het = rng.uniform(0.5, 1.0, p)                # drawn always (keeps the RNG stream identical across regimes)
    sd = sd_het if var_regime == "het" else np.ones(p)
    return A, sd


# ----------------------------------------------------------------------------- generators
B_BURN = 50


def stationary_cov(A, sd):
    return linalg.solve_discrete_lyapunov(A.T, np.diag(sd ** 2))


def simulate(A, sd, T, n, gen, rng, B=B_BURN):
    """Returns X of shape (n, T*p), column t*p+v. gen in {G1, G2, G2s}."""
    p = A.shape[0]
    if gen == "G1":
        x = rng.standard_normal((n, p)) * sd
        steps_before = 0
    elif gen == "G2":
        x = rng.standard_normal((n, p)) * sd
        steps_before = B
    elif gen == "G2s":
        L = np.linalg.cholesky(stationary_cov(A, sd))
        x = rng.standard_normal((n, p)) @ L.T
        steps_before = 0
    else:
        raise ValueError(gen)
    for _ in range(steps_before):
        x = x @ A + rng.standard_normal((n, p)) * sd
    out = np.empty((n, T * p))
    out[:, :p] = x
    for t in range(1, T):
        x = x @ A + rng.standard_normal((n, p)) * sd
        out[:, t * p:(t + 1) * p] = x
    return out


def true_dag(S, T):
    p = S.shape[0]
    G = np.zeros((T * p, T * p), dtype=bool)
    for t in range(T - 1):
        G[t * p:(t + 1) * p, (t + 1) * p:(t + 2) * p] = S
    return G


# ----------------------------------------------------------------------------- exact theory
def dist_matrix(S):
    p = S.shape[0]
    d = np.full((p, p), INF, dtype=np.int64)
    np.fill_diagonal(d, 0)
    d[S & ~np.eye(p, dtype=bool)] = 1
    for k in range(p):
        d = np.minimum(d, d[:, [k]] + d[[k], :])
    d[d >= INF] = INF
    return d


def scc_labels(d):
    p = d.shape[0]
    comp = -np.ones(p, dtype=int); c = 0
    for i in range(p):
        if comp[i] < 0:
            mem = (d[i] < INF) & (d[:, i] < INF)
            comp[mem] = c; c += 1
    return comp


def delta_max(S):
    d = dist_matrix(S)
    return int(d[d < INF].max())


def edge_counts(S):
    """|E_S|, |E_x| (crossing), |E_o| (within SCC incl. loops), |E_up| (Src(j) != Src(i))."""
    d = dist_matrix(S); comp = scc_labels(d); p = S.shape[0]
    reach = d < INF
    anc = reach.T                                   # anc[v,u] = u reaches v
    rel = (anc.astype(int) @ reach.astype(int)) > 0  # rel[v,w] = exists u: u->v and u->w
    E = np.argwhere(S)
    ES = len(E)
    Ex = sum(comp[i] != comp[j] for i, j in E)
    Eup = sum(not np.array_equal(rel[i], rel[j]) for i, j in E)
    return ES, Ex, ES - Ex, Eup


def srho_G1_exact(S, T):
    """Thm 1.3 summed: s_rho(G_T) for Reisach's finite unrolled DAG (exact rational as float)."""
    d = dist_matrix(S)
    E = np.argwhere(S)
    ones = 0
    for i, j in E:
        ts = set()
        for u in range(S.shape[0]):
            lo, hi = d[u, j], d[u, i]
            if lo < INF:
                hi = min(hi, T - 1)
                ts.update(range(max(lo, 1), hi + 1))
        ones += len(ts)
    m = (T - 1) * len(E)
    return 0.5 + 0.5 * ones / m


def srho_limit(S):
    ES, Ex, Eo, Eup = edge_counts(S)
    return 0.5 + Ex / (2 * ES)


def srho_stationary(S):
    ES, Ex, Eo, Eup = edge_counts(S)
    return 0.5 + Eup / (2 * ES)


def rel_window_enum(S, T, B):
    """Exact relatives counts of window nodes (last T slices of G_{B+T}), relatives = marginal d-connection in the
    full DAG restricted to the window. Bitset enumeration (independent of the theorems). Returns counts (T*p,)."""
    p = S.shape[0]; Tt = B + T; N = p * Tt
    idx = lambda v, t: t * p + v
    anc = [0] * N; desc = [0] * N
    par = [[] for _ in range(N)]; ch = [[] for _ in range(N)]
    for t in range(Tt - 1):
        for i, j in np.argwhere(S):
            par[idx(j, t + 1)].append(idx(i, t)); ch[idx(i, t)].append(idx(j, t + 1))
    for x in range(N):
        a = 1 << x
        for q in par[x]: a |= anc[q]
        anc[x] = a
    for x in range(N - 1, -1, -1):
        a = 1 << x
        for q in ch[x]: a |= desc[q]
        desc[x] = a
    wmask = ((1 << (p * T)) - 1) << (p * B)
    cnt = np.zeros(p * T, dtype=int)
    for k in range(p * T):
        x = p * B + k; a = anc[x]; r = 0
        while a:
            lb = a & -a; r |= desc[lb.bit_length() - 1]; a ^= lb
        cnt[k] = bin(r & wmask).count("1")
    return cnt


def sortability(score, G, tol=0.0):
    """Edge-based sortability (Reisach et al. eq. 4) of a node score on the edges of DAG G."""
    E = np.argwhere(G)
    a, b = score[E[:, 0]], score[E[:, 1]]
    diff = b - a
    v = np.where(diff > tol, 1.0, np.where(diff < -tol, 0.0, 0.5))
    return float(v.mean())


# ----------------------------------------------------------------------------- estimators of node scores
def rel_counts(X, alpha=0.05):
    """Reisach et al. Algorithm 1: number of |corr| > eps per column (diagonal included), eps from the t-test."""
    n = X.shape[0]
    C = np.corrcoef(X, rowvar=False)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 2)
    eps = tcrit / np.sqrt(n - 2 + tcrit ** 2)
    return (np.abs(C) > eps).sum(axis=0).astype(float)


def r2_scores(X):
    C = np.cov(X, rowvar=False)
    P = np.linalg.inv(C)
    return 1.0 - 1.0 / (np.diag(C) * np.diag(P))


def var_scores(X):
    return X.var(axis=0)


def order_from_scores(score, rng, rtol=1e-9):
    """Ascending order, ties (within rtol of the scale) broken uniformly at random."""
    s = np.asarray(score, float)
    scale = max(np.abs(s).max(), 1e-300)
    q = np.round(s / (scale * rtol))
    tieb = rng.random(len(s))
    return np.lexsort((tieb, q))


# ----------------------------------------------------------------------------- regression in an order (Alg. 2)
def regress_in_order(X, order):
    """Reisach et al. Algorithm 2 (as in the sortnregress reference code): standardise, OLS, adaptive LassoLarsIC(BIC)."""
    from sklearn.linear_model import LinearRegression, LassoLarsIC
    Xs = (X - X.mean(0)) / X.std(0)
    d = X.shape[1]
    W = np.zeros((d, d))
    LR = LinearRegression(); LL = LassoLarsIC(criterion="bic")
    for k in range(1, d):
        cov = order[:k]; tgt = order[k]
        LR.fit(Xs[:, cov], Xs[:, tgt])
        w = np.abs(LR.coef_)
        LL.fit(Xs[:, cov] * w, Xs[:, tgt])
        W[cov, tgt] = LL.coef_ * w
    return W != 0


# ----------------------------------------------------------------------------- metrics
def shd(G_true, M_dir, M_und, exclude=None):
    """SHD between true DAG (bool) and an estimate given by directed part M_dir (bool, i->j) and undirected part
    M_und (bool symmetric). Each unordered pair counts 1 if its state differs. exclude: bool matrix of pairs to skip."""
    d = G_true.shape[0]
    iu = np.triu_indices(d, 1)
    t_state = np.where(G_true[iu], 1, np.where(G_true.T[iu], 2, 0))
    e_state = np.where(M_und[iu], 3, np.where(M_dir[iu], 1, np.where(M_dir.T[iu], 2, 0)))
    err = t_state != e_state
    if exclude is not None:
        err &= ~exclude[iu]
    return int(err.sum())


def order_accuracy(G_true, order):
    pos = np.empty(len(order), dtype=int); pos[order] = np.arange(len(order))
    E = np.argwhere(G_true)
    return float((pos[E[:, 0]] < pos[E[:, 1]]).mean())


def orient_score(G_true, M_dir, M_und):
    E = np.argwhere(G_true)
    return float(np.where(M_dir[E[:, 0], E[:, 1]], 1.0, np.where(M_und[E[:, 0], E[:, 1]], 0.5, 0.0)).mean())


def sid(G_true, M_dir):
    import gadjid
    r = gadjid.sid(G_true.astype(np.int8), M_dir.astype(np.int8), edge_direction="from row to column")
    return r[0], int(r[1])
