"""
SPOT: Screened, Order-restricted, spline-additive DAG learner (pilot for gap G3 / RQ-C).

ALGORITHM (per-edge spline functions kept at d in the low hundreds; no acyclicity penalty needed)
  1. ORDER   : SCORE-style leaf peeling (Rolland et al. 2022) with a Stein kernel estimate of diag(Hessian log p) on a
               subsample of n_ord rows; repeatedly remove the node whose Hessian-diagonal has the smallest variance.
               (Own re-implementation; cost O(d * n_ord^3).)
  2. SCREEN  : for node i at position p, candidates = predecessors in the order; if more than K, keep the K with the
               highest ExtraTrees importance (GENIE3-style) fitted on predecessors only.
  3. FIT     : per node, cubic B-spline additive regression (8 interior intervals, 11 coefs/edge, as in SPADE) on the
               candidates with a group-lasso penalty, solved exactly (FISTA) in Gram form, optional adaptive
               (Zou 2006) second stage. Edge score = std of the fitted edge function's contribution (same readout as
               SPADE.importance). The graph is acyclic BY CONSTRUCTION (edges only follow the estimated order).
GUARANTEES vs HEURISTICS
  * Guaranteed (by construction): the output is a DAG; per-edge functions are additive splines (readable shapes).
  * Conditional statement (standard, not new): IF the estimated order is a valid topological order of the true ANM and
    the screen keeps every true parent, then regressing node i on its candidates targets E[X_i | candidates] =
    sum_j f_ij(X_j) (noise independent of non-descendants), so the group-lasso additive fit is a consistent
    estimator of the true edge functions under the usual sparse-additive conditions (spline approximation + compatibility).
  * HEURISTIC (no guarantee given here): finite-sample correctness of the Stein/SCORE order (theory is for Gaussian noise
    and population quantities); recall of the ExtraTrees screen (weak or non-marginal parents can be dropped);
    ordering errors permanently delete the edges they flip (hard ceiling on recall = fraction of true edges whose
    direction agrees with the estimated order, reported as `order_agree`); no finite-sample error bound.
"""
import os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(1)
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import ExtraTreesRegressor

NB = 11; GRID = 8; ORD3 = 3; LO, HI = -3.0, 3.0


def zscore(X): return ((X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)).astype(np.float64)


def bspline_bases(x):
    """x [n, d] -> [n, d, 11], same knot vector and clamping as CausalKANInstant."""
    step = (HI - LO) / GRID
    g = np.arange(-ORD3, GRID + ORD3 + 1) * step + LO
    x = np.clip(x, LO, HI)[..., None]
    b = ((x >= g[:-1]) & (x < g[1:])).astype(np.float64)
    for i in range(1, ORD3 + 1):
        b = ((x - g[:-(i + 1)]) / (g[i:-1] - g[:-(i + 1)] + 1e-8)) * b[..., :-1] + ((g[i + 1:] - x) / (g[i + 1:] - g[1:-i] + 1e-8)) * b[..., 1:]
    return b


# ---------------------------------------------------------------- step 1: order
def score_order(Z, n_ord=1000, eta=1e-2, seed=0):
    rng = np.random.RandomState(seed)
    Zs = Z[rng.choice(len(Z), min(n_ord, len(Z)), replace=False)]
    remaining = list(range(Z.shape[1])); order_rev = []
    while len(remaining) > 1:
        X = Zs[:, remaining]; n, m = X.shape
        sq = (X ** 2).sum(1); D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
        sig2 = np.median(D2[D2 > 0]) / 2 if (D2 > 0).any() else 1.0
        K = np.exp(-D2 / (2 * sig2)); cf = cho_factor(K + eta * np.eye(n))
        rs = K.sum(1, keepdims=True); KX = K @ X; KX2 = K @ (X ** 2)
        g1 = (X * rs - KX) / sig2                                   # sum_b k_ab (x_al - x_bl)/sig2
        g2 = (X ** 2 * rs - 2 * X * KX + KX2) / sig2 ** 2 - rs / sig2   # sum_b d^2k/dx_bl^2
        s = -cho_solve(cf, g1); H = -s ** 2 + cho_solve(cf, g2)   # sign chosen on validation seeds 700-702 (d=15-20): +1 recovers the leaf, -1 does not
        leaf = int(np.argmin(H.var(0))); order_rev.append(remaining.pop(leaf))
    order_rev.append(remaining[0])
    return order_rev[::-1]                                          # sources first


# ---------------------------------------------------------------- step 2: screen
def screen(Z, order, Kmax, seed, n_est=30, n_rows=1500):
    d = Z.shape[1]; rng = np.random.RandomState(seed); rows = rng.choice(len(Z), min(n_rows, len(Z)), replace=False)
    ranked = {}                                                     # node -> predecessors sorted by importance
    for p, i in enumerate(order):
        pred = order[:p]
        if len(pred) <= 1: ranked[i] = list(pred); continue
        et = ExtraTreesRegressor(n_estimators=n_est, max_depth=8, min_samples_leaf=20, max_features=0.5, n_jobs=1,
                                 random_state=seed + i).fit(Z[rows][:, pred], Z[rows, i])
        ranked[i] = [pred[t] for t in np.argsort(-et.feature_importances_)[:Kmax]]
    return ranked


# ---------------------------------------------------------------- step 3: group-lasso spline fit in Gram form
def gram_all(Z):
    n, d = Z.shape; B = bspline_bases(Z); B = B - B.mean(0, keepdims=True)
    Bf = B.reshape(n, d * NB); Gf = Bf.T @ Bf / n; BtY = Bf.T @ (Z - Z.mean(0)) / n          # [d*NB, d*NB], [d*NB, d]
    return Gf, BtY


def fit_groups(Gf, BtY, cands, K, lam, adaptive=False, iters=200, ridge=1e-3, gamma=1.0):
    """cands: dict node -> list of candidate parents (<=K). Returns S[cause, effect] edge scores and support."""
    d = BtY.shape[1]; p = K * NB
    idx = np.zeros((d, K), int); msk = np.zeros((d, K), bool)
    for i, c in cands.items():
        idx[i, :len(c)] = c; msk[i, :len(c)] = True
    fi = (idx[:, :, None] * NB + np.arange(NB)).reshape(d, p)
    G = torch.tensor(Gf[fi[:, :, None], fi[:, None, :]]); c = torch.tensor(BtY[fi, np.arange(d)[:, None]])
    m = torch.tensor(np.repeat(msk, NB, axis=1)); G = G * (m[:, :, None] & m[:, None, :]); c = c * m
    G = G + ridge * torch.eye(p, dtype=G.dtype)[None]
    L = torch.linalg.eigvalsh(G)[:, -1].clamp_min(1e-6)[:, None]
    mk = torch.tensor(msk)

    def solve(w):
        b = torch.zeros(d, p, dtype=G.dtype); y = b.clone(); t = 1.0
        for _ in range(iters):
            gr = torch.einsum("dpq,dq->dp", G, y) - c; z = (y - gr / L).view(d, K, NB)
            nz = z.norm(dim=2, keepdim=True).clamp_min(1e-12)
            bn = (torch.clamp(1 - (lam * w / L).view(d, K, 1) / nz, min=0) * z).reshape(d, p) * m
            t2 = (1 + (1 + 4 * t * t) ** 0.5) / 2; y = bn + ((t - 1) / t2) * (bn - b); b, t = bn, t2
        return b
    w = torch.ones(d, K, dtype=G.dtype); b = solve(w)
    if adaptive:
        nrm = b.view(d, K, NB).norm(dim=2); w = 1.0 / (nrm + 0.05) ** gamma; w = w / w[mk].mean(); b = solve(w)
    bb = b.view(d, K, NB)
    contrib = torch.sqrt(torch.einsum("dkb,dkbc,dkc->dk", bb, G.view(d, K, NB, K, NB)[torch.arange(d)[:, None], torch.arange(K)[None], :, torch.arange(K)[None], :], bb).clamp_min(0))
    S = np.zeros((d, d)); sup = np.zeros((d, d), bool)
    cn = contrib.numpy(); nn_ = bb.norm(dim=2).numpy()
    for i, cl in cands.items():
        for k, j in enumerate(cl): S[j, i] = cn[i, k]; sup[j, i] = nn_[i, k] > 1e-8
    return S, sup


def spot(X, seed, K=15, lam=0.02, adaptive=True, order=None, n_ord=1000, cache=None):
    """Returns (S[cause,effect], support, info). `order` may be passed to run an oracle-order diagnostic."""
    t0 = time.time(); Z = zscore(X); info = {}
    if cache is not None and "order" in cache: est = cache["order"]
    else: est = order if order is not None else score_order(Z, n_ord, seed=seed)
    info["t_order"] = time.time() - t0; t1 = time.time()
    if cache is not None and "ranked" in cache: ranked = cache["ranked"]
    else: ranked = screen(Z, est, 15, seed)
    info["t_screen"] = time.time() - t1; t2 = time.time()
    if cache is not None and "gram" in cache: Gf, BtY = cache["gram"]
    else: Gf, BtY = gram_all(Z)
    if cache is not None: cache.update(order=est, ranked=ranked, gram=(Gf, BtY))
    S, sup = fit_groups(Gf, BtY, {i: r[:K] for i, r in ranked.items()}, K, lam, adaptive)
    info["t_fit"] = time.time() - t2; info["order"] = est
    return S, sup, info
