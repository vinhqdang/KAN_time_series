"""Shared code for the F1-F3 simulations (research/theory2/finite/FINITE.md).
Linear Gaussian VAR(1) on a summary graph with self-loops, optional additive latent driver C(t) (AR(1)),
optional proxy Z(t) = sqrt(r) C(t) + sqrt(1-r) e(t).  Exact population autocovariances, the F1 constants
(B_ab, D, kappa1, kappa2), the F1 deviation functions and the finite-sample bound."""
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------- population moments
def var1_gamma0(A, S):
    """Gamma(0) of X(t) = A X(t-1) + e, Cov(e) = S (doubling algorithm)."""
    G = S.copy(); Ak = A.copy()
    for _ in range(60):
        G = G + Ak @ G @ Ak.T; Ak = Ak @ Ak
        if np.abs(Ak).max() < 1e-300: break
    return G

def var1_acov(A, S, H):
    """gam[h + H] = Gamma(h) = E X(t+h) X(t)^T for h = -H..H."""
    G0 = var1_gamma0(A, S); d = len(A); out = np.zeros((2 * H + 1, d, d)); P = np.eye(d)
    for h in range(H + 1):
        out[H + h] = P @ G0; out[H - h] = (P @ G0).T; P = P @ A
    return out

def standardise_acov(gam):
    H = (len(gam) - 1) // 2; s = np.sqrt(np.diag(gam[H]))
    return gam / np.outer(s, s)[None]

def f1_constants(g):
    """g: standardised autocovariances, shape (2H+1, d, d), g[H+h][a,b] = gamma_ab(h).
    Returns B (d x d, off-diagonal Bartlett-type sums with absolute value), D, kappa1, kappa2, W (asymptotic
    n*Var(r_ab), delta method, Isserlis)."""
    H = (len(g) - 1) // 2; d = g.shape[1]
    gr = g[::-1]                                  # gr[H+h] = gamma(-h)
    B = np.zeros((d, d)); W = np.zeros((d, d))
    for a in range(d):
        for b in range(d):
            if a == b: continue
            B[a, b] = np.abs(g[:, a, a] * g[:, b, b] + g[:, a, b] * gr[:, a, b]).sum()
    D = max(2 * (g[:, a, a] ** 2).sum() for a in range(d))
    k1 = max(np.abs(g[:, a, a]).sum() for a in range(d))
    k2 = 0.0
    for a in range(d):
        for b in range(d):
            if a != b:
                k2 = max(k2, np.abs(g[:, a, a]).sum() + np.abs(g[:, a, b]).sum())
    # asymptotic covariance of (gamma_aa, gamma_bb, gamma_ab): n Cov(g_ab, g_cd) -> sum_h g_ac(h) g_bd(h) + g_ad(h) g_bc(h)
    def ac(a, b, c, dd): return float((g[:, a, c] * g[:, b, dd] + g[:, a, dd] * g[:, b, c]).sum())
    for a in range(d):
        for b in range(d):
            if a == b: continue
            rho = g[H, a, b]; idx = [(a, a), (b, b), (a, b)]
            M = np.array([[ac(*p, *q) for q in idx] for p in idx]); gr_ = np.array([-rho / 2, -rho / 2, 1.0])
            W[a, b] = gr_ @ M @ gr_
    return B, D, k1, k2, W

# ---------------------------------------------------------------- the F1 bound
def eps_off(n, x, y, Bab, k1, k2): return np.sqrt(2 * Bab * x / n) + k2 * x / n + 2 * k1 * y / n
def eps_diag(n, y, D, k1): return np.sqrt(2 * D * y / n) + 4 * k1 * y / n

def f1_bound(n, R, B, D, k1, k2, tau, nonzero, ys=None):
    """Lower bound on P(estimated relatives set == true), Theorem F1.  R: population correlation matrix,
    tau: d x d thresholds, nonzero: boolean d x d target decision (pairs that must be declared; the others must
    not be).  For F1 the target is the true relatives set (zero correlation off it).  Returns (bound, x*, y*)."""
    d = len(R); iu = np.triu_indices(d, 1)
    if ys is None: ys = np.linspace(0.05, 60, 600)
    best = (0.0, None, None)
    for y in ys:
        ed = eps_diag(n, y, D, k1)
        if ed >= 1: break
        # largest x with all conditions: bisection on x (conditions monotone)
        def ok(x):
            eo = eps_off(n, x, y, B[iu], k1, k2); delta = (eo + np.abs(R[iu]) * ed) / (1 - ed)
            nz = nonzero[iu]; t = tau[iu]
            return np.all(np.where(nz, np.abs(R[iu]) - delta > t, np.abs(R[iu]) + delta <= t))
        if not ok(1e-9): continue
        lo, hi = 1e-9, 1.0
        while ok(hi) and hi < 1e7: hi *= 2
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        x = lo; fail = d * (d - 1) * np.exp(-x) + 4 * d * np.exp(-y)
        if 1 - fail > best[0]: best = (1 - fail, x, y)
    return best

def fisher_tau(n_test, d, alpha=0.01, k=0):
    """Correlation threshold equivalent to a two-sided Fisher-z test, Bonferroni over d(d-1)/2 pairs, k conditioning vars."""
    m = d * (d - 1) / 2; zc = norm.isf(alpha / (2 * m))
    return np.tanh(zc / np.sqrt(np.maximum(n_test - 3 - k, 1))), zc

# ---------------------------------------------------------------- simulation (state-space, chunked)
def psd_sqrt(S):
    w, V = np.linalg.eigh((S + S.T) / 2); return V * np.sqrt(np.clip(w, 0, None))

class SS:
    """State s(t) = A s(t-1) + e(t), Cov(e) = S; observation O(t) = M s(t) + Nn eta(t), eta iid N(0, I)."""
    def __init__(self, A, S, M, Nn=None):
        self.A, self.S, self.M = A, S, M
        self.Nn = np.zeros((M.shape[0], 1)) if Nn is None else Nn
    def acov(self, H):
        gs = var1_acov(self.A, self.S, H); g = np.einsum('ik,hkl,jl->hij', self.M, gs, self.M)
        g[H] += self.Nn @ self.Nn.T; return g
    def sample_stats(self, n, reps, rng, chunk=2000):
        """Returns centred sample correlation matrices (reps, q, q) of the observations over n steps."""
        k = len(self.A); q = self.M.shape[0]
        s = rng.standard_normal((reps, k)) @ psd_sqrt(var1_gamma0(self.A, self.S)).T
        Lc = psd_sqrt(self.S).T; At = self.A.T; Mt = self.M.T; Nt = self.Nn.T
        S1 = np.zeros((reps, q)); S2 = np.zeros((reps, q, q)); done = 0
        while done < n:
            m = min(chunk, n - done); buf = np.empty((reps, m, k))
            E = rng.standard_normal((reps, m, k)) @ Lc
            for t in range(m):
                s = s @ At + E[:, t]; buf[:, t] = s
            O = buf @ Mt + rng.standard_normal((reps, m, Nt.shape[0])) @ Nt
            S1 += O.sum(1); S2 += np.einsum('rti,rtj->rij', O, O); done += m
        C = S2 / n - S1[:, :, None] * S1[:, None, :] / n ** 2
        sd = np.sqrt(np.einsum('rii->ri', C)); return C / (sd[:, :, None] * sd[:, None, :])

def partial_given_last(R):
    """R: (..., d+1, d+1) correlation incl. Z as last variable -> partial correlations among the first d given Z."""
    rz = R[..., :-1, -1]; Rxx = R[..., :-1, :-1]
    num = Rxx - rz[..., :, None] * rz[..., None, :]; s = np.sqrt(1 - rz ** 2)
    P = num / (s[..., :, None] * s[..., None, :]); return P

# ---------------------------------------------------------------- graphs
def in_forest_A(parents, lam, w):
    """parents: dict child -> list of parents. Returns VAR(1) matrix A (A[child, parent] = w)."""
    d = len(lam); A = np.diag(lam).astype(float)
    for c, ps in parents.items():
        for p in ps: A[c, p] = w
    return A

def relatives(A):
    d = len(A); R = (np.abs(A) > 0) | np.eye(d, dtype=bool)
    reach = R.copy()
    for _ in range(d): reach = reach | ((reach.astype(int) @ reach.astype(int)) > 0)   # reach[i,j]: j reaches i
    anc = reach                                                                      # anc[v] row: ancestors of v
    rel = (anc.astype(int) @ anc.T.astype(int)) > 0
    return rel

def rel_sort(counts, edges):
    v = [1.0 if counts[u] < counts[w] else (0.5 if counts[u] == counts[w] else 0.0) for (u, w) in edges]
    return float(np.mean(v)) if v else np.nan

def deviation_event(n, x, y, R, B, D, k1, k2):
    """delta_ab of Theorem F1 at (x, y) (d x d), or None if eps_d >= 1."""
    ed = eps_diag(n, y, D, k1)
    if ed >= 1: return None
    eo = eps_off(n, x, y, B, k1, k2); return (eo + np.abs(R) * ed) / (1 - ed)

def lip_partial(q):
    """Lipschitz constant of (a,b,c) -> (a-bc)/sqrt((1-b^2)(1-c^2)) in the max-norm, |b|,|c| <= q < 1 (Lemma F3.3)."""
    return 1 / (1 - q ** 2) + 4 * q / (1 - q ** 2) ** 2

def f3_bracket_prob(n, Raug, B, D, k1, k2, tau, target, ys=None):
    """Lower bound on P(conditional decisions == target) for partial correlations given the last variable
    (Theorem F3(c)): sample correlations of (X, Z) within delta on an event of prob >= 1 - fail, then
    |pi_hat - pi| <= L(q') max(delta_uv, delta_uZ, delta_vZ)."""
    q1 = len(Raug); d = q1 - 1; iu = np.triu_indices(d, 1)
    Pi = partial_given_last(Raug[None])[0]
    if ys is None: ys = np.linspace(0.05, 60, 300)
    best = 0.0
    for y in ys:
        if eps_diag(n, y, D, k1) >= 1: break
        def ok(x):
            dl = deviation_event(n, x, y, Raug, B, D, k1, k2)
            dz = dl[:d, d]; qq = np.abs(Raug[:d, d]) + dz
            qp = np.maximum(qq[:, None], qq[None, :])
            if np.any(qp[iu] >= 1): return False
            dm = np.maximum(dl[:d, :d], np.maximum(dz[:, None], dz[None, :]))
            dev = lip_partial(qp) * dm
            p, t, tg = np.abs(Pi[iu]), tau[iu], target[iu]
            return np.all(np.where(tg, p - dev[iu] > t, p + dev[iu] <= t))
        if not ok(1e-9): continue
        lo, hi = 1e-9, 1.0
        while ok(hi) and hi < 1e7: hi *= 2
        for _ in range(50):
            mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        fail = q1 * (q1 - 1) * np.exp(-lo) + 4 * q1 * np.exp(-y); best = max(best, 1 - fail)
    return best
