"""b1y common code: counterfactual starts, exact window covariances, variance matching, first-slice-excluded scoring.

Imports B1's summary graphs, weights, generators and metrics unchanged (../b1/common.py). Conventions as in B1:
one step is x_{t+1} = A^T x_t + e_{t+1}, window column index k = t*p + v, true window DAG (t,i) -> (t+1,j) for S[i,j].
No learner is run in this module.
"""
import os, sys
import numpy as np
from scipy import linalg, stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b1"))
from common import (FAMILIES, SC_FAMILIES, NONSC_FAMILIES, summary_graph, weights, simulate, true_dag, stationary_cov,
                    srho_G1_exact, srho_stationary, sortability, shd, order_accuracy, B_BURN, rel_counts,
                    order_from_scores, regress_in_order)

N_BASE = 1000
ALPHA = 0.05


def eps_threshold(n=N_BASE, alpha=ALPHA):
    """Alg. 1 threshold: |corr| > eps from the t-test at level alpha."""
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return t / np.sqrt(n - 2 + t ** 2)


# ------------------------------------------------------------------------------------------------ arms
# Fixed order: the arm index enters the data RNG key, so never reorder or insert before the end.
ARMS = ["G1", "G1rep", "G2", "G2s",
        "CF1_0.25", "CF1_0.5", "CF1_2", "CF1_4",          # G1 pattern (diagonal start, population relatives = G1's), start variance scaled by c
        "CF2_m1", "CF2_m3", "CF2_m10"]                    # G2 pattern, start = (1-lam) D + lam Sigma_stat, lam = min(1, m * lam_star)
CF1_C = {"CF1_0.25": 0.25, "CF1_0.5": 0.5, "CF1_2": 2.0, "CF1_4": 4.0}
CF2_M = {"CF2_m1": 1.0, "CF2_m3": 3.0, "CF2_m10": 10.0}
FRAC_EPS = 0.5          # lam_star keeps every pair that is exactly uncorrelated under G1 below FRAC_EPS * eps


def sigma_burn(A, sd, B=B_BURN):
    """Covariance of x after B latent steps from x_1 = e_1 (what the G2 generator simulates)."""
    D = np.diag(sd ** 2); S = D.copy()
    for _ in range(B):
        S = A.T @ S @ A + D
    return S


def window_cov(A, sd, T, Sigma1):
    """Exact covariance of (x_1..x_T) stacked (column t*p+v) for x_1 ~ N(0, Sigma1), then T-1 steps."""
    p = A.shape[0]; D = np.diag(sd ** 2)
    Sig = [Sigma1]
    for _ in range(1, T):
        Sig.append(A.T @ Sig[-1] @ A + D)
    C = np.zeros((T * p, T * p))
    for s in range(T):
        Ak = np.eye(p)
        for t in range(s, T):
            blk = Sig[s] @ Ak                           # Cov(x_s, x_t) = Sigma_s A^{t-s}
            C[s * p:(s + 1) * p, t * p:(t + 1) * p] = blk
            C[t * p:(t + 1) * p, s * p:(s + 1) * p] = blk.T
            Ak = Ak @ A
    return C


def corr_of(C):
    s = np.sqrt(np.diag(C)); return C / np.outer(s, s)


def zero_pairs_G1(A, sd, T, tol=1e-12):
    """Boolean matrix of window pairs that are exactly uncorrelated under G1 (population, positive weights)."""
    R = np.abs(corr_of(window_cov(A, sd, T, np.diag(sd ** 2))))
    Z = R < tol
    np.fill_diagonal(Z, False)
    return Z


def lam_star(A, sd, T, n=N_BASE, frac=FRAC_EPS, grid=None):
    """Largest lam on a geometric grid such that, for every lam' <= lam on the grid, all window pairs that are exactly
    uncorrelated under G1 have |rho| <= frac * eps under the start (1-lam') D + lam' Sigma_stat. Returns (lam, degenerate)
    with degenerate=True if no pair is uncorrelated under G1 (cannot happen for p >= 2: slice-1 nodes are independent)."""
    D = np.diag(sd ** 2); Ss = stationary_cov(A, sd); eps = eps_threshold(n)
    Z = zero_pairs_G1(A, sd, T)
    if not Z.any():
        return 1.0, True
    grid = np.geomspace(1e-7, 1.0, 71) if grid is None else grid
    best = 0.0
    for lam in grid:
        R = np.abs(corr_of(window_cov(A, sd, T, (1 - lam) * D + lam * Ss)))
        if R[Z].max() <= frac * eps:
            best = lam
        else:
            break
    return float(best), False


def start_cov(arm, A, sd, T, n=N_BASE):
    """Population covariance of the first slice for an arm (G2: after the latent burn-in B_BURN)."""
    D = np.diag(sd ** 2)
    if arm in ("G1", "G1rep"):
        return D
    if arm == "G2":
        return sigma_burn(A, sd)
    if arm == "G2s":
        return stationary_cov(A, sd)
    if arm in CF1_C:
        return CF1_C[arm] * D
    if arm in CF2_M:
        lam, _ = lam_star(A, sd, T, n)
        lam = min(1.0, CF2_M[arm] * lam)
        return (1 - lam) * D + lam * stationary_cov(A, sd)
    raise ValueError(arm)


def simulate_arm(arm, A, sd, T, n, rng):
    """Window data X (n, T*p), column t*p+v. G2 uses the B1 latent-burn-in simulator; every other arm draws x_1 from
    N(0, start_cov) (exact Gaussian) and runs T-1 steps."""
    if arm in ("G1", "G1rep"):
        return simulate(A, sd, T, n, "G1", rng)
    if arm == "G2":
        return simulate(A, sd, T, n, "G2", rng)
    p = A.shape[0]
    S1 = start_cov(arm, A, sd, T)          # lam_star is always defined at the base n = 1000 threshold
    L = np.linalg.cholesky(S1 + 1e-14 * np.eye(p))
    x = rng.standard_normal((n, p)) @ L.T
    out = np.empty((n, T * p)); out[:, :p] = x
    for t in range(1, T):
        x = x @ A + rng.standard_normal((n, p)) * sd
        out[:, t * p:(t + 1) * p] = x
    return out


# ------------------------------------------------------------------------------------------------ oracle (population) predictions
def oracle_oa(C, G, n=N_BASE):
    """Threshold-oracle order accuracy of rel-SortnRegress: edge-sortability (ties 1/2) of the counts
    c_i = #{j : |rho_ij| > eps} computed from the population correlations (b1x/oracle_b3.py, threshold version)."""
    R = np.abs(corr_of(C)); eps = eps_threshold(n)
    cnt = (R > eps).sum(axis=0).astype(float)
    return sortability(cnt, G)


def population_pattern(C, tol=1e-12):
    R = np.abs(corr_of(C)); P = R > tol; np.fill_diagonal(P, True); return P


# ------------------------------------------------------------------------------------------------ variance matching (Q2)
def marginal_var(A, sd, T, arm):
    return np.diag(window_cov(A, sd, T, start_cov(arm, A, sd, T)))


def vm_factors(A, sd, T, arm_from, arm_to):
    """Column scale factors that give data of arm_from the population marginal variances of arm_to. Correlations are
    unchanged (scale invariance), so rel-SortnRegress, FLOP and PC see identical data; DAGMA/NOTEARS on raw data do not."""
    return np.sqrt(marginal_var(A, sd, T, arm_to) / marginal_var(A, sd, T, arm_from))


# ------------------------------------------------------------------------------------------------ first-slice-excluded scoring (Q3)
def slice_of_cols(perm, p):
    """slice index (0-based) of each permuted column."""
    return perm // p


def mask_inside_first_slice(slice_of):
    return (slice_of[:, None] == 0) & (slice_of[None, :] == 0)


def oa_x1(G, order, slice_of):
    """Order accuracy over true edges whose tail is NOT in slice 1 (the edges leaving slice 1 are dropped)."""
    pos = np.empty(len(order), dtype=int); pos[order] = np.arange(len(order))
    E = np.argwhere(G)
    keep = slice_of[E[:, 0]] != 0
    E = E[keep]
    return float((pos[E[:, 0]] < pos[E[:, 1]]).mean()) if len(E) else float("nan")


def anc_oa(G, D):
    """Ancestral order accuracy of an estimated directed graph D (bool): mean over true edges (i,j) of 1 if i is an
    ancestor of j in D and j is not an ancestor of i, 0 if the reverse, 1/2 otherwise (incomparable, or both on a cycle);
    same definition as b1x/run_b1x.py anc_order_accuracy."""
    d = D.shape[0]
    R = np.linalg.matrix_power((D | np.eye(d, dtype=bool)).astype(np.int64), d) > 0   # reachability (incl. self)
    E = np.argwhere(G)
    a = R[E[:, 0], E[:, 1]]; b = R[E[:, 1], E[:, 0]]
    v = np.where(a & ~b, 1.0, np.where(b & ~a, 0.0, 0.5))
    return float(v.mean())


# ------------------------------------------------------------------------------------------------ rho-dose (Q1c)
def weights_cap(S, rng, cap):
    """Same RNG stream and draws as b1/common.weights (het regime), with the spectral-radius cap as a parameter
    (b1: 0.8). For cap = 0.8 this returns exactly b1/common.weights(S, rng, 'het')."""
    p = S.shape[0]
    A = np.where(S, rng.uniform(0.5, 1.0, (p, p)), 0.0)
    r = max(abs(np.linalg.eigvals(A)))
    if r > cap:
        A *= cap / r
    sd = rng.uniform(0.5, 1.0, p)
    return A, sd
