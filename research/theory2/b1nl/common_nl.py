"""B1-NL common code. Imports graphs, exact s_rho, estimators, Alg. 2 and metrics unchanged from ../b1/common.py
and adds the three new parametrisations of PLAN.md Section 1.2:

  NL    : nonlinear additive VAR(1), x_{t+1,j} = a_jj x_{t,j} + sum_{i != j} w_ij g_ij(x_{t,i}) + e_{t+1,j},
          w, a ~ Unif(0.5, 1) (positive), g_ij drawn uniformly from {tanh, sin, g3}, g3(x) = (x + x^2/2)/(1 + x^2/4);
          Gaussian noise. Rescaled so that the spectral radius of the Lipschitz bound matrix M = |W| * Lip(g) is <= 0.8.
  SGN   : linear VAR(1), every summary edge (self-loops included) has weight s * Unif(0.5, 1), s = +-1 w.p. 1/2;
          rescaled to spectral radius <= 0.8; Gaussian noise.
  SGNU  : the same graphs and weights as SGN (paired), noise Uniform with the same s.d.

Conventions as in b1: S[i,j] = edge i -> j; window column t*p + v.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b1"))
from common import *          # noqa: F401,F403  (summary_graph, srho_*, rel_counts, regress_in_order, shd, ...)
from common import FAMILIES, RHO_CAP

CONFIGS = ["NL", "SGN", "SGNU"]
PARAM_KEY = dict(NL=101, SGN=102, SGNU=102)     # SGN and SGNU share graphs and parameters (paired by design)
DATA_KEY = dict(NL=201, SGN=202, SGNU=203)
B_BURN = 100        # as in b1 after deviation D1
B_LONG = 500        # G2L: long burn-in, used only for sanity rule S1 (rel-SnR order accuracy)


def g3(x):
    return (x + 0.5 * x * x) / (1.0 + 0.25 * x * x)


_xx = np.linspace(-50, 50, 200001)
LIP_G3 = float(np.max(np.abs(np.gradient(g3(_xx), _xx))))
FUNCS = [np.tanh, np.sin, g3]
LIPS = np.array([1.0, 1.0, LIP_G3])


def params(cfg, family, p, seed):
    """Summary graph and parameters. Returns dict(S, W, F (function index per edge, -1 = linear), sd)."""
    rng = np.random.default_rng([PARAM_KEY[cfg], FAMILIES.index(family), p, seed])
    S = summary_graph(family, p, rng)
    mag = np.where(S, rng.uniform(0.5, 1.0, (p, p)), 0.0)
    sign = np.where(rng.random((p, p)) < 0.5, -1.0, 1.0)
    fidx = rng.integers(0, 3, (p, p))
    sd = rng.uniform(0.5, 1.0, p)
    if cfg == "NL":
        F = np.where(S & ~np.eye(p, dtype=bool), fidx, -1)
        W = mag
        M = W * np.where(F >= 0, LIPS[np.clip(F, 0, 2)], 1.0)
        r = max(abs(np.linalg.eigvals(M)))
        if r > RHO_CAP:
            W = W * RHO_CAP / r
    else:
        F = -np.ones((p, p), dtype=int)
        W = mag * sign
        r = max(abs(np.linalg.eigvals(W)))
        if r > RHO_CAP:
            W = W * RHO_CAP / r
    return dict(S=S, W=W, F=F, sd=sd, cfg=cfg)


def step(x, P, noise):
    """One step for a batch x (n, p)."""
    W, F = P["W"], P["F"]
    if P["cfg"] != "NL":
        return x @ W + noise
    p = W.shape[0]
    out = x * np.diag(W) + noise
    for k, g in enumerate(FUNCS):
        Wk = np.where(F == k, W, 0.0)
        if Wk.any():
            cols = np.flatnonzero(Wk.any(axis=1))
            out += g(x[:, cols]) @ Wk[cols]
    return out


def draw_noise(P, n, rng):
    sd = P["sd"]; p = len(sd)
    if P["cfg"] == "SGNU":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), (n, p)) * sd
    return rng.standard_normal((n, p)) * sd


def simulate_nl(P, T, n, gen, rng):
    """X of shape (n, T*p). gen: G1 (fixed start x_1 = e_1), G2 (burn-in B_BURN), G2L (burn-in B_LONG)."""
    p = len(P["sd"])
    B = dict(G1=0, G2=B_BURN, G2L=B_LONG)[gen]
    x = draw_noise(P, n, rng)
    for _ in range(B):
        x = step(x, P, draw_noise(P, n, rng))
    out = np.empty((n, T * p)); out[:, :p] = x
    for t in range(1, T):
        x = step(x, P, draw_noise(P, n, rng))
        out[:, t * p:(t + 1) * p] = x
    return out


def coupling_error(P, n=200, seed=0):
    """Sanity rule S3 (all configs): two copies driven by the SAME noise, one started at x_1 = e_1 (as G2), the other
    at 5 * sd * N(0, 1). Returns max over paths and nodes of |x_B - y_B| after B_BURN steps, divided by the largest
    across-path s.d. of x_B (a proxy for the stationary scale). A contraction (rho(M) <= 0.8) makes this tiny."""
    rng = np.random.default_rng([seed, 99])
    p = len(P["sd"])
    x = draw_noise(P, n, rng)
    y = 5.0 * rng.standard_normal((n, p)) * P["sd"]
    for _ in range(B_BURN):
        e = draw_noise(P, n, rng)
        x = step(x, P, e); y = step(y, P, e)
    scale = x.std(axis=0).max()
    return float(np.abs(x - y).max() / scale)


def window_relatives(S, T, gen):
    """Graph relatives (common ancestor, reflexive) among the window nodes, as a boolean (Tp, Tp) matrix.
    G1: the DAG G_T. G2/G2L: last T slices of G_{B+T} with B = 2p + T (>= Delta; adding past slices only adds
    relatives and they saturate at B >= Delta, so this equals the stationary relation of Thm 1.9)."""
    p = S.shape[0]
    B = 0 if gen == "G1" else 2 * p + T
    Tt = B + T; N = p * Tt
    G = true_dag(S, Tt).astype(np.int64)
    R = np.eye(N, dtype=np.int64)                 # R[a, b] = a is an ancestor of b (reflexive)
    for _ in range(Tt):
        R = ((R + R @ G) > 0).astype(np.int64)
    rel = (R.T @ R) > 0
    w = slice(p * B, N)
    return rel[w, w]
