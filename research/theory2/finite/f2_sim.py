"""F2 checks (FINITE.md Section 2): saturation by an additive latent driver.  Fixed seeds.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 f2_sim.py > f2_results.txt"""
import numpy as np
from scipy.stats import norm
from common import *

H = 400; d = 5; par = {2: [0, 1], 3: [2]}; edges = [(p, c) for c, ps in par.items() for p in ps]
A_Y = in_forest_A(par, [0.7] * d, 0.4); rel = relatives(A_Y); np.fill_diagonal(rel, False)
iu = np.triu_indices(d, 1); nonrel = ~rel[iu]
s_true_river = rel_sort(1 + rel.sum(1), edges)
print("F2: river graph edges", edges, "| river-only stationary rel-sortability", s_true_river,
      "| non-relative pairs", [(int(a), int(b)) for a, b, m in zip(*iu, nonrel) if m])

def model(ell, phi=0.9):
    """state (Y_0..Y_4, C); X_v = Y_v + b_v C with driver share ell (equal for all v)."""
    k = d + 1; A = np.zeros((k, k)); A[:d, :d] = A_Y; A[d, d] = phi
    S = np.eye(k); S[d, d] = 1 - phi ** 2
    vY = np.diag(var1_gamma0(A_Y, np.eye(d))); b = np.sqrt(ell / (1 - ell) * vY) if ell > 0 else np.zeros(d)
    M = np.zeros((d, k)); M[:, :d] = np.eye(d); M[:, d] = b
    return SS(A, S, M)

ns = [30, 60, 100, 200, 500, 1000, 2000, 5000]; REPS = 1000
for li, ell in enumerate([0.02, 0.1, 0.3, 0.64]):
    ss = model(ell); g = standardise_acov(ss.acov(H)); R = g[H]; B, D, k1, k2, W = f1_constants(g)
    kap = np.abs(R[iu][nonrel]); zc = fisher_tau(100, d)[1]
    nhalf = 3 + (zc / np.arctanh(kap)) ** 2
    print(f"\n== driver share ell={ell}: kappa (non-rel corr) = {kap.round(4).tolist()}  (theory sqrt(l_u l_v) = {ell})")
    print(f"   predicted half-point n_1/2 = 3 + (z_c/atanh kappa)^2 (nominal test) = {nhalf.round(1).tolist()}; z_c={zc:.3f}")
    print("   n | frac non-rel declared (emp) | predicted (asympt.) | P(all pairs declared) emp | F1-type bound | P(no non-rel declared) emp | bound | mean s_hat | P(s_hat=1/2)")
    for ni, n in enumerate(ns):
        rng = np.random.default_rng(2000 + 100 * li + ni); Rh = ss.sample_stats(n, REPS, rng)
        tau = np.full((d, d), fisher_tau(n, d)[0]); dec = np.abs(Rh) > tau[None]
        D_ = dec[:, iu[0], iu[1]]; fr = D_[:, nonrel].mean()
        sd = np.sqrt(W[iu] / (n * (1 - R[iu] ** 2) ** 2)); mu = np.arctanh(np.abs(R[iu])); tz = np.arctanh(tau[iu])
        pdec = norm.sf((tz - mu) / sd) + norm.cdf((-tz - mu) / sd)
        all_dec = D_.all(1).mean(); b_all = f1_bound(n, R, B, D, k1, k2, tau, np.ones((d, d), bool))[0]
        tgt = rel.copy(); none_nr = (~D_[:, nonrel]).all(1).mean()
        # bound on P(no non-relative declared): target "declare relatives only" restricted to non-relatives -> use
        # the F1 bound with target = river relatives (a lower bound on P(Rhat = river relatives) <= P(no non-rel))
        b_none = f1_bound(n, R, B, D, k1, k2, tau, tgt)[0]
        cnt = 1 + dec.sum(2) - np.diagonal(dec, axis1=1, axis2=2)
        sh = np.array([rel_sort(cnt[k], edges) for k in range(REPS)])
        print(f"   {n:5d} | {fr:.3f} | {pdec[nonrel].mean():.3f} | {all_dec:.3f} | {b_all:.3f} | {none_nr:.3f} | {b_none:.3f} | {sh.mean():.3f} | {(sh == 0.5).mean():.3f}")
    # smallest n where the saturation bound >= 0.9, and the exponential rate of P(s_hat != 1/2) bound
    def bsat(n): return f1_bound(n, R, B, D, k1, k2, np.full((d, d), fisher_tau(n, d)[0]), np.ones((d, d), bool))[0]
    hi = 50
    while bsat(hi) < 0.9 and hi < 1e9: hi *= 2
    lo = hi // 2
    for _ in range(30):
        mid = (lo + hi) // 2; lo, hi = (lo, mid) if bsat(mid) >= 0.9 else (mid, hi)
    rates = [(n, 1 - bsat(n)) for n in [hi, 2 * hi, 4 * hi, 8 * hi]]
    print(f"   nonasymptotic saturation onset (bound >= 0.9): n = {hi}; bound on P(s_hat != 1/2): " + ", ".join(f"n={n}: {p:.2e}" for n, p in rates))
