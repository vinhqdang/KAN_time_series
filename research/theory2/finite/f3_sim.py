"""F3 checks (FINITE.md Section 3): conditioning on a proxy Z of the driver.  Fixed seeds.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 f3_sim.py > f3_results.txt"""
import numpy as np
from scipy.stats import norm
from common import *

H = 400; d = 5; par = {2: [0, 1], 3: [2]}; edges = [(p, c) for c, ps in par.items() for p in ps]
A_Y = in_forest_A(par, [0.7] * d, 0.4); rel = relatives(A_Y); np.fill_diagonal(rel, False)
iu = np.triu_indices(d, 1); nonrel = ~rel[iu]; s_river = rel_sort(1 + rel.sum(1), edges)
vY = np.diag(var1_gamma0(A_Y, np.eye(d)))
phi = 0.9; LAGMAX = 8

def model(ell, r, lags=None, contam=0.0):
    """state (Y, C, C(t-1..t-LAGMAX)); X_v = Y_v + b_v C(t - lag_v); Z = sqrt(r) C(t) + sqrt(1-r) e + contam * Y_3/sd."""
    k = d + 1 + LAGMAX; A = np.zeros((k, k)); A[:d, :d] = A_Y; A[d, d] = phi
    for j in range(LAGMAX): A[d + 1 + j, d + j] = 1.0
    S = np.zeros((k, k)); S[:d, :d] = np.eye(d); S[d, d] = 1 - phi ** 2
    b = np.sqrt(ell / (1 - ell) * vY); lags = [0] * d if lags is None else lags
    M = np.zeros((d + 1, k)); M[:d, :d] = np.eye(d)
    for v in range(d): M[v, d + lags[v]] = b[v]
    M[d, d] = np.sqrt(r); M[d, 3] = contam / np.sqrt(vY[3])
    Nn = np.zeros((d + 1, 1)); Nn[d, 0] = np.sqrt(1 - r)
    return SS(A, S, M, Nn)

def delta_var_partial(g):
    """Asymptotic n*Var of sample partial correlations given the last variable (delta method, Isserlis)."""
    Hh = (len(g) - 1) // 2; q = g.shape[1]; G0 = g[Hh]
    pairs = [(a, b) for a in range(q) for b in range(a, q)]
    Mcov = np.array([[float((g[:, a, c] * g[:, b, dd] + g[:, a, dd] * g[:, b, c]).sum()) for (c, dd) in pairs] for (a, b) in pairs])
    def pc(Gm):
        s = np.sqrt(np.diag(Gm)); Rm = Gm / np.outer(s, s); return partial_given_last(Rm[None])[0]
    base = pc(G0); J = np.zeros((len(pairs), q - 1, q - 1)); h = 1e-6
    for k, (a, b) in enumerate(pairs):
        Gp = G0.copy(); Gp[a, b] += h
        if a != b: Gp[b, a] += h
        J[k] = (pc(Gp) - base) / h
    return np.einsum('kij,kl,lij->ij', J, Mcov, J), base

def formula_pi(ell, r):   # Prop F3(b): equal shares
    return (1 - r) * ell / (1 - r * ell)

print("F3: river-only stationary rel-sortability", s_river, "| non-rel pairs", [(int(a), int(b)) for a, b, m in zip(*iu, nonrel) if m])
REPS = 600
for li, ell in enumerate([0.3, 0.64]):
    for ri, r in enumerate([1.0, 0.99, 0.95, 0.9, 0.8]):
        ss = model(ell, r); g = standardise_acov(ss.acov(H)); Raug = g[H]
        Wp, Pi = delta_var_partial(g); pin = np.abs(Pi[iu][nonrel]); zc = fisher_tau(100, d)[1]
        ncrit = 4 + (zc / np.arctanh(pin.max())) ** 2 if pin.max() > 0 else np.inf
        rz = Raug[:d, d]; r_triad = np.median([rz[a] * rz[b] / Raug[a, b] for a, b, m in zip(*iu, nonrel) if m])
        print(f"\n== ell={ell} r={r}: population |pi| non-rel = {pin.round(5).tolist()}  formula {formula_pi(ell, r):.5f}; "
              f"population triad (rho_uZ rho_vZ / rho_uv) = {r_triad:.5f}; predicted n_crit (nominal, k=1) = {ncrit:.0f}")
        grid = [200, 1000, 5000] if not np.isfinite(ncrit) else sorted(set([200, 1000, 5000] + [int(ncrit * f) for f in (0.5, 1, 2)]))
        grid = [n for n in grid if 50 <= n <= 20000]
        print("   n | frac non-rel cond. declared emp | predicted asympt. | frac rel cond. declared | P(s_hat_Z = s_river) | mean acc s_hat_Z | mean triad r_hat | F3 bound P(no spurious & all rel)")
        for ni, n in enumerate(grid):
            rng = np.random.default_rng(3000 + 1000 * li + 100 * ri + ni); Rh = ss.sample_stats(n, REPS, rng)
            P = partial_given_last(Rh); tau = np.full((d, d), fisher_tau(n, d, k=1)[0]); dec = np.abs(P) > tau[None]
            D_ = dec[:, iu[0], iu[1]]
            sd = np.sqrt(Wp[iu] / (n * (1 - Pi[iu] ** 2) ** 2)); mu = np.arctanh(np.abs(Pi[iu])); tz = np.arctanh(tau[iu])
            pdec = norm.sf((tz - mu) / sd) + norm.cdf((-tz - mu) / sd)
            cnt = 1 + dec.sum(2) - np.diagonal(dec, axis1=1, axis2=2); sh = np.array([rel_sort(cnt[k], edges) for k in range(REPS)])
            rzh = Rh[:, :d, d]; tri = np.median(np.stack([rzh[:, a] * rzh[:, b] / Rh[:, a, b] for a, b, m in zip(*iu, nonrel) if m], 1), 1)
            Bc, Dc, k1, k2, _ = f1_constants(g)
            bnd = f3_bracket_prob(n, Raug, Bc, Dc, k1, k2, tau, rel) if n >= 1000 else float('nan')
            print(f"   {n:5d} | {D_[:, nonrel].mean():.3f} | {pdec[nonrel].mean():.3f} | {D_[:, ~nonrel].mean():.3f} | {(sh == s_river).mean():.3f} | {sh.mean():.3f} | {np.mean(tri):.4f} | {bnd:.3f}")

print("\n== Break attempt 1 (multi-filter driver): X_v = Y_v + b_v C(t - k_v), k = [0, 1, 2, 4, 8], Z = C(t) exactly (r = 1)")
ell = 0.64; lags = [0, 1, 2, 4, 8]; ss = model(ell, 1.0, lags=lags); g = standardise_acov(ss.acov(H)); Wp, Pi = delta_var_partial(g)
b = np.sqrt(ell / (1 - ell) * vY); G = var1_gamma0(A_Y, np.eye(d)); Vx = vY + b ** 2
for a, c, m in zip(*iu, nonrel):
    if not m: continue
    num = b[a] * b[c] * (phi ** abs(lags[a] - lags[c]) - phi ** (lags[a] + lags[c]))
    den = np.sqrt((Vx[a] - b[a] ** 2 * phi ** (2 * lags[a])) * (Vx[c] - b[c] ** 2 * phi ** (2 * lags[c])))
    print(f"   pair ({a},{c}) lags ({lags[a]},{lags[c]}): population pi = {Pi[a, c]:.5f}, Prop F3(d) formula = {num / den:.5f}, "
          f"n_crit = {4 + (fisher_tau(100, d)[1] / np.arctanh(abs(Pi[a, c]))) ** 2:.0f}")
for ni, n in enumerate([500, 2000, 10000]):
    rng = np.random.default_rng(3900 + ni); Rh = ss.sample_stats(n, REPS, rng); P = partial_given_last(Rh)
    dec = np.abs(P) > fisher_tau(n, d, k=1)[0]; print(f"   n={n}: frac non-rel cond. declared {dec[:, iu[0], iu[1]][:, nonrel].mean():.3f}")

print("\n== Break attempt 2 (proxy contaminated by a common descendant): Z = C + c * Y_3/sd(Y_3), r_C=1, ell = 0.64")
for ci, cont in enumerate([0.0, 0.1, 0.3]):
    ss = model(0.64, 1.0, contam=cont); g = standardise_acov(ss.acov(H)); Wp, Pi = delta_var_partial(g)
    # Prop F3(e): numerator = Cov(u,v) - Cov(u,Z)Cov(v,Z)/Var Z, contamination only through Y_3
    Gm = g[H]; num01 = Gm[0, 1] - Gm[0, d] * Gm[1, d] / Gm[d, d]
    print(f"   c={cont}: pi(0,1) = {Pi[0, 1]:+.5f} (headwaters into the confluence, river non-relatives); pi(0,4) = {Pi[0, 4]:+.5f}; "
          f"numerator check {num01:+.5f}")

print("\n== Lemma F3.3 (Lipschitz bound for partial correlation): random 3x3 correlation matrices + perturbations")
rng = np.random.default_rng(3999); worst = 0.0; nchk = 0
for _ in range(20000):
    Lm = rng.standard_normal((3, 3 + rng.integers(0, 3))); Cm = Lm @ Lm.T; s = np.sqrt(np.diag(Cm)); R0 = Cm / np.outer(s, s)
    Lm2 = Lm + rng.standard_normal(Lm.shape) * 10 ** rng.uniform(-4, -0.5); C2 = Lm2 @ Lm2.T; s2 = np.sqrt(np.diag(C2)); R1 = C2 / np.outer(s2, s2)
    dmax = np.abs(R1 - R0).max(); q = max(abs(R0[0, 2]), abs(R0[1, 2])) + dmax
    if q >= 1: continue
    p0 = partial_given_last(R0[None])[0][0, 1]; p1 = partial_given_last(R1[None])[0][0, 1]
    worst = max(worst, abs(p1 - p0) / (lip_partial(q) * dmax)); nchk += 1
print(f"   {nchk} cases, max |dpi| / (L(q') * max|dr|) = {worst:.4f} (must be <= 1)")
