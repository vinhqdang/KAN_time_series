"""F1 checks (FINITE.md Section 1).  Fixed seeds.  Run: OMP_NUM_THREADS=1 /usr/bin/python3 f1_sim.py > f1_results.txt"""
import numpy as np
from scipy.stats import norm
from common import *

H = 400
def setup(parents, lam, w, d):
    A = in_forest_A(parents, lam, w); S = np.eye(d); ss = SS(A, S, np.eye(d))
    g = standardise_acov(ss.acov(H)); R = g[H]; rel = relatives(A); np.fill_diagonal(rel, False)
    B, D, k1, k2, W = f1_constants(g)
    return ss, g, R, rel, B, D, k1, k2, W

cases = {
  "C1 d=5 lam=0.5 w=0.5": ({2: [0, 1], 3: [2]}, [0.5] * 5, 0.5, 5),
  "C2 d=5 lam=0.9 w=0.3": ({2: [0, 1], 3: [2]}, [0.9] * 5, 0.3, 5),
  "C3 d=5 lam=0.97 w=0.1 (break attempt)": ({2: [0, 1], 3: [2]}, [0.97] * 5, 0.1, 5),
  "C4 d=10 in-forest lam=0.7 w=0.4": ({2: [0, 1], 4: [2, 3], 5: [4], 8: [6, 7], 9: [8]}, [0.7] * 10, 0.4, 10),
  "C5 d=5 iid-like lam=0.05 w=0.9 (contrast)": ({2: [0, 1], 3: [2]}, [0.05] * 5, 0.9, 5),
}
ns = [200, 500, 1000, 2000, 5000, 10000, 20000]
REPS = 1000
print("F1 simulation; reps per cell", REPS, "; seeds 1000+case*100+n-index")
for ci, (name, (par, lam, w, d)) in enumerate(cases.items()):
    ss, g, R, rel, B, D, k1, k2, W = setup(par, lam, w, d)
    iu = np.triu_indices(d, 1); rmin = np.abs(R[iu][rel[iu]]).min()
    Bnull = B[iu][~rel[iu]]; Wnull = W[iu][~rel[iu]]
    print(f"\n== {name}: rho_min(relatives) = {rmin:.4f}, max|rho| non-rel = {np.abs(R[iu][~rel[iu]]).max():.1e}, "
          f"Bartlett B(null) range [{Bnull.min():.2f},{Bnull.max():.2f}], W(null)==B(null): {np.allclose(Bnull, Wnull)}, "
          f"max B {B[iu].max():.2f}, D {D:.2f}, kappa1 {k1:.2f}, kappa2 {k2:.2f}")
    print(" n | tau | P_emp(Rhat=R) | F1 bound | P_emp(|r-rho|<=delta all) | 1-fail(x*,y*) | P_emp(s_hat=s) | FP rate/null pair (nominal) | predicted FP (asympt.)")
    for ni, n in enumerate(ns):
        rng = np.random.default_rng(1000 + 100 * ci + ni)
        Rh = ss.sample_stats(n, REPS, rng)
        for mode in ["nominal", "oracle_neff", "fixed_half_rhomin"]:
            if mode == "nominal":
                tau = np.full((d, d), fisher_tau(n, d)[0])
            elif mode == "oracle_neff":
                ne = n * (1 - R ** 2) ** 2 / np.where(W > 0, W, 1); zc = fisher_tau(n, d)[1]
                tau = np.tanh(zc / np.sqrt(np.maximum(ne - 3, 1)))
            else:
                tau = np.full((d, d), rmin / 2)
            dec = np.abs(Rh) > tau[None]
            ok = np.all(dec[:, iu[0], iu[1]] == rel[iu][None], axis=1)
            bnd, xs, ys = f1_bound(n, R, B, D, k1, k2, tau, rel)
            # the deviation event of the proof at (x*, y*)
            if xs is not None:
                ed = eps_diag(n, ys, D, k1); eo = eps_off(n, xs, ys, B[iu], k1, k2); delta = (eo + np.abs(R[iu]) * ed) / (1 - ed)
                ev = np.all(np.abs(Rh[:, iu[0], iu[1]] - R[iu][None]) <= delta[None], axis=1).mean()
            else: ev = np.nan
            # rel-sortability over non-self-loop edges
            edges = [(p, c) for c, ps in par.items() for p in ps]
            cnt_true = 1 + rel.sum(1); s_true = rel_sort(cnt_true, edges)
            cnt = 1 + dec[:, :, :].sum(2) - np.diagonal(dec, axis1=1, axis2=2)
            s_eq = np.mean([rel_sort(cnt[k], edges) == s_true for k in range(REPS)])
            nullmask = ~rel[iu]
            fp = dec[:, iu[0], iu[1]][:, nullmask].mean()
            zc = fisher_tau(n, d)[1]
            if mode == "nominal":   # asymptotic FP of nominal test on a null pair: P(|N(0,B/n)| > tau)
                pfp = np.mean(2 * norm.sf(np.arctanh(tau[0, 1]) * np.sqrt(n / Bnull)))
            elif mode == "oracle_neff": pfp = 2 * norm.sf(zc)
            else: pfp = np.mean(2 * norm.sf(rmin / 2 * np.sqrt(n / Bnull)))
            # asymptotic (sketched) approximation: independent pairs, Fisher z ~ N(atanh rho, W/(n(1-rho^2)^2))
            sd = np.sqrt(W[iu] / (n * (1 - R[iu] ** 2) ** 2)); mu = np.arctanh(np.abs(R[iu])); tz = np.arctanh(tau[iu])
            pdec = norm.sf((tz - mu) / sd) + norm.cdf((-tz - mu) / sd)
            papprox = np.prod(np.where(rel[iu], pdec, 1 - pdec))
            print(f" {n:6d} {mode:18s} approx {papprox:.3f} tau~{np.median(tau[iu]):.3f} | {ok.mean():.3f} | {bnd:.3f} | {ev:.3f} | {bnd if xs is not None else float('nan'):.3f} | {s_eq:.3f} | {fp:.4f} | {pfp:.4f}")
    # tightness: smallest n with bound >= 0.9 (oracle and fixed thresholds) vs smallest grid n with empirical >= 0.9
    for mode in ["oracle_neff", "fixed_half_rhomin"]:
        def bnd_at(n):
            if mode == "oracle_neff":
                ne = n * (1 - R ** 2) ** 2 / np.where(W > 0, W, 1); tau = np.tanh(fisher_tau(n, d)[1] / np.sqrt(np.maximum(ne - 3, 1)))
            else: tau = np.full((d, d), rmin / 2)
            return f1_bound(n, R, B, D, k1, k2, tau, rel)[0]
        lo, hi = 100, 100
        while bnd_at(hi) < 0.9 and hi < 1e9: hi *= 2
        lo = hi // 2
        for _ in range(30):
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if bnd_at(mid) >= 0.9 else (mid, hi)
        print(f" smallest n with F1 bound >= 0.9 ({mode}): {hi}")
    # Bartlett check at n=5000: empirical n Var(r_ab) vs W_ab (delta method)
    rng = np.random.default_rng(9000 + ci); Rh = ss.sample_stats(5000, 2000, rng)
    emp = 5000 * Rh.var(0); ratio = emp[iu] / W[iu]
    print(f" Bartlett/delta-method check n=5000, 2000 reps: empirical n*Var(r)/W over pairs: min {ratio.min():.3f} median {np.median(ratio):.3f} max {ratio.max():.3f}")
