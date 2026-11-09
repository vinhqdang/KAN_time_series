"""Pre-data informativeness of the F1/F2 bounds at CausalRivers-like sample sizes (FINITE.md Section 4).
Synthetic constants: every series AR(1)-like with persistence lam (B_null = (1+lam^2)/(1-lam^2), D = 2 B_null,
kappa1 = (1+lam)/(1-lam), kappa2 = 2 kappa1; B_ab := 2 B_null is a synthetic stand-in, not a bound).  d = 5 and 10, one pair class.
Run: /usr/bin/python3 f_informative.py > f_informative_results.txt"""
import numpy as np
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # sibling modules also under python -I
from common import f1_bound, fisher_tau

def consts(lam, d):
    Bn = (1 + lam ** 2) / (1 - lam ** 2); k1 = (1 + lam) / (1 - lam)
    return np.full((d, d), 2 * Bn), 2 * Bn, k1, 2 * k1      # synthetic: B_ab := 2 B_null

def nmin(fun, target=0.9):
    hi = 50
    while fun(hi) < target and hi < 2e9: hi *= 2
    if hi >= 2e9: return np.inf
    lo = hi // 2
    for _ in range(40):
        mid = (lo + hi) // 2; lo, hi = (lo, mid) if fun(mid) >= target else (mid, hi)
    return hi

print("smallest n at which the F1 lower bound on P(Rhat = R) reaches 0.9 (fixed threshold rho_min/2; half the pairs related)")
print(" d | lam | rho_min | n_bound | n_eff=n/B_null at that n")
for d in [5, 10]:
    rel = np.zeros((d, d), bool); rel[: d // 2, : d // 2] = True; np.fill_diagonal(rel, False)
    for lam in [0.5, 0.9, 0.97, 0.99]:
        B, D, k1, k2 = consts(lam, d); Bn = (1 + lam ** 2) / (1 - lam ** 2)
        for rm in [0.1, 0.3, 0.6]:
            R = np.where(rel, rm, 0.0); np.fill_diagonal(R, 1)
            n = nmin(lambda n: f1_bound(n, R, B, D, k1, k2, np.full((d, d), rm / 2), rel)[0])
            print(f" {d:2d} | {lam} | {rm} | {n} | {n / Bn:.0f}")
print("\nsmallest n at which the F2 saturation bound (all pairs declared, nominal Fisher-z, alpha=0.01 Bonferroni) reaches 0.9,"
      " vs the asymptotic half-point 3 + (z_c/atanh kappa)^2")
print(" d | lam | kappa | n_bound | n_1/2 (asympt., independent of lam)")
for d in [5, 10]:
    for lam in [0.5, 0.9, 0.97, 0.99]:
        B, D, k1, k2 = consts(lam, d)
        for kap in [0.1, 0.3, 0.64]:
            R = np.full((d, d), kap); np.fill_diagonal(R, 1)
            n = nmin(lambda n: f1_bound(n, R, B, D, k1, k2, np.full((d, d), fisher_tau(n, d)[0]), np.ones((d, d), bool))[0])
            zc = fisher_tau(100, d)[1]
            print(f" {d:2d} | {lam} | {kap} | {n} | {3 + (zc / np.arctanh(kap)) ** 2:.0f}")
