"""Post hoc, descriptive (not confirmatory; no verdict of B3 is re-evaluated): sensitivity of the plug-in Bartlett
factor B_hat = max(1, 1 + 2 sum_{h<=K} rho_a(h) rho_b(h)) to the truncation lag K on the B3 primary data
(ar_1_uniform_*, 30 segments of 3,000 rows, K = 50 in the pre-registered test).  For each unit and pair: B_hat at
K = 50 (used), K = ceil(n^0.5) = 55, K = ceil(n^0.6) = 122, K = 300, and whether the declared status
(|atanh r| sqrt(n/B_hat - 3) > z_c, alpha = 0.01, Bonferroni over the pairs of the full node set) changes.
Usage: OMP_NUM_THREADS=1 /usr/bin/python3 -I k_sensitivity_b3.py <csv_dir> <b3_code_dir>"""
import sys, os, math
import numpy as np, pandas as pd
from scipy.stats import norm
DATA, B3 = sys.argv[1:3]
sys.path.insert(0, B3); import graphs_b3 as G

def acf(X, K):
    Xc = X - X.mean(0); n = len(X); g0 = (Xc ** 2).sum(0)
    return np.array([(Xc[h:] * Xc[:n - h]).sum(0) / g0 for h in range(1, K + 1)])

nodes = G.ar1_system()[0]; Ks = [50, 55, 122, 300]
rat = {K: [] for K in Ks}; flips = {K: 0 for K in Ks}; npairs = 0; rawW = {K: [] for K in Ks}
for exp in ["ar_1_uniform_ref", "ar_1_uniform_pol_1_90", "ar_1_uniform_pol_12_30"]:
    X = pd.read_csv(os.path.join(DATA, exp + ".csv"), usecols=nodes)[nodes].to_numpy(float)
    for j in range(len(X) // 3000):
        U = X[j * 3000:(j + 1) * 3000]; U = U[np.all(np.isfinite(U), 1)]
        keep = U.std(0) > 0; U = (U[:, keep] - U[:, keep].mean(0)) / U[:, keep].std(0); n, d = U.shape
        iu = np.triu_indices(d, 1); m = d * (d - 1) / 2; zc = norm.isf(0.01 / (2 * m))
        R = np.clip(np.corrcoef(U, rowvar=False), -.999999, .999999)[iu]; A = acf(U, max(Ks))
        dec = {}
        for K in Ks:
            S = 1 + 2 * np.einsum('ku,kv->uv', A[:K], A[:K]); B = np.maximum(1, S)[iu]; rawW[K] += list(S[iu])
            dec[K] = np.abs(np.arctanh(R)) * np.sqrt(np.maximum(n / B - 3, 1e-9)) > zc; rat[K] += list(B)
        for K in Ks: flips[K] += int((dec[K] != dec[50]).sum())
        npairs += len(iu[0])
b50 = np.array(rat[50])
print(f"B3 ar_1: {npairs} unit-pairs (30 units)")
for K in Ks:
    b = np.array(rat[K]); r = b / b50
    print(f"K={K:3d}: median B_hat {np.median(b):.2f} (q90 {np.quantile(b,.9):.2f}, max {b.max():.1f}); "
          f"ratio B_hat(K)/B_hat(50): median {np.median(r):.3f}, q10 {np.quantile(r,.1):.3f}, q90 {np.quantile(r,.9):.3f}; "
          f"share of unfloored sums < 1: {np.mean(np.array(rawW[K]) < 1):.3f}; declared status changed vs K=50: {flips[K]} of {npairs}")
