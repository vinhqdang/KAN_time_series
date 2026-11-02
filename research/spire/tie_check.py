"""Post-hoc diagnostic (not pre-registered): is BASIL's AUROC loss due to ties among near-zero edge probabilities?"""
import sys, os
R = "/home/user/KAN_time_series"; sys.path[:0] = [R, R + "/scripts", R + "/research/spire"]
import numpy as np
from bench import gen
from basil import fit_basil
from sklearn.metrics import roc_auc_score
rows = []
for n in (1000, 2000):
    for s in range(1000, 1010):
        X, A, f, sd = gen(n, 10, s, "unif"); Xz = ((X - X.mean(0)) / X.std(0)); d = 10
        o = fit_basil(Xz, 0, g=float(n), pi=2.0 / (d - 1)); P = o["P"]; m = ~np.eye(d, dtype=bool)
        strength = np.array([[np.std(o["mean_curve"](c, e)) if c != e else 0 for e in range(d)] for c in range(d)])
        a0 = roc_auc_score(A[m], P[m]); a1 = roc_auc_score(A[m], (P + 1e-3 * strength / strength.max())[m])
        rows.append((n, a0, a1, float((P[m] < 1e-3).mean())))
r = np.array(rows)
for n in (1000, 2000):
    q = r[r[:, 0] == n]; print(f"n={n}: AUROC probs {q[:,1].mean():.3f}  with strength tie-break {q[:,2].mean():.3f}  share of pairs with P<1e-3: {q[:,3].mean():.2f}")
