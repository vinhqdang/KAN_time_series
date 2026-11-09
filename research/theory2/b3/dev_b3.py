"""Development only (disclosed in PLAN.md): per-edge detection of the colour-source -> colour-sensor pairs on the
study-3 light-tunnel data (lt_walks_v1/actuators_white.csv, already analysed in realdata/test3.py), with the
estimator of PLAN.md (effective-n Fisher z, K = 50, L = 0, alpha = 0.01, Bonferroni over the 14-node set's 91 pairs)
on segments of 3,000 rows.  Reads no B3 held-out data.
Usage: python3 -I dev_b3.py <actuators_white.csv>"""
import sys
import numpy as np, pandas as pd
from scipy.stats import norm
df = pd.read_csv(sys.argv[1])
col = ["red", "green", "blue"]; sens = ["ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3", "current"]
X0 = df[col + sens + ["pol_1", "pol_2"]].to_numpy(float)
K = 50; zc = norm.isf(0.01 / (2 * 91))
res = {}
nseg = len(X0) // 3000
for j in range(nseg):
    X = X0[j * 3000:(j + 1) * 3000]; X = (X - X.mean(0)) / X.std(0); n = len(X)
    A = np.array([(X[h:] * X[:n - h]).sum(0) / (X ** 2).sum(0) for h in range(1, K + 1)])
    B = np.maximum(1, 1 + 2 * A.T @ A); R = np.corrcoef(X, rowvar=False)
    T = np.abs(np.arctanh(np.clip(R, -.999999, .999999))) * np.sqrt(np.maximum(n / B - 3, 1e-9))
    for i, c in enumerate(col):
        for k, s in enumerate(sens):
            res.setdefault((c, s), []).append((R[i, 3 + k], T[i, 3 + k] > zc))
    for a in range(3):
        for b in range(a + 1, 3):
            res.setdefault((col[a], col[b]), []).append((R[a, b], T[a, b] > zc))
print(f"segments of 3,000 rows: {nseg}; z_c = {zc:.2f}")
for k, v in res.items():
    r = np.array([x[0] for x in v]); d = np.mean([x[1] for x in v])
    print(f"  {k[0]:6s} {k[1]:8s} median r = {np.median(r):+.3f}   detected {d:.2f}")
