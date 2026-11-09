"""Design check, inputs only (post-plan diagnostic, logged as deviation D1 in README): which generator variant
reproduces the recorded ar_1 actuator columns?  Tries seed index k in 0..5 and the README / generator
mean-scale variants for every actuator separately.  Reads only red, green, blue, pol_1, pol_2.
Usage: python3 -I design_check2.py <data_dir>"""
import sys, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def sample_AR_1_uniform(n, coef, random_state):          # copied verbatim from generators/actuators_ar.py
    rng = np.random.default_rng(random_state)
    innovations = rng.uniform(-1, 1, size=n)
    x = np.zeros_like(innovations); x[0] = innovations[0]
    for i in range(1, n): x[i] = x[i - 1] * coef + innovations[i]
    x -= x.min(); x /= x.max(); return (2 * x) - 1
D = sys.argv[1]
COEF = {"red": 0.5, "green": 0.6, "blue": 0.7, "pol_1": 0.8, "pol_2": 0.9}
for exp in ["ar_1_uniform_ref", "ar_1_uniform_pol_1_90", "ar_1_uniform_pol_12_30"]:
    df = pd.read_csv(os.path.join(D, exp + ".csv"), usecols=list(COEF))
    print(exp, {v: (round(float(df[v].min()), 1), round(float(df[v].max()), 1)) for v in COEF})
    for i, (v, c) in enumerate(COEF.items()):
        best = None
        for k in range(6):
            x = sample_AR_1_uniform(30000, c, 100 * k + i + 1728296067)
            y = df[v].to_numpy(float)
            # affine fit of recorded values on the regenerated series (scale/mean unknown)
            A = np.vstack([x, np.ones_like(x)]).T; coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            err = float(np.max(np.abs(A @ coef - y)))
            if best is None or err < best[0]: best = (err, k, [round(float(t), 2) for t in coef])
        print(f"   {v}: best seed index {best[1]}, max |affine residual| {best[0]:.2f}, scale/mean {best[2]}")
