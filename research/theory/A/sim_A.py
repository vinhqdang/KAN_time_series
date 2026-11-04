"""Candidate A: do-effect error propagation. Seeds fixed. Run: OMP_NUM_THREADS=1 /usr/bin/python3 sim_A.py"""
import numpy as np, itertools, csv, collections
rng = np.random.default_rng(0)
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# ---- Test 1: exact resolvent identity  T_hat - T = T_hat (B_hat - B) T  and path-sum bound (linear)
maxerr = 0; worst_ratio = 0; ratios = []
for rep in range(2000):
    d = rng.integers(4, 12)
    B = np.tril(rng.normal(size=(d, d)) * (rng.random((d, d)) < 0.35), -1)   # B[j,i]: i->j (i<j)
    Bh = B + np.tril(rng.normal(size=(d, d)) * 0.1 * (rng.random((d, d)) < 0.5), -1)
    I = np.eye(d); T = np.linalg.inv(I - B); Th = np.linalg.inv(I - Bh)
    maxerr = max(maxerr, np.abs((Th - T) - Th @ (Bh - B) @ T).max())
    # path-sum bound: |dT_ki| <= sum_{a->b} |dB_ba| * (|T_hat|_{k b}) (|T|_{a i}) with abs matrices (Neumann of |B|)
    Ta = np.linalg.inv(I - np.abs(Bh)); Tb = np.linalg.inv(I - np.abs(B))
    bound = Ta @ np.abs(Bh - B) @ Tb
    viol = (np.abs(Th - T) - bound).max(); worst_ratio = max(worst_ratio, viol)
    m = np.abs(Th - T) > 1e-12; ratios += list((bound[m] / np.abs(Th - T)[m]))
P("T1 resolvent identity max abs error:", f"{maxerr:.2e}", "| max violation of abs-path bound:", f"{worst_ratio:.2e}")
P("T1 bound looseness (bound/actual) median %.2f, 90pct %.2f, max %.1f" % (np.median(ratios), np.percentile(ratios, 90), max(ratios)))

# ---- Test 2: uniform amplitude shrinkage alpha: relative error of path effect = 1 - alpha^m (exact on a path); chain simulation
for alpha in (0.44, 0.7):
    for m in (1, 2, 3, 4):
        b = 0.8
        B = np.zeros((m + 1, m + 1))
        for k in range(m): B[k + 1, k] = b
        T = np.linalg.inv(np.eye(m + 1) - B); Th = np.linalg.inv(np.eye(m + 1) - alpha * B)
        rel = 1 - Th[m, 0] / T[m, 0]
        P(f"T2 alpha={alpha} path length m={m}: relative error {rel:.4f}  predicted 1-alpha^m={1-alpha**m:.4f}")

# ---- Test 3: group-lasso orthonormal shrinkage lemma: argmin 0.5||y-b||^2+lam||b|| = (1-lam/||y||)_+ y
from scipy.optimize import minimize
y = np.array([1.0, -2.0, 0.5]); lam = 0.7
f = lambda b: 0.5 * ((y - b) ** 2).sum() + lam * np.linalg.norm(b)
r = minimize(f, y * 0.3, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=20000))
P("T3 group-soft-threshold: numeric", np.round(r.x, 5), "closed form", np.round(max(0, 1 - lam / np.linalg.norm(y)) * y, 5))

# ---- Test 4: nonlinear pathwise bound (sup-norm eps_j on interventional range, L_ij): random SEMs
def make_sem(d, rng):
    pa = {j: [i for i in range(j) if rng.random() < 0.4] for j in range(d)}
    par = {(j, i): (rng.normal(), rng.uniform(0.5, 2)) for j in pa for i in pa[j]}   # a*tanh(w x)
    return pa, par
viol = 0; tight = []
for rep in range(300):
    d = 8; pa, par = make_sem(d, rng)
    # learned: a' tanh(w' x) + extra smooth perturbation of sup size eps_j on |x|<=R
    pert = {k: (rng.normal() * 0.05, rng.uniform(0.5, 3)) for k in par}
    def g(j, vals, learned):
        s = 0
        for i in pa[j]:
            a, w = par[(j, i)]
            s += a * np.tanh(w * vals[i])
            if learned:
                c, om = pert[(j, i)]; s += c * np.sin(om * vals[i])
        return s
    s_node = 0; xs = 1.5
    eps_j = {j: sum(abs(pert[(j, i)][0]) for i in pa[j]) for j in range(d)}      # sup-norm error (|sin|<=1)
    L = {k: abs(par[k][0]) * par[k][1] + abs(pert[k][0]) * pert[k][1] for k in par}  # Lipschitz of learned edge fn
    # bound recursion: Delta_j <= eps_j + sum_i L_ji Delta_i  (Delta_s = 0)
    Dl = np.zeros(d)
    for j in range(d):
        if j == s_node: continue
        Dl[j] = eps_j[j] + sum(L[(j, i)] * Dl[i] for i in pa[j])
    for _ in range(20):
        e = rng.normal(size=d) * 0.5
        v = np.zeros(d); vh = np.zeros(d)
        for j in range(d):
            if j == s_node: v[j] = vh[j] = xs; continue
            v[j] = g(j, v, False) + e[j]; vh[j] = g(j, vh, True) + e[j]
        dv = np.abs(v - vh)
        viol += (dv > Dl + 1e-12).sum()
        for j in range(d):
            if j != s_node and Dl[j] > 0 and dv[j] > 0: tight.append(Dl[j] / dv[j])
P("T4 nonlinear path-Lipschitz bound: violations", viol, "| looseness bound/actual median %.2f, 90pct %.1f" % (np.median(tight), np.percentile(tight, 90)))

# ---- Test 5: counterexample: right structure, tiny observational error, huge do error (support / covariate shift)
# X1 ~ N(0,1); X2 = X1 + e. learned g2(x) = x + delta*exp(-(x-x0)^2/(2 s^2)) with x0 = 6 (far outside obs. support)
x0, s = 6.0, 0.3
for delta in (1.0, 10.0, 100.0):
    xs_grid = np.linspace(-10, 10, 400001); dens = np.exp(-xs_grid ** 2 / 2) / np.sqrt(2 * np.pi)
    h = delta * np.exp(-(xs_grid - x0) ** 2 / (2 * s ** 2))
    L2obs = np.sqrt(np.trapz(h ** 2 * dens, xs_grid))
    P(f"T5 delta={delta}: obs L2 error {L2obs:.3e}, error at do(X1=6): {h[np.argmin(abs(xs_grid-6))]:.1f}  (Lipschitz const of error bump ~ {delta/s*0.61:.1f})")

# ---- Test 6: adversarial: heterogeneous shrinkage and cancellation -> relative error > 1 (sign flip), law 1-alpha^m fails for multi-path
# X1 -> X3 direct b13 = 1 ; X1 -> X2 -> X3 with b12=1, b23=-1 ; total effect 0 ; shrink direct path alpha_d, indirect alpha_i
for ad, ai in ((0.5, 0.5), (0.5, 0.9), (0.9, 0.5)):
    tot = 1 * ad + (1 * ai) * (-1 * ai)       # path products with per-edge shrinkage
    P(f"T6 cancel: alpha_direct={ad}, alpha_edge(indirect)={ai}: learned total {tot:+.3f} vs true 0.000; effect-size NRMSE undefined (true 0). direct-only effect error {1-ad:.2f}")
open("results_A.txt", "w").write("\n".join(out))

# ---- Validation on repo data: SSC (SPADE-scaled) do_direct / do_indirect vs 1-slope^m
import pandas as pd
df = pd.read_csv("../../spire/results/do_all.csv")
ssc = df[df.method == "SSC"]
g_ = ssc.groupby("cell")[["slope", "do_direct", "do_indirect"]].mean()
g_["pred_m1"] = 1 - g_.slope; g_["pred_m2"] = 1 - g_.slope ** 2; g_["pred_m3"] = 1 - g_.slope ** 3
print(g_.round(3).to_string())
open("results_A_repo_validation.txt", "w").write(g_.round(3).to_string())
for m in ("SSR_k1.5",):
    h = df[df.method == m].groupby("cell")[["slope", "do_direct", "do_indirect"]].mean()
    print(m); print(h.round(3).to_string())
