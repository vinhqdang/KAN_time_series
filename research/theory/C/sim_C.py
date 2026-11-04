"""Candidate C: forecast-risk selection vs causal parents. Exact population computations (Lyapunov), seeds fixed.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 sim_C.py"""
import numpy as np, itertools
from scipy.linalg import solve_discrete_lyapunov as lyap
rng = np.random.default_rng(2024)
lines = []
def P(s): print(s); lines.append(s)

def system(M0, M1, noise_var):
    n = len(M0); T = np.linalg.inv(np.eye(n) - M0); Phi = T @ M1
    Q = T @ np.diag(noise_var) @ T.T
    S0 = lyap(Phi, Q); C1 = Phi @ S0        # C1 = Cov(Z_t, Z_{t-1})
    return Phi, S0, C1

def risk(S0, C1, obs, j, supp):
    """population one-step MSE of the best linear predictor of X_j(t) from X_supp(t-1)"""
    s = list(supp); vj = S0[j, j]
    if not s: return vj
    Sss = S0[np.ix_(s, s)]; c = C1[j, s]
    return vj - c @ np.linalg.solve(Sss, c)

# ---- C1: contemporaneous edges are invisible; Bayes lag-1 predictor support = supp(Phi) strictly larger than lag-1 parents
d = 6; extra = []; gaps = []; same = 0
for rep in range(2000):
    A0 = np.tril(rng.normal(size=(d, d)) * 0.5 * (rng.random((d, d)) < 0.3), -1)
    A1 = rng.normal(size=(d, d)) * 0.4 * (rng.random((d, d)) < 0.2)
    if max(abs(np.linalg.eigvals(np.linalg.inv(np.eye(d) - A0) @ A1))) > 0.9: continue
    Phi, S0, C1 = system(A0, A1, np.ones(d))
    # invisibility: any other A0' with same Phi gives the same risk; construct A0'=0, A1'=Phi
    Phi2, S02, C12 = system(np.zeros((d, d)), Phi, np.ones(d))
    # (noise covariance differs: Q' = I vs T T^T) so use matched innovations covariance instead:
    for j in range(d):
        parents = [i for i in range(d) if A1[j, i] != 0]
        pstar = [i for i in range(d) if abs(Phi[j, i]) > 1e-12]
        extra.append(len(set(pstar) - set(parents))); 
        gaps.append(risk(S0, C1, None, j, parents) - risk(S0, C1, None, j, range(d)))
extra = np.array(extra); gaps = np.array(gaps)
P(f"C1 random SVAR(d=6, contemporaneous+lag-1): over {len(extra)} node-targets, mean #non-parent lag-1 predictors in supp(Phi) = {extra.mean():.2f}; share of targets with >=1 spurious predictor = {(extra>0).mean():.2f}")
P(f"C1 excess forecast risk of using ONLY the true lag-1 parent set vs best lag-1 predictor: mean {gaps.mean():.4f}, share with gap>1e-6: {(gaps>1e-6).mean():.2f}")
# exact invisibility: two SVARs with different A0 but identical Phi and identical risk
d2 = 3
A0 = np.array([[0, 0, 0], [0.8, 0, 0], [0, 0.0, 0]]); A1 = np.array([[0.5, 0, 0], [0, 0, 0], [0, 0, 0.3]])
Phi = np.linalg.inv(np.eye(3) - A0) @ A1
nv = np.ones(3); _, S0a, C1a = system(A0, A1, nv)
# alternative: no contemporaneous edge but A1' with Phi' = Phi (lag edge 1->2 instead of contemporaneous); noise covariance must match: Cov(T eps) -> use diag noise only if T=I; here
# keep identical reduced form by construction: Z_t = Phi Z_{t-1} + u_t with u = T eps (correlated innovations). Risk depends only on (Phi, Cov(u)):
Q = np.linalg.inv(np.eye(3)-A0) @ np.diag(nv) @ np.linalg.inv(np.eye(3)-A0).T
S0b = lyap(Phi, Q); C1b = Phi @ S0b
P(f"C1b two different causal structures (contemporaneous edge 1->2 vs correlated innovations) share Phi and Cov(u): max |risk diff| over all supports of node 2 = "
  f"{max(abs(risk(S0a,C1a,None,1,s)-risk(S0b,C1b,None,1,s)) for r_ in range(4) for s in itertools.combinations(range(3), r_)):.2e}; Phi[1,:]={np.round(Phi[1],3)} (lag parent set of node 2 is EMPTY in A1, Phi says {{0}})")

# ---- C2: hidden shared forcing -> risk-optimal sparse predictor non-causal (explicit counterexample)
# nodes: X0 (cause), X1 (non-cause), X2 (target). True: X0 -> X2 lag 1 with weight a. Hidden H (AR rho) loads on X1 and X2 (contemporaneously), not on X0.
def hidden_example(a, rho, h1, h2, s1, s2, s0=1.0):
    # state Z=[X0,X1,X2,H]; M0: X1=h1*H(t)+e, X2=h2*H(t)+a*X0(t-1)+e, X0=e ; H=rho*H(t-1)+eta
    M0 = np.zeros((4, 4)); M0[1, 3] = h1; M0[2, 3] = h2
    M1 = np.zeros((4, 4)); M1[2, 0] = a; M1[3, 3] = rho
    nvv = np.array([s0, s1, s2, 1.0])
    return system(M0, M1, nvv)
a, rho, h1, h2, s1, s2 = 0.3, 0.9, 1.0, 1.0, 0.1, 0.1
Phi, S0, C1 = hidden_example(a, rho, h1, h2, s1, s2)
obs = [0, 1, 2]
S0o = S0[np.ix_(obs, obs)]; C1o = C1[np.ix_(obs, obs)]
tab = {}
for k in range(0, 4):
    for s in itertools.combinations(obs, k):
        tab[s] = risk(S0o, C1o, obs, 2, s)
P("C2 hidden-forcing counterexample (a=0.3 true edge X0->X2 lag1; H AR(1) rho=0.9 loads X1,X2; noise sd small): population one-step MSE of target X2 by predictor support (from X(t-1)):")
for s, v in sorted(tab.items(), key=lambda kv: kv[1]): P(f"     support {s}: MSE {v:.4f}")
best1 = min((s for s in tab if len(s) == 1), key=lambda s: tab[s])
P(f"C2 best 1-parent predictor = {best1}  (true causal lag parent is (0,)) -> forecast-risk selection under budget 1 picks the NON-CAUSAL node; gap to true-parent model: {tab[(0,)]-tab[best1]:.4f}")
# unconstrained penalised selection: smallest lambda (per-edge cost) at which spurious X1 is dropped
Rfull = tab[(0, 1, 2)]; Rno1 = tab[(0, 2)]; Rno1b = tab[(0,)]
P(f"C2 population MSE reduction from ADDING spurious X1(t-1) to {{X0,X2}}: {Rno1 - Rfull:.4f}; to {{X0}}: {Rno1b - tab[(0,1)]:.4f}")
# and 'forecast-optimal graph' generically contains non-parents: scan over parameters for conditions
cnt = tot = 0
for rep in range(5000):
    a = rng.uniform(0.1, 0.8); rho = rng.uniform(0.3, 0.98); h1 = rng.uniform(0.2, 1.5); h2 = rng.uniform(0.2, 1.5)
    s1 = rng.uniform(0.05, 1); s2 = rng.uniform(0.05, 1)
    Phi, S0, C1 = hidden_example(a, rho, h1, h2, s1, s2)
    S0o = S0[np.ix_(obs, obs)]; C1o = C1[np.ix_(obs, obs)]
    r0 = risk(S0o, C1o, obs, 2, (0,)); r1 = risk(S0o, C1o, obs, 2, (1,))
    tot += 1; cnt += r1 < r0
P(f"C2b random parameter scan (5000 draws): share where non-causal X1 is the better 1-parent predictor of X2 than the true parent X0: {cnt/tot:.2f}")
# no-hidden control: remove H (h1=h2=0) -> true parent must win
Phi, S0, C1 = hidden_example(0.3, 0.9, 0.0, 0.0, 0.1, 0.1); S0o = S0[np.ix_(obs, obs)]; C1o = C1[np.ix_(obs, obs)]
P(f"C2c control without hidden forcing: MSE support(0,)={risk(S0o,C1o,obs,2,(0,)):.4f}, (1,)={risk(S0o,C1o,obs,2,(1,)):.4f}, (2,)={risk(S0o,C1o,obs,2,(2,)):.4f}")

# ---- C3: sufficiency (A0=0, no hidden): minimal risk-minimising support == true lag parents. exhaustive over supports
bad = 0; N = 600; d = 5
for rep in range(N):
    A1 = rng.normal(size=(d, d)) * 0.5 * (rng.random((d, d)) < 0.3)
    if max(abs(np.linalg.eigvals(A1))) > 0.9: continue
    Phi, S0, C1 = system(np.zeros((d, d)), A1, rng.uniform(0.3, 2, d))
    for j in range(d):
        rs = {s: risk(S0, C1, None, j, s) for k in range(d + 1) for s in itertools.combinations(range(d), k)}
        rmin = min(rs.values()); opt = [s for s, v in rs.items() if v <= rmin + 1e-10]
        minimal = min(opt, key=len)
        if set(minimal) != {i for i in range(d) if A1[j, i] != 0} or len([s for s in opt if len(s) == len(minimal)]) != 1: bad += 1
P(f"C3 sufficiency check (A0=0, no hidden, random VAR(1), d=5, exhaustive supports): targets where the unique minimal risk-minimiser != true parents: {bad} of {N*d} (theory: 0)")
# ---- C4: near-ties: risk gap between true and a non-causal alternative can be arbitrarily small -> finite-sample indistinguishability
for s1v in (0.1, 0.5, 2.0):
    Phi, S0, C1 = hidden_example(0.3, 0.9, 1.0, 1.0, s1v, s1v)
    S0o = S0[np.ix_(obs, obs)]; C1o = C1[np.ix_(obs, obs)]
    P(f"C4 proxy noise s={s1v}: MSE(true parent X0)={risk(S0o,C1o,obs,2,(0,)):.4f}  MSE(proxy X1)={risk(S0o,C1o,obs,2,(1,)):.4f}")
open("results_C.txt", "w").write("\n".join(lines))
