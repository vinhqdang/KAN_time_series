"""Candidate B: aggregation window vs R2-sortability, exact population covariances. seeds fixed.
Model: X_j(t)=sum_i b_ji X_i(t-tau_ji)+eps_j(t), eps ~ N(0,1) iid, DAG on nodes 0..d-1 (edge i->j needs i<j), integer delays.
Block-sum aggregation with window D: Sigma_Y(D)=sum_{|h|<D}(D-|h|)Gamma(h).  D=inf limit: (I-sum_edges B)^-1 (I-..)^-T (static summary SEM).
Run: OMP_NUM_THREADS=1 /usr/bin/python3 sim_B.py"""
import numpy as np, sys
rng = np.random.default_rng(12345)
K = 400  # MA truncation (delays <= 8, |b| small enough -> geometric decay; checked below)

def sample_graph(kind, d, rng, wlo=0.3, whi=0.9, tmax=8):
    E = []   # (i,j,b,tau)
    if kind == "er":
        p = 2.0 / (d - 1)
        for j in range(d):
            for i in range(j):
                if rng.random() < p: E.append((i, j))
    elif kind == "intree":      # river-like: every node except the last has exactly one child (downstream) with larger index
        for i in range(d - 1): E.append((i, rng.integers(i + 1, d)))
    elif kind == "outtree":     # every node except node 0 has exactly one parent
        for j in range(1, d): E.append((rng.integers(0, j), j))
    elif kind == "chain":
        for i in range(d - 1): E.append((i, i + 1))
    return [(i, j, rng.choice([-1, 1]) * rng.uniform(wlo, whi), int(rng.integers(1, tmax + 1))) for i, j in E]

def ma_coeffs(E, d, K):
    Psi = np.zeros((K, d, d)); Psi[0] = np.eye(d)
    for k in range(1, K):
        for (i, j, b, tau) in E:
            if k - tau >= 0: Psi[k, j, :] += b * Psi[k - tau, i, :]
    return Psi

def gamma_list(Psi, H):
    K = Psi.shape[0]
    G = []
    for h in range(H + 1):
        G.append(np.einsum("kab,kcb->ac", Psi[h:], Psi[:K - h]))   # Gamma(h)=E[X(t+h)X(t)^T]
    return G

def agg_cov(G, D):
    S = D * G[0].copy()
    for h in range(1, D): S += (D - h) * (G[h] + G[h].T)
    return S

def r2(S):
    P = np.linalg.inv(S); return 1 - 1.0 / (np.diag(S) * np.diag(P))

def sortability_edges(r, E):
    """fraction of true edges i->j with r_i<r_j (tie=1/2)"""
    s = 0
    for (i, j, b, t) in E:
        s += 1.0 if r[i] < r[j] - 1e-9 else (0.5 if abs(r[i] - r[j]) <= 1e-9 else 0.0)
    return s / len(E)

def sortability_pairs(r, E, d):
    A = np.zeros((d, d), bool)
    for (i, j, b, t) in E: A[i, j] = True
    R = A.copy()
    for _ in range(d): R = R | (R.astype(int) @ A.astype(int) > 0)
    s = n = 0
    for i in range(d):
        for j in range(d):
            if R[i, j]:
                n += 1; s += 1.0 if r[i] < r[j] - 1e-9 else (0.5 if abs(r[i] - r[j]) <= 1e-9 else 0.0)
    return s / max(n, 1)

if __name__ == "__main__":
    d = 10; Ds = [1, 2, 4, 8, 16, 32, 64]
    lines = []
    def P_(s): print(s); lines.append(s)
    # ---- Check 0: R2 closed form for contemporaneous SEM  R2_i = 1 - u_i/(1+omega_i c_i)
    mx = 0
    for rep in range(500):
        E = sample_graph("er", d, rng); B = np.zeros((d, d))
        for (i, j, b, t) in E: B[j, i] = b
        om = rng.uniform(0.3, 2, d); T = np.linalg.inv(np.eye(d) - B); S = T @ np.diag(om) @ T.T
        c = np.array([sum(B[k, i] ** 2 / om[k] for k in range(d)) for i in range(d)])
        u = om / np.diag(S)
        mx = max(mx, np.abs(r2(S) - (1 - u / (1 + om * c))).max())
    P_(f"B0 closed form R2_i = 1 - u_i/(1+omega_i c_i): max abs err over 500 random SEMs = {mx:.2e}")
    # ---- Check 1: bivariate tie, standardised: R2 equal
    b = 0.7; S = np.array([[1, b], [b, 1]]); P_(f"B1 bivariate R2: {r2(S)} (tie)")
    # ---- Check 2: standardised chain palindromic R2 profile
    d5 = 7; rho = 0.6; S = rho ** np.abs(np.subtract.outer(range(d5), range(d5))); P_("B2 chain(7) rho=.6 R2 profile: " + str(np.round(r2(S), 4)))
    # ---- Main: sortability vs window
    for kind in ("er", "intree", "outtree", "chain"):
        res = {D: [] for D in Ds + ["inf"]}; resp = {D: [] for D in Ds + ["inf"]}; tie1 = []
        R = 150
        for rep in range(R):
            E = sample_graph(kind, d, rng)
            if not E: continue
            Psi = ma_coeffs(E, d, K)
            # truncation check: spectral decay
            assert np.abs(Psi[-1]).max() < 1e-6, "truncation"
            G = gamma_list(Psi, max(Ds))
            for D in Ds:
                S = agg_cov(G, D) / D
                r = r2(S); res[D].append(sortability_edges(r, E)); resp[D].append(sortability_pairs(r, E, d))
                if D == 1: tie1.append(np.mean(np.abs(S - np.diag(np.diag(S))).max(axis=1) < 1e-12))
            B = np.zeros((d, d))
            for (i, j, b, t) in E: B[j, i] += b
            T = np.linalg.inv(np.eye(d) - B); S = T @ T.T; r = r2(S)
            res["inf"].append(sortability_edges(r, E)); resp["inf"].append(sortability_pairs(r, E, d))
        P_(f"B3 graph={kind} d={d} reps={R}: R2-sortability on true edges (pairs: ancestor-descendant) by window D (native step=1, delays U{{1..8}}):")
        P_("   D:      " + "  ".join(f"{str(D):>6}" for D in res))
        P_("   edges:  " + "  ".join(f"{np.mean(res[D]):6.3f}" for D in res))
        P_("   pairs:  " + "  ".join(f"{np.mean(resp[D]):6.3f}" for D in res))
        P_("   SE edges (max): %.3f ; fraction of nodes with zero cross-cov at D=1 (isolated => tied R2=0): %.3f" % (max(np.std(res[D]) / np.sqrt(len(res[D])) for D in res), np.mean(tie1)))
    # ---- Check 3: Delta<=min delay makes aggregated process contemporaneously uncorrelated for source->child pair (2 nodes)
    for tau in (3, 6):
        E = [(0, 1, 0.8, tau)]; Psi = ma_coeffs(E, 2, 50); G = gamma_list(Psi, 20)
        out = []
        for D in (1, 2, 3, 4, 6, 12):
            S = agg_cov(G, D) / D; out.append(f"D={D}: corr={S[0,1]/np.sqrt(S[0,0]*S[1,1]):.3f}")
        P_(f"B4 two nodes delay tau={tau}, contemporaneous correlation of aggregated data: " + "; ".join(out))
        # closed form (white-noise source): Cov(Y0,Y1)=b(D-tau)^+ (block sum); Var Y0=D; Var Y1=D(1+b^2)
        D = 12; S = agg_cov(G, D); P_(f"     closed form check D=12: cov={S[0,1]:.4f} vs b(D-tau)={0.8*max(D-tau,0):.4f}")
    # ---- Adversarial: is sortability monotone in D? count reps where D=64 < D=1 for ER
    open("results_B.txt", "w").write("\n".join(lines))
