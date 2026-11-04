"""T2 checks: block-mean aggregation vs point subsampling for a stable linear SVAR with self-loops (exact population
second moments, no sampling).  X(t) = sum_{tau=1..L} A_tau X(t-tau) + eps(t), eps ~ (0, diag(sig2)) iid.
Block mean Ybar_s = (1/D) sum_{u<D} X(sD+u);  subsample Z_s = X(sD).
Checks:
 A1 D*Cov(Ybar) = sum_{|h|<D}(1-|h|/D) Gamma(h)       (Fejer identity; exact)
 A2 ||D Cov(Ybar) - S_LR|| <= (1/D) sum_h |h| ||Gamma(h)||, S_LR = (I-A(1))^-1 Sig (I-A(1))^-T, A(1)=sum_tau A_tau
 A3 S_LR equals the covariance of the static "long-run SEM" X = Bt X + e, Bt_ji = A(1)_ji/(1-lam_j), Var e_j = sig2_j/(1-lam_j)^2,
    lam_j = A(1)_jj   (summary graph acyclic apart from self-loops)
 A4 Cov(Z) = Gamma(0) for every D (trivial) and Cov(Z_{s+1}, Z_s) = Gamma(D) -> 0
 A5 B4 star theorem applied to the long-run SEM: block-mean R2-sortability -> 0 (out-star), 1 (in-star) as D -> inf
 A6 the earlier formula Cov(Y_X, Y_Z) = b (D - tau)_+ (block SUM, research/theory/B) holds for white-noise source, fails with a self-loop
 A7 sortability curves (var, R2) for block mean vs subsampling on in-trees / out-trees / chains with self-loops
Run: OMP_NUM_THREADS=1 /usr/bin/python3 t2_aggregation.py"""
import numpy as np, sys
rng = np.random.default_rng(2026)
out = []
def P_(s): print(s); sys.stdout.flush(); out.append(s)

def companion(As):
    d = As[0].shape[0]; L = len(As); F = np.zeros((d * L, d * L))
    F[:d, :] = np.hstack(As); F[d:, :-d] = np.eye(d * (L - 1)); return F

def gammas(As, sig2, H):
    d = As[0].shape[0]; L = len(As); F = companion(As)
    assert np.max(np.abs(np.linalg.eigvals(F))) < 1
    Q = np.zeros((d * L, d * L)); Q[:d, :d] = np.diag(sig2)
    # doubling algorithm for P = F P F^T + Q
    P = Q.copy(); Fk = F.copy()
    for _ in range(60):
        P = P + Fk @ P @ Fk.T; Fk = Fk @ Fk
        if np.abs(Fk).max() < 1e-300: break
    G = [P[:d, :d].copy()]; C = P.copy()   # C = Cov(state(t+h), state(t))
    for h in range(1, H + 1):
        C = F @ C; G.append(C[:d, :d].copy())
    return G   # G[h] = E X(t+h) X(t)^T

def block_cov(G, D):   # D * Cov(Ybar)
    S = G[0].copy()
    for h in range(1, D): S += (1 - h / D) * (G[h] + G[h].T)
    return S

def r2(S):
    return 1 - 1 / (np.diag(S) * np.diag(np.linalg.inv(S)))

def sortab(c, edges):
    v = [1.0 if c[i] < c[j] - 1e-10 else (0.5 if abs(c[i] - c[j]) <= 1e-10 else 0.0) for (i, j) in edges]
    return float(np.mean(v))

def sample_model(kind, d, L=3):
    if kind == "intree": E = [(i, int(rng.integers(i + 1, d))) for i in range(d - 1)]
    elif kind == "outtree": E = [(int(rng.integers(0, j)), j) for j in range(1, d)]
    elif kind == "chain": E = [(i, i + 1) for i in range(d - 1)]
    elif kind == "outstar": E = [(0, j) for j in range(1, d)]
    elif kind == "instar": E = [(j, 0) for j in range(1, d)]
    As = [np.zeros((d, d)) for _ in range(L)]
    lam = rng.uniform(0.0, 0.9, d)
    for j in range(d):   # AR part: put self coefficient on lag 1 (stable since DAG + |lam|<1)
        As[0][j, j] = lam[j]
    for (i, j) in E:
        tau = int(rng.integers(1, L + 1)); As[tau - 1][j, i] += rng.choice([-1, 1]) * rng.uniform(0.3, 0.9)
    sig2 = rng.uniform(0.5, 2.0, d)
    return As, sig2, E

H = 4000
# ---------- A1-A4 on random in/out-trees ----------
worst_bound_ratio = 0; maxA1 = 0; maxA3 = 0
for rep in range(60):
    kind = ["intree", "outtree", "chain"][rep % 3]; d = int(rng.integers(3, 8))
    As, sig2, E = sample_model(kind, d)
    G = gammas(As, sig2, H)
    A1 = sum(As); I = np.eye(d); Minv = np.linalg.inv(I - A1)
    SLR = Minv @ np.diag(sig2) @ Minv.T
    lam = np.diag(A1); Bt = (A1 - np.diag(lam)) / (1 - lam)[:, None]; om = sig2 / (1 - lam) ** 2
    T = np.linalg.inv(I - Bt); maxA3 = max(maxA3, np.abs(T @ np.diag(om) @ T.T - SLR).max())
    K = sum(h * np.linalg.norm(G[h], 2) * 2 for h in range(1, H + 1))
    for D in [1, 2, 4, 16, 64, 256]:
        Sb = block_cov(G, D)
        # A1 by brute force: covariance of the mean of D consecutive X's
        if D <= 16:
            M = np.zeros((d, d))
            for u in range(D):
                for w in range(D):
                    h = u - w; M += G[h] if h >= 0 else G[-h].T
            maxA1 = max(maxA1, np.abs(M / D - Sb).max())
        err = np.linalg.norm(Sb - SLR, 2)
        worst_bound_ratio = max(worst_bound_ratio, err / (K / D))
P_(f"A1 Fejer identity max abs err (D<=16): {maxA1:.2e}")
P_(f"A2 max over models and D of ||D Cov(Ybar)-S_LR|| / ((1/D) sum|h| ||Gamma(h)||): {worst_bound_ratio:.3f} (must be <= 1)")
P_(f"A3 long-run SEM covariance vs S_LR max abs err: {maxA3:.2e}")
P_("A4 Cov(Z)=Gamma(0) for all D by definition; lag-1 covariance of Z is Gamma(D), geometric decay")

# ---------- A5: stars under block mean ----------
res = {"outstar": [], "instar": []}
Ds = [1, 4, 16, 64, 256, 1024]
for kind in res:
    for rep in range(200):
        d = int(rng.integers(3, 7)); As, sig2, E = sample_model(kind, d)
        G = gammas(As, sig2, 4 * max(Ds))
        A1 = sum(As); Minv = np.linalg.inv(np.eye(d) - A1); SLR = Minv @ np.diag(sig2) @ Minv.T
        row = [sortab(r2(block_cov(G, D)), E) for D in Ds] + [sortab(r2(SLR), E), sortab(r2(G[0]), E)]
        res[kind].append(row)
    M = np.array(res[kind]).mean(0)
    P_(f"A5 {kind}: mean R2-sortability, block mean D={Ds}: {np.round(M[:len(Ds)], 3).tolist()}; long-run SEM: {M[-2]:.3f}; subsample (Gamma(0)): {M[-1]:.3f}")
    P_(f"    fraction of graphs with long-run R2-sortability exactly {'0' if kind=='outstar' else '1'}: {np.mean(np.array(res[kind])[:, -2] == (0.0 if kind=='outstar' else 1.0)):.3f}")

# ---------- A6: earlier block-SUM formula ----------
for lam_src in [0.0, 0.6]:
    b, tau, D = 0.8, 2, 6
    As = [np.zeros((2, 2)) for _ in range(3)]; As[0][0, 0] = lam_src; As[tau - 1][1, 0] = b
    G = gammas(As, np.array([1.0, 1.0]), 50)
    blocksum_cov = D * block_cov(G, D)[1, 0]
    P_(f"A6 source self-loop {lam_src}: block-sum Cov = {blocksum_cov:.4f}; formula b(D-tau)_+ = {b*(D-tau):.4f}")

# ---------- A7: curves on trees (with self-loops) ----------
Ds = [1, 2, 4, 8, 16, 32, 64, 256]
for kind in ["intree", "outtree", "chain"]:
    rows_r2b, rows_r2s, rows_vb, rows_vs = [], [], [], []
    for rep in range(150):
        d = 10; As, sig2, E = sample_model(kind, d, L=4)
        G = gammas(As, sig2, 4 * max(Ds))
        rows_r2b.append([sortab(r2(block_cov(G, D)), E) for D in Ds])
        rows_vb.append([sortab(np.diag(block_cov(G, D)), E) for D in Ds])
        rows_r2s.append(sortab(r2(G[0]), E)); rows_vs.append(sortab(np.diag(G[0]), E))
    P_(f"A7 {kind} (d=10, self-loops U(0,0.9), lags<=4, 150 graphs): R2-sort block mean D={Ds}: {np.round(np.mean(rows_r2b,0),3).tolist()}; subsample (any D): {np.mean(rows_r2s):.3f}")
    P_(f"    var-sort block mean: {np.round(np.mean(rows_vb,0),3).tolist()}; subsample (any D): {np.mean(rows_vs):.3f}")
with open(__file__.replace(".py", "_results.txt"), "w") as f: f.write("\n".join(out) + "\n")
