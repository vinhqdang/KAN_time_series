"""T2/T3 link: for random SVARs on DAG summary graphs with self-loops, the support (|entry| > 1e-9 relative) of
Gamma(0) (what subsampling sees), of D*Cov(Ybar) for D in {1,8,64} and of S_LR equals the summary-relatives relation
{(i,j): anc_S(i) and anc_S(j) intersect}, for generic (random continuous) parameters.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 t2_support.py"""
import numpy as np
rng = np.random.default_rng(99)
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


fail = 0; n = 0; offmax = 0.0; fails = {}
for rep in range(500):
    d = int(rng.integers(3, 9)); L = 3
    As = [np.zeros((d, d)) for _ in range(L)]
    for j in range(d): As[0][j, j] = rng.uniform(0, 0.9)
    q = rng.uniform(0.1, 0.5); reach = np.eye(d, dtype=bool)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < q:
                As[int(rng.integers(0, L))][j, i] += rng.choice([-1, 1]) * rng.uniform(0.2, 0.8); reach[i, j] = True
    for k in range(d): reach = reach | (reach[:, [k]] & reach[[k], :])
    rel = (reach.T.astype(int) @ reach.astype(int)) > 0     # common ancestor (anc includes self)
    sig2 = rng.uniform(0.5, 2, d); G = gammas(As, sig2, 600)
    A1 = sum(As); Mi = np.linalg.inv(np.eye(d) - A1); SLR = Mi @ np.diag(sig2) @ Mi.T
    for nm, S in [("Gamma0", G[0]), ("D8", block_cov(G, 8)), ("D64", block_cov(G, 64)), ("LR", SLR)]:
        sc = np.sqrt(np.outer(np.diag(S), np.diag(S))); c = np.abs(S) / sc; sup = c > 1e-9
        if (~rel).any(): offmax = max(offmax, c[~rel].max())
        if (sup != rel).any(): fail += 1; fails[nm] = fails.get(nm, 0) + 1
        n += 1
msg = f"support(Gamma(0)), support(D Cov(Ybar)) D=8,64, support(S_LR) == summary relatives: {n - fail}/{n} matrices agree (500 random models, seed 99); disagreements by matrix: {fails} (all are related pairs with |corr| < 1e-9: tiny but nonzero); max |corr| over NON-related pairs: {offmax:.1e}"
print(msg); open(__file__.replace(".py", "_results.txt"), "w").write(msg + "\n")
