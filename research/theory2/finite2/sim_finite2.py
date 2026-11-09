"""Fixed-seed simulation checks for FINITE2.md.  Usage: OMP_NUM_THREADS=1 /usr/bin/python3 sim_finite2.py [part]
Parts: A (consistency of the plug-in Bartlett factor), B (size of the plug-in test), C (recovery probability of
the Bonferroni Fisher-z estimator vs the best F1 guarantee), D (non-Gaussian innovations), E (B vs W example).
The estimator is the one in ../realdata/test3.py, ../icml_exp/test4.py (L = 0) and ../b3/test_b3.py:
  B_hat_ab = max(1, 1 + 2 sum_{h=1}^{K} rho_hat_a(h) rho_hat_b(h)),  rho_hat(h) = sum_{t<=n-h} Xc(t+h)Xc(t) / sum Xc^2,
  declare iff |atanh r_hat_ab| sqrt(n / B_hat_ab - 3) > z_c."""
import sys, os, math
import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "finite"))
import common as CM

def acf(X, K):
    """X (n, q) or (reps, n, q): code's ACF estimator for lags 1..K (same values as the code's loop, via FFT)."""
    Xc = X - X.mean(-2, keepdims=True); n = X.shape[-2]
    F = np.fft.rfft(Xc, n=2 * n, axis=-2); ac = np.fft.irfft(F * np.conj(F), n=2 * n, axis=-2)
    return np.take(ac, np.arange(1, K + 1), axis=-2) / np.take(ac, [0], axis=-2)

def bhat_pair(X, K, floor=True):
    A = acf(X, K); s = 1 + 2 * (A[..., 0] * A[..., 1]).sum(-1)
    return np.maximum(1.0, s) if floor else s

def corr_pair(X):
    Xc = X - X.mean(-2, keepdims=True)
    return (Xc[..., 0] * Xc[..., 1]).sum(-1) / np.sqrt((Xc[..., 0] ** 2).sum(-1) * (Xc[..., 1] ** 2).sum(-1))

def ar1(lam, n, reps, rng, innov="gauss", burn=None):
    burn = burn or int(min(20000, 10 / (1 - abs(lam)) + 50))
    e = draw(innov, (reps, n + burn), rng)
    return lfilter([1.0], [1.0, -lam], e, axis=1)[:, burn:]

def draw(innov, shape, rng):
    if innov == "gauss": return rng.standard_normal(shape)
    if innov == "exp": return rng.exponential(1.0, shape) - 1.0             # kappa4 = 6
    if innov == "laplace": return rng.laplace(0, 1 / math.sqrt(2), shape)   # unit variance, kappa4 = 3
    if innov == "t5": return rng.standard_t(5, shape) / math.sqrt(5 / 3)    # unit variance, kappa4 = 6
    raise ValueError(innov)

def W_ar(la, lb): return (1 + la * lb) / (1 - la * lb)                     # sum_h rho_a(h) rho_b(h)
def B_ar(la, lb): return (1 + abs(la * lb)) / (1 - abs(la * lb))           # sum_h |rho_a(h) rho_b(h)|
def BK_ar(la, lb, K): q = la * lb; return 1 + 2 * q * (1 - q ** K) / (1 - q)

def partA():
    print("== A. plug-in Bartlett factor: mean (sd) of B_hat (no floor) vs W; M_n = ceil(n^0.5) and fixed K = 50")
    rng = np.random.default_rng(20261109)
    for la, lb in [(0.9, 0.9), (0.8, -0.8), (0.97, 0.97), (0.99, 0.99)]:
        W = W_ar(la, lb)
        for n in [1000, 4000, 16000, 64000]:
            reps = 200 if n <= 16000 else 60
            X = np.stack([ar1(la, n, reps, rng), ar1(lb, n, reps, rng)], -1)
            M = int(math.ceil(n ** 0.5)); b1 = bhat_pair(X, M, False); b2 = bhat_pair(X, 50, False)
            print(f"lam=({la},{lb}) W={W:.3f} B_K50={BK_ar(la,lb,50):.3f} n={n:6d} M={M:4d}: "
                  f"Bhat_M {b1.mean():.3f} ({b1.std():.3f})  Bhat_50 {b2.mean():.3f} ({b2.std():.3f})", flush=True)

def partB():
    print("== B. size of the test for an independent AR(1) pair (per-pair level 2*Phibar(z)); reps 4000 (2000 at n=64000)")
    rng = np.random.default_rng(20261110)
    for z in [1.959964, 3.290527]:
        for la, lb, n in [(0.9, 0.9, 4000), (0.9, 0.9, 16000), (0.97, 0.97, 16000), (0.99, 0.99, 64000), (0.8, -0.8, 4000)]:
            reps = 4000 if n <= 16000 else 2000
            hits = {"nominal": 0, "plugin_M": 0, "plugin_K50": 0, "oracle": 0}; done = 0
            W = W_ar(la, lb); M = int(math.ceil(n ** 0.5)) if la < 0.99 else int(math.ceil(n ** 0.6))
            while done < reps:
                r = min(500 if n <= 16000 else 100, reps - done)
                X = np.stack([ar1(la, n, r, rng), ar1(lb, n, r, rng)], -1)
                at = np.abs(np.arctanh(corr_pair(X)))
                hits["nominal"] += int((at * np.sqrt(n - 3) > z).sum())
                hits["oracle"] += int((at * np.sqrt(n / max(1, W) - 3) > z).sum())
                hits["plugin_M"] += int((at * np.sqrt(n / bhat_pair(X, M) - 3) > z).sum())
                hits["plugin_K50"] += int((at * np.sqrt(n / bhat_pair(X, 50) - 3) > z).sum())
                done += r
            a = 2 * norm.sf(z); BK = max(1, BK_ar(la, lb, 50))
            pred = {"nominal": 2 * norm.sf(z / math.sqrt(W)), "oracle": 2 * norm.sf(z * math.sqrt(max(1, W) / W)),
                    "plugin_M": 2 * norm.sf(z * math.sqrt(max(1, W) / W)), "plugin_K50": 2 * norm.sf(z * math.sqrt(BK / W))}
            se = lambda p: math.sqrt(max(p, 1e-12) * (1 - p) / reps)
            print(f"z={z:.2f} (level {a:.4f}) lam=({la},{lb}) n={n} W={W:.2f} M={M}: " +
                  "  ".join(f"{k} {hits[k]/reps:.4f}[pred {pred[k]:.4f}, mc-se {se(pred[k]):.4f}]" for k in hits), flush=True)

def var_sim(A, n, reps, rng, innov="gauss", burn=500):
    d = len(A); s = np.zeros((reps, d)); out = np.empty((reps, n, d)); At = A.T
    E = draw(innov, (reps, n + burn, d), rng)
    for t in range(n + burn):
        s = s @ At + E[:, t]
        if t >= burn: out[:, t - burn] = s
    return out

def stats_full(X, K):
    """X (reps, n, d) -> R (reps,d,d), Bhat (reps,d,d) floor 1."""
    Xc = X - X.mean(1, keepdims=True); n = X.shape[1]
    C = np.einsum('rti,rtj->rij', Xc, Xc); sd = np.sqrt(np.einsum('rii->ri', C)); R = C / (sd[:, :, None] * sd[:, None, :])
    Aa = acf(X, K)                                           # (reps, K, d)
    B = np.maximum(1.0, 1 + 2 * np.einsum('rku,rkv->ruv', Aa, Aa)); return R, B

def partC():
    print("== C. recovery P(R_hat = R), d = 5 in-forest (0->2<-1, 2->3, 4 isolated), Bonferroni Fisher z")
    rng = np.random.default_rng(20261111)
    for lam, w in [(0.5, 0.5), (0.9, 0.3)]:
        d = 5; A = CM.in_forest_A({2: [0, 1], 3: [2]}, [lam] * d, w)
        g = CM.standardise_acov(CM.var1_acov(A, np.eye(d), 3000)); B, D, k1, k2, W = CM.f1_constants(g)
        Rpop = g[3000]; rel = CM.relatives(A); np.fill_diagonal(rel, False); iu = np.triu_indices(d, 1)
        nul = ~rel[iu]; Wnull = np.array([sum(g[:, a, a] * g[:, b, b]) for a, b in zip(*iu)])
        rmin = np.abs(Rpop[iu][~nul]).min()
        print(f"lam={lam} w={w}: rho_min={rmin:.4f}, null pairs {nul.sum()}/{len(nul)}, W_null range "
              f"[{Wnull[nul].min():.2f},{Wnull[nul].max():.2f}], B_null max {B[iu][nul].max():.2f}")
        for alpha in [0.05, 0.2]:
            m = d * (d - 1) / 2; zc = norm.isf(alpha / (2 * m))
            # best F1 guarantee for the oracle Fisher-z threshold (n' = n/W_null per pair; relatives n' = n/W)
            for n in [2000, 8000, 32000]:
                reps = 400; hit = {"nominal": 0, "plugin_M": 0, "plugin_K50": 0, "oracle": 0}; fp = dict(hit); done = 0
                M = int(math.ceil(n ** 0.5))
                Wo = W.copy(); Wo[np.eye(d, dtype=bool)] = 1.0
                while done < reps:
                    r = min(100, reps - done); X = var_sim(A, n, r, rng)
                    Xc = X - X.mean(1, keepdims=True); C = np.einsum('rti,rtj->rij', Xc, Xc)
                    sd = np.sqrt(np.einsum('rii->ri', C)); R = np.clip(C / (sd[:, :, None] * sd[:, None, :]), -.999999, .999999)
                    Aa_M = acf(X, M); BM = np.maximum(1.0, 1 + 2 * np.einsum('rku,rkv->ruv', Aa_M, Aa_M))
                    B50 = np.maximum(1.0, 1 + 2 * np.einsum('rku,rkv->ruv', Aa_M[:, :50], Aa_M[:, :50])) if M >= 50 else None
                    if B50 is None:
                        Aa50 = acf(X, 50); B50 = np.maximum(1.0, 1 + 2 * np.einsum('rku,rkv->ruv', Aa50, Aa50))
                    at = np.abs(np.arctanh(R))
                    # oracle: the null-pair Bartlett sum sum_h rho_a rho_b (what B_hat estimates), floor 1
                    Bor = np.ones((d, d))
                    for a in range(d):
                        for b in range(d):
                            if a != b: Bor[a, b] = max(1.0, float(sum(g[:, a, a] * g[:, b, b])))
                    for k, ne in [("nominal", np.full((d, d), float(n))), ("plugin_M", n / BM), ("plugin_K50", n / B50), ("oracle", n / Bor[None])]:
                        Dm = at * np.sqrt(np.maximum(ne - 3, 1e-9)) > zc
                        Dm = Dm[:, iu[0], iu[1]]
                        hit[k] += int(np.all(Dm == ~nul[None], 1).sum())
                        fp[k] += int(np.any(Dm[:, nul], 1).sum())
                    done += r
                # best F1 guarantee for the oracle threshold at this n
                tau = np.ones((d, d)) * 0.999
                for a, b in zip(*iu): tau[a, b] = tau[b, a] = math.tanh(zc / math.sqrt(max(n / Bor[a, b] - 3, 1)))
                f1b = CM.f1_bound(n, Rpop, B, D, k1, k2, tau, rel)[0]
                print(f"  alpha={alpha} n={n:6d}: P(R_hat=R) " + " ".join(f"{k} {hit[k]/reps:.3f}" for k in hit) +
                      " | P(any FP) " + " ".join(f"{k} {fp[k]/reps:.3f}" for k in fp) + f" | best F1 bound (oracle thr) {f1b:.3f}", flush=True)
            # limit of the best F1 guarantee for the oracle threshold (Prop 2 of FINITE2): 1 - d(d-1) exp(-zc^2 min_N max(1,W)/B / 2)
            ratio = min(max(1.0, float(sum(g[:, a, a] * g[:, b, b]))) / B[a, b] for a, b in zip(*iu) if not rel[a, b])
            print(f"  alpha={alpha}: z_c={zc:.3f}; limit of best F1 guarantee = {1 - d*(d-1)*math.exp(-zc**2*ratio/2):.3f}; "
                  f"asymptotic guarantee of Prop 3 = {1-alpha*nul.sum()/m:.3f}; independent-pairs approx {(1-alpha/m)**nul.sum():.3f}")
            for n in [10**6, 10**8]:
                f1b = CM.f1_bound(n, Rpop, B, D, k1, k2, np.where(np.eye(d, dtype=bool), .999,
                      np.tanh(zc / np.sqrt(np.maximum(n / np.where(np.eye(d,dtype=bool),1,np.maximum(1,np.einsum('hu,hv->uv', np.einsum('hii->hi', g), np.einsum('hii->hi', g)))) - 3, 1)))), rel, ys=np.linspace(0.05, 200, 800))[0]
                print(f"  alpha={alpha} n={n:.0e}: best F1 bound (oracle thr) {f1b:.4f}")

def partD():
    print("== D. non-Gaussian innovations: n Var(r_hat) vs Gaussian W and W_ng = W + sum_i kappa_i g_i^2")
    rng = np.random.default_rng(20261112)
    # (i) iid: X = L eps, rho = 0.6
    rho = 0.6; Lm = np.array([[1, 0], [rho, math.sqrt(1 - rho ** 2)]]); n, reps = 2000, 4000
    for innov, k4 in [("gauss", 0), ("laplace", 3), ("exp", 6)]:
        E = draw(innov, (reps, n, 2), rng); X = E @ Lm.T; r = corr_pair(X)
        G = [np.outer(Lm[:, i], Lm[:, i]) for i in range(2)]
        gi = [Gi[0, 1] - rho / 2 * (Gi[0, 0] + Gi[1, 1]) for Gi in G]
        Wg = (1 - rho ** 2) ** 2; Wng = Wg + k4 * sum(x ** 2 for x in gi)
        print(f"  iid rho={rho} {innov:7s}: n Var(r) = {n*r.var():.4f}  Gaussian W = {Wg:.4f}  W_ng = {Wng:.4f}", flush=True)
    # (ii) independent AR(1) pair, lam = 0.9: no cumulant term; plug-in size at z = 1.96
    n, reps = 4000, 3000
    for innov in ["exp", "t5"]:
        X = np.stack([ar1(0.9, n, reps, rng, innov), ar1(0.9, n, reps, rng, innov)], -1); r = corr_pair(X)
        at = np.abs(np.arctanh(r)); b = bhat_pair(X, int(math.ceil(n ** 0.5)))
        print(f"  independent AR(1) 0.9 {innov}: n Var(r) = {n*r.var():.3f} (W = {W_ar(.9,.9):.3f}); plug-in size at z=1.96: "
              f"{(at*np.sqrt(n/b-3) > 1.96).mean():.4f} (target 0.05, mc-se 0.004); nominal {(at*np.sqrt(n-3)>1.96).mean():.3f} "
              f"(pred {2*norm.sf(1.96/math.sqrt(W_ar(.9,.9))):.3f})", flush=True)
    # (iii) VAR(1) relatives: 0 -> 1 with lam 0.7, w 0.5, exponential innovations
    A = np.array([[0.7, 0.0], [0.5, 0.7]]); H = 400
    g = CM.standardise_acov(CM.var1_acov(A, np.eye(2), H)); _, _, _, _, W = CM.f1_constants(g)
    G0 = CM.var1_gamma0(A, np.eye(2)); s = np.sqrt(np.diag(G0)); rho = G0[0, 1] / (s[0] * s[1])
    Gi = np.zeros((2, 2, 2)); P = np.eye(2)
    for j in range(H):
        for i in range(2): Gi[i] += np.outer(P[:, i], P[:, i])
        P = P @ A
    Gi = Gi / np.outer(s, s)[None]
    gi = [Gi[i][0, 1] - rho / 2 * (Gi[i][0, 0] + Gi[i][1, 1]) for i in range(2)]
    n, reps = 4000, 3000
    for innov, k4 in [("gauss", 0), ("exp", 6)]:
        X = var_sim(A, n, reps, rng, innov); r = corr_pair(X)
        print(f"  VAR(1) relatives rho={rho:.4f} {innov:5s}: n Var(r) = {n*r.var():.4f}  Gaussian W = {W[0,1]:.4f}  "
              f"W_ng = {W[0,1] + k4*sum(x**2 for x in gi):.4f}", flush=True)

def partE():
    print("== E. B_ab vs W_ab for independent AR(1) pairs")
    for la, lb in [(0.8, 0.8), (0.8, -0.8), (0.9, -0.5), (0.5, 0.3)]:
        print(f"  lam=({la},{lb}): W = sum rho_a rho_b = {W_ar(la,lb):.4f}, B = sum |rho_a rho_b| = {B_ar(la,lb):.4f}")

if __name__ == "__main__":
    parts = sys.argv[1] if len(sys.argv) > 1 else "EADBC"
    for p in parts: {"A": partA, "B": partB, "C": partC, "D": partD, "E": partE}[p]()
