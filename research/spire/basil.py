"""BASIL: Bayesian spline additive DAG learner (research/04_bayes_formulation.md).
g-prior closed-form marginal likelihood per (node, parent set) + order MCMC."""
import itertools, math, numpy as np, torch
from src.cdkan.causal_kan import CausalKANInstant


def whitened_features(X, G=8):
    n, d = X.shape
    helper = CausalKANInstant(d, grid_size=G)
    Bs = helper._bases(torch.as_tensor(X, dtype=torch.float64)).double().numpy()     # [n,d,K]
    Bc = Bs - Bs.mean(0, keepdims=True); K = Bc.shape[2]; Kc = K - 1
    Z = np.zeros((n, d, Kc))
    for j in range(d):
        ev, U = np.linalg.eigh(Bc[:, j].T @ Bc[:, j] / n)
        ev, U = ev[::-1][:Kc], U[:, ::-1][:, :Kc]
        Z[:, j] = Bc[:, j] @ (U / np.sqrt(ev + 1e-9))
    return Z, Kc


def lse(a):
    m = a.max(); return m + math.log(np.exp(a - m).sum())


def fit_basil(X, seed=0, g=None, pi=None, kmax=3, iters=2500, burn=500, chains=4, thin=5, n_curve_draws=100, keep_curves=True):
    rng = np.random.RandomState(seed); X = np.asarray(X, np.float64); n, d = X.shape
    g = float(n if g is None else g); pi = (2.0 / (d - 1)) if pi is None else pi; rho = math.log(pi / (1 - pi))
    Z, Kc = whitened_features(X); F = Z.reshape(n, d * Kc); Gm = F.T @ F; Y = (X - X.mean(0)) / X.std(0)
    FY = F.T @ Y                                                                      # [dKc, d]
    tabs = []                                                                         # per node: masks, logscore, member matrix, sets
    for i in range(d):
        cand = [j for j in range(d) if j != i]; masks, ls, mem, sets = [], [], [], []
        for m in range(0, kmax + 1):
            combos = np.zeros((1, 0), int) if m == 0 else np.array(list(itertools.combinations(cand, m)), dtype=int)
            S = len(combos)
            if m == 0: r2 = np.zeros(1)
            else:
                idx = (combos[:, :, None] * Kc + np.arange(Kc)).reshape(S, -1)
                Gs = Gm[idx[:, :, None], idx[:, None, :]] + 1e-9 * np.eye(m * Kc); c = FY[idx, i]
                r2 = np.clip(np.einsum("sp,sp->s", c, np.linalg.solve(Gs, c[..., None])[..., 0]) / (Y[:, i] @ Y[:, i]), 0, 1 - 1e-9)
            p = m * Kc
            sc = (n - 1 - p) / 2 * math.log(1 + g) - (n - 1) / 2 * np.log(1 + g * (1 - r2)) + m * rho
            for s in range(S):
                sets.append(tuple(combos[s])); masks.append(sum(1 << int(j) for j in combos[s]))
            ls.append(sc)
        ls = np.concatenate(ls); masks = np.array(masks, dtype=np.int64)
        M = np.zeros((len(sets), d)); 
        for s, st in enumerate(sets):
            for j in st: M[s, j] = 1.0
        tabs.append((masks, ls, M, sets))

    def node_lse(i, pm):
        masks, ls, _, _ = tabs[i]; return lse(ls[(masks & ~pm) == 0])

    def run_chain(order, cseed):
        r = np.random.RandomState(cseed); order = list(order); full = (1 << d) - 1
        def predmask(pos):  # bitmask of nodes before position
            pm = 0
            for k in range(pos): pm |= 1 << order[k]
            return pm
        ns = [node_lse(order[k], predmask(k)) for k in range(d)]; tot = sum(ns); trace = []; samples = []
        for it in range(iters + burn):
            a, b = sorted(r.choice(d, 2, replace=False)); old = order.copy(); oldns = ns.copy()
            order[a], order[b] = order[b], order[a]
            new = ns.copy()
            for k in range(a, b + 1): new[k] = node_lse(order[k], predmask(k))
            if math.log(r.rand() + 1e-300) < sum(new) - tot: ns = new; tot = sum(new)
            else: order = old
            trace.append(tot)
            if it >= burn and (it - burn) % thin == 0: samples.append(order.copy())
        return samples, np.array(trace[burn:])

    P = np.zeros((d, d)); Wset = [np.zeros(len(t[1])) for t in tabs]; nsamp = 0; traces = []; Pc = []
    for ch in range(chains):
        order0 = list(rng.permutation(d)); samples, tr = run_chain(order0, seed * 100 + ch); traces.append(tr); Pch = np.zeros((d, d))
        for order in samples:
            pm = 0
            for k in range(d):
                i = order[k]; masks, ls, M, _ = tabs[i]; v = (masks & ~pm) == 0
                w = np.zeros(len(ls)); wv = np.exp(ls[v] - lse(ls[v])); w[v] = wv
                Pch[:, i] += w @ M; Wset[i] += w; pm |= 1 << i
            nsamp += 1
        Pc.append(Pch / len(samples)); P += Pch
    P /= nsamp; Wset = [w / nsamp for w in Wset]
    # R-hat of order log-score across chains
    L = np.array(traces); Lm = L.mean(1); B = L.shape[1] * Lm.var(ddof=1); Wv = L.var(1, ddof=1).mean(); rhat = math.sqrt(((L.shape[1] - 1) / L.shape[1] * Wv + B / L.shape[1]) / (Wv + 1e-12))
    def order_score(order):
        pm = 0; tot = 0.0
        for i in order: tot += node_lse(i, pm); pm |= 1 << int(i)
        return tot
    out = dict(order_score=order_score, last_order=samples[-1], P=P, rhat=rhat, chain_gap=float(max(np.abs(Pc[a] - Pc[b]).max() for a in range(chains) for b in range(a))))
    if keep_curves:
        cache = {}
        def post(i, s_idx):
            key = (i, s_idx)
            if key in cache: return cache[key]
            st = tabs[i][3][s_idx]; idx = (np.array(st, int)[:, None] * Kc + np.arange(Kc)).reshape(-1) if st else np.zeros(0, int)
            if len(idx) == 0: cache[key] = (st, idx, None, None, None); return cache[key]
            Gs = Gm[np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx)); u = np.linalg.solve(Gs, FY[idx, i]); r2 = float(FY[idx, i] @ u / (Y[:, i] @ Y[:, i]))
            s2 = (Y[:, i] @ Y[:, i]) * (1 - g / (1 + g) * r2) / max(n - 3, 1)
            cov = s2 * g / (1 + g) * np.linalg.inv(Gs); cache[key] = (st, idx, g / (1 + g) * u, cov, s2); return cache[key]
        def top_sets(i, mass=0.99, maxn=40):
            o = np.argsort(-Wset[i])[:maxn]; c = np.cumsum(Wset[i][o]); return o[: int(np.searchsorted(c, mass)) + 1]
        def mean_curve(c, e):
            f = np.zeros(n)
            for s_idx in top_sets(e, ):
                st, idx, mu, cov, _ = post(e, s_idx)
                if c in st:
                    k = st.index(c); f += Wset[e][s_idx] * (Z[:, c, :] @ mu[k * Kc:(k + 1) * Kc])
            return f
        def band(c, e, draws=n_curve_draws):
            ts = top_sets(e); w = Wset[e][ts] / Wset[e][ts].sum(); out_ = np.zeros((draws, n))
            for b in range(draws):
                s_idx = ts[rng.choice(len(ts), p=w)]; st, idx, mu, cov, _ = post(e, s_idx)
                if mu is not None and c in st:
                    k = st.index(c); sl = slice(k * Kc, (k + 1) * Kc)
                    u = rng.multivariate_normal(mu, (cov + cov.T) / 2, check_valid="ignore")[sl]; out_[b] = Z[:, c, :] @ u
            return np.percentile(out_, [5, 95], axis=0)
        out.update(mean_curve=mean_curve, band=band)
    return out
