"""BASIL-S: scalable BASIL (research/05_bayes_scale.md). Candidate screening + local order MCMC."""
import itertools, math, numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../spire'))
from basil import whitened_features, lse


def fit_basil_s(X, seed=0, m=20, kmax=3, g=None, pi=None, iters=None, burn_frac=0.5, chains=2, n_samples=300, curves=True, verbose=False, init='random', select=False, cand_score=None, order0=None):
    rng = np.random.RandomState(seed); X = np.asarray(X, np.float64); n, d = X.shape
    g = float(n if g is None else g); pi = (2.0 / (d - 1)) if pi is None else pi; rho = math.log(pi / (1 - pi)); m = min(m, d - 1)
    Z, Kc, Tm, Tst = whitened_features(X, return_T=True); Y = (X - X.mean(0)) / X.std(0)
    C = np.einsum("njk,ni->jik", Z, Y, optimize=True); R2 = (C ** 2).sum(-1) / (n * n); np.fill_diagonal(R2, -1)
    rank_sc = R2 if cand_score is None else (np.asarray(cand_score, float) + 1e-6 * R2)
    np.fill_diagonal(rank_sc, -1e9)
    cand = np.argsort(-rank_sc, axis=0)[:m].T.copy()                                       # cand[i] = top-m candidate parents of i
    combos = [np.zeros((1, 0), int)] + [np.array(list(itertools.combinations(range(m), k)), dtype=int) for k in range(1, kmax + 1)]
    sizes = np.concatenate([np.full(len(c), k) for k, c in enumerate(combos)])
    masks = np.concatenate([(1 << c).sum(1) if len(c) and c.shape[1] else np.zeros(len(c), np.int64) for c in combos]).astype(np.int64)
    Mem = np.zeros((len(masks), m)); r0 = 0
    for k, c in enumerate(combos):
        for t in range(c.shape[1]): Mem[r0 + np.arange(len(c)), c[:, t]] = 1
        r0 += len(c)
    LS = np.zeros((d, len(masks))); Gs, Cs = [], []
    for i in range(d):
        F = Z[:, cand[i], :].reshape(n, m * Kc); G = F.T @ F; c = F.T @ Y[:, i]; Gs.append(G); Cs.append(c); r0 = 0
        for k, cb in enumerate(combos):
            S = len(cb)
            if k == 0: r2 = np.zeros(1)
            else:
                idx = (cb[:, :, None] * Kc + np.arange(Kc)).reshape(S, -1)
                Gsub = G[idx[:, :, None], idx[:, None, :]] + 1e-9 * np.eye(k * Kc)
                cc = c[idx]; r2 = np.clip(np.einsum("sp,sp->s", cc, np.linalg.solve(Gsub, cc[..., None])[..., 0]) / (Y[:, i] @ Y[:, i]), 0, 1 - 1e-9)
            p = k * Kc
            LS[i, r0:r0 + S] = (n - 1 - p) / 2 * math.log(1 + g) - (n - 1) / 2 * np.log(1 + g * (1 - r2)) + k * rho; r0 += S
    pw = (2 ** np.arange(m)).astype(np.int64)
    def node_lse(i, pos):
        pm = int(((pos[cand[i]] < pos[i]) * pw).sum()); return lse(LS[i][(masks & ~pm) == 0])
    def greedy_order():
        R = np.ones(d, bool); rev = []
        for _ in range(d):
            best, bi = -1e300, -1
            for i in np.nonzero(R)[0]:
                pm = int((R[cand[i]] * pw).sum()); v = lse(LS[i][(masks & ~pm) == 0])
                if v > best: best, bi = v, i
            rev.append(bi); R[bi] = False
        return rev[::-1]
    iters = iters or 600 * d; burn = int(iters * burn_frac); samples = []; traces = []; chain_samples = []
    for ch in range(chains):
        r = np.random.RandomState(seed * 1000 + ch); order = (list(order0) if order0 is not None else greedy_order() if (init == 'greedy' and ch == 0) else list(r.permutation(d))); pos = np.empty(d, int); pos[order] = np.arange(d)
        ns = np.array([node_lse(order[k], pos) for k in range(d)]); tot = ns.sum(); tr = []
        every = max(1, (iters - burn) // n_samples); samples = []
        for it in range(iters):
            a = r.randint(d); b = a + int(r.choice([-4, -3, -2, -1, 1, 2, 3, 4])) if r.rand() < 0.4 else a + (1 if a < d - 1 else -1)
            if b < 0 or b >= d: tr.append(tot); continue
            lo, hi = min(a, b), max(a, b); old = order.copy(); old_ns = ns.copy()
            if abs(a - b) == 1 or r.rand() < 0.5: order[a], order[b] = order[b], order[a]                     # swap
            else: node = order.pop(a); order.insert(b, node)                                                    # shift
            for k in range(lo, hi + 1): pos[order[k]] = k
            new_ns = ns.copy()
            for k in range(lo, hi + 1): new_ns[k] = node_lse(order[k], pos)
            if math.log(r.rand() + 1e-300) < new_ns.sum() - tot: ns = new_ns; tot = ns.sum()
            else:
                order = old
                for k in range(lo, hi + 1): pos[order[k]] = k
            tr.append(tot)
            if it >= burn and (it - burn) % every == 0: samples.append(pos.copy())
            if verbose and it % (iters // 10) == 0: print(ch, it, round(tot, 1), flush=True)
        traces.append(np.array(tr)); chain_samples.append(samples)
    fin = np.array([t[-200:].mean() for t in traces]); keep = [c for c in range(chains) if (not select) or fin[c] >= fin.max() - 10]
    samples = [p for c in keep for p in chain_samples[c]]
    P = np.zeros((d, d)); Wset = np.zeros((d, len(masks)))
    for pos in samples:
        for i in range(d):
            pm = int(((pos[cand[i]] < pos[i]) * pw).sum()); v = (masks & ~pm) == 0
            w = np.zeros(len(masks)); wv = np.exp(LS[i][v] - lse(LS[i][v])); w[v] = wv
            P[cand[i], i] += w @ Mem; Wset[i] += w
    P /= len(samples); Wset /= len(samples)
    out = dict(P=P, cand=cand, trace_gap=float(fin.max() - fin.min()), final_score=float(fin.max()), kept=len(keep))
    if curves:
        cache = {}
        def mean_curve(c, e):
            f = np.zeros(n); loc = np.where(cand[e] == c)[0]
            if len(loc) == 0: return f
            for s in np.argsort(-Wset[e])[:60]:
                if Wset[e][s] < 1e-4 or Mem[s, loc[0]] == 0: continue
                key = (e, s)
                if key not in cache:
                    mem = np.nonzero(Mem[s])[0]; idx = (mem[:, None] * Kc + np.arange(Kc)).reshape(-1)
                    u = np.linalg.solve(Gs[e][np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx)), Cs[e][idx]) * g / (1 + g); cache[key] = (list(mem), u)
                mem, u = cache[key]; k = mem.index(loc[0]); f += Wset[e][s] * (Z[:, c, :] @ u[k * Kc:(k + 1) * Kc])
            return f
        out["mean_curve"] = mean_curve
        def coef_tensor(stable=False):
            TT = Tst if stable else Tm; K = Tm.shape[1]; coef = np.zeros((d, d, K))
            for e in range(d):
                for sidx in np.argsort(-Wset[e])[:60]:
                    if Wset[e][sidx] < 1e-4: continue
                    mem = list(np.nonzero(Mem[sidx])[0])
                    if not mem: continue
                    idx = (np.array(mem)[:, None] * Kc + np.arange(Kc)).reshape(-1)
                    u = np.linalg.solve(Gs[e][np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx)), Cs[e][idx]) * g / (1 + g)
                    for k, lm in enumerate(mem): coef[e, cand[e][lm]] += Wset[e][sidx] * (TT[cand[e][lm]] @ u[k * Kc:(k + 1) * Kc])
            return coef
        out["coef"] = coef_tensor
        invc = {}
        def band(c, e, draws=100):
            loc = np.where(cand[e] == c)[0]; outb = np.zeros((draws, n))
            if len(loc) == 0: return np.zeros((2, n))
            ts = np.argsort(-Wset[e])[:60]; w = Wset[e][ts]; w = w / w.sum()
            for b in range(draws):
                sidx = ts[rng.choice(len(ts), p=w)]
                if Mem[sidx, loc[0]] == 0: continue
                mem = list(np.nonzero(Mem[sidx])[0]); idx = (np.array(mem)[:, None] * Kc + np.arange(Kc)).reshape(-1)
                if (e, sidx) not in invc:
                    Gsub = Gs[e][np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx)); u0 = np.linalg.solve(Gsub, Cs[e][idx]); r2 = float(Cs[e][idx] @ u0 / n)
                    s2 = n * (1 - g / (1 + g) * r2) / max(n - 3, 1); invc[(e, sidx)] = (g / (1 + g) * u0, s2 * g / (1 + g) * np.linalg.inv(Gsub))
                mu, cov = invc[(e, sidx)]; u = rng.multivariate_normal(mu, (cov + cov.T) / 2, check_valid="ignore")
                k = mem.index(loc[0]); outb[b] = Z[:, c, :] @ u[k * Kc:(k + 1) * Kc]
            return np.percentile(outb, [5, 95], axis=0)
        out["band"] = band

        def band_draws(c, e, draws=200, seed_=777):
            """posterior draws (mixture over models, 0 where edge absent) + inclusion flags + s2 of top model (diagnostics)."""
            r_ = np.random.RandomState(seed_ + 31 * e + c); loc = np.where(cand[e] == c)[0]; outb = np.zeros((draws, n)); incl = np.zeros(draws, bool)
            if len(loc) == 0: return outb, incl
            ts = np.argsort(-Wset[e])[:60]; w = Wset[e][ts]; w = w / w.sum()
            for b in range(draws):
                sidx = ts[r_.choice(len(ts), p=w)]
                if Mem[sidx, loc[0]] == 0: continue
                mem = list(np.nonzero(Mem[sidx])[0]); idx = (np.array(mem)[:, None] * Kc + np.arange(Kc)).reshape(-1)
                if (e, sidx) not in invc:
                    Gsub = Gs[e][np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx)); u0 = np.linalg.solve(Gsub, Cs[e][idx]); r2 = float(Cs[e][idx] @ u0 / n)
                    s2 = n * (1 - g / (1 + g) * r2) / max(n - 3, 1); invc[(e, sidx)] = (g / (1 + g) * u0, s2 * g / (1 + g) * np.linalg.inv(Gsub))
                mu, cov = invc[(e, sidx)]; u = r_.multivariate_normal(mu, (cov + cov.T) / 2, check_valid="ignore")
                k = mem.index(loc[0]); outb[b] = Z[:, c, :] @ u[k * Kc:(k + 1) * Kc]; incl[b] = True
            return outb, incl
        out["band_draws"] = band_draws
        def node_s2(e):
            """s2 of the highest-weight model of node e (sigma^2 estimate in standardised-Y units)."""
            sidx = int(np.argmax(Wset[e])); mem = np.nonzero(Mem[sidx])[0]
            if len(mem) == 0: return 1.0
            idx = (mem[:, None] * Kc + np.arange(Kc)).reshape(-1); u0 = np.linalg.solve(Gs[e][np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx)), Cs[e][idx]); r2 = float(Cs[e][idx] @ u0 / n)
            return n * (1 - g / (1 + g) * r2) / max(n - 3, 1)
        out["node_s2"] = node_s2; out["Z"] = Z
    return out
