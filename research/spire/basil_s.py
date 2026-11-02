"""BASIL-S: scalable BASIL (research/05_bayes_scale.md). Candidate screening + local order MCMC."""
import itertools, math, numpy as np
from basil import whitened_features, lse


def fit_basil_s(X, seed=0, m=20, kmax=3, g=None, pi=None, iters=None, burn_frac=0.5, chains=2, n_samples=300, curves=True, verbose=False, init='random', select=False):
    rng = np.random.RandomState(seed); X = np.asarray(X, np.float64); n, d = X.shape
    g = float(n if g is None else g); pi = (2.0 / (d - 1)) if pi is None else pi; rho = math.log(pi / (1 - pi)); m = min(m, d - 1)
    Z, Kc = whitened_features(X); Y = (X - X.mean(0)) / X.std(0)
    C = np.einsum("njk,ni->jik", Z, Y, optimize=True); R2 = (C ** 2).sum(-1) / (n * n); np.fill_diagonal(R2, -1)
    cand = np.argsort(-R2, axis=0)[:m].T.copy()                                       # cand[i] = top-m candidate parents of i
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
        r = np.random.RandomState(seed * 1000 + ch); order = greedy_order() if (init == 'greedy' and ch == 0) else list(r.permutation(d)); pos = np.empty(d, int); pos[order] = np.arange(d)
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
    out = dict(P=P, cand=cand, trace_gap=float(abs(traces[0][-200:].mean() - traces[-1][-200:].mean())), final_score=float(fin.max()), kept=len(keep))
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
    return out
