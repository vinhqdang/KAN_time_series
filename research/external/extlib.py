"""External / published nonlinear DAG learners wrapped with a common interface (see PLAN.md).

Every fitted object offers
    S            [cause, effect] edge score (>= 0)
    G_own        [cause, effect] bool, the method's own selected graph (None if the method has no selection step)
    curve(c, e, xs, mode)   partial-dependence / additive-component curve of edge c->e at xs ; mode 'raw' (no selection) or 'own'
    node_pred(x, e)         prediction of node e from rows x using only its own-graph parents ('own' model), for do-effects
    order                   topological order of the (DAG-ified) own graph
All data are z-scored by the caller. Imports of research/ code are read-only; method code that is
copied from the repo (NOTEARS-MLP, DAGMA call) is marked.
"""
import os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); warnings.filterwarnings("ignore")
import numpy as np, torch, networkx as nx
torch.set_num_threads(1)

PD_ROWS, PD_GRID = 300, 41


def reset_dtype():
    torch.set_default_dtype(torch.float32)
    try: torch.set_default_tensor_type(torch.FloatTensor)
    except Exception: pass


def dagify(G, S):
    """remove the weakest edge (by S) of a cycle until the graph is acyclic"""
    G = G.copy().astype(bool); np.fill_diagonal(G, False)
    while True:
        g = nx.from_numpy_array(G.astype(int), create_using=nx.DiGraph)
        try: cyc = nx.find_cycle(g)
        except nx.NetworkXNoCycle: return G
        c, e = min(((u, v) for u, v, *_ in cyc), key=lambda t: S[t[0], t[1]]); G[c, e] = False


def topo(G):
    g = nx.from_numpy_array(G.astype(int), create_using=nx.DiGraph)
    return list(nx.topological_sort(g))


class PDModel:
    """Base for black-box node predictors. Subclasses provide predict_all(X, mode) -> [N, d] (numpy)."""
    def _init_pd(self, Xz):
        self.Xz = Xz; r = np.random.RandomState(0); self.rows = Xz[r.permutation(len(Xz))[:PD_ROWS]]
        self._cache = {}

    def _pd_all(self, c, mode):
        key = (c, mode)
        if key not in self._cache:
            grid = np.quantile(self.Xz[:, c], np.linspace(0, 1, PD_GRID)); R = len(self.rows)
            Xb = np.repeat(self.rows[None], PD_GRID, 0).copy(); Xb[:, :, c] = grid[:, None]
            out = self.predict_all(Xb.reshape(-1, Xb.shape[-1]), mode).reshape(PD_GRID, R, -1).mean(1)   # [grid, d]
            self._cache[key] = (grid, out)
        return self._cache[key]

    def curve(self, c, e, xs, mode="raw"):
        grid, out = self._pd_all(c, mode); return np.interp(np.asarray(xs, float), grid, out[:, e])

    def node_pred(self, x, e): return self.predict_all(np.asarray(x, np.float64), "own")[:, e]


def finish(M, S, G_own_raw=None):
    M.S = np.asarray(S, float); np.fill_diagonal(M.S, 0)
    if G_own_raw is None: M.G_own = None; M.order = None
    else: M.G_own = dagify(G_own_raw, M.S); M.order = topo(M.G_own)
    return M


# ------------------------------------------------------------------ NOTEARS-MLP (repo version, copied from scripts/instantaneous_dag_benchmark.py)
def _h_notears(W):
    return torch.trace(torch.linalg.matrix_exp(W * W)) - W.shape[0]


class NotearsMLP(PDModel):
    def __init__(self, Xz, seed, ep=600, hidden=10, thr=0.3):
        self._init_pd(Xz); f32 = torch.float32; torch.set_default_dtype(f32)
        torch.manual_seed(seed); np.random.seed(seed); Xt = torch.tensor(Xz, dtype=f32); d = Xz.shape[1]
        W1 = torch.nn.Parameter(0.1 * torch.randn(d, hidden, d, dtype=f32)); b1 = torch.nn.Parameter(torch.zeros(d, hidden, dtype=f32))
        W2 = torch.nn.Parameter(0.1 * torch.randn(d, hidden, dtype=f32)); b2 = torch.nn.Parameter(torch.zeros(d, dtype=f32))
        mask = (1 - torch.eye(d)).to(f32); opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-2); rho, al = 1.0, 0.0
        def Amat(): return torch.linalg.vector_norm(W1 * mask.unsqueeze(1), dim=1).t()
        for e in range(ep):
            opt.zero_grad(); Wm = W1 * mask.unsqueeze(1)
            hl = torch.tanh(torch.einsum("bc,ehc->beh", Xt, Wm) + b1); pred = torch.einsum("beh,eh->be", hl, W2) + b2
            h = _h_notears(Amat()); ((pred - Xt) ** 2).mean().add(0.01 * Wm.abs().mean() + al * h + 0.5 * rho * h * h).backward(); opt.step()
            if (e + 1) % 60 == 0:
                with torch.no_grad(): hv = _h_notears(Amat()).item()
                if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
        self.d = d; self.W1 = W1.detach().double().numpy(); self.b1 = b1.detach().double().numpy()
        self.W2 = W2.detach().double().numpy(); self.b2 = b2.detach().double().numpy(); self.mask_raw = (1 - np.eye(d))
        S = Amat().detach().numpy(); finish(self, S, S >= thr)

    def predict_all(self, X, mode):
        m = self.mask_raw if mode == "raw" else self.G_own.T.astype(float)        # [effect, cause]
        Wm = self.W1 * m[:, None, :]; h = np.tanh(np.einsum("bc,ehc->beh", X, Wm) + self.b1)
        return np.einsum("beh,eh->be", h, self.W2) + self.b2


# ------------------------------------------------------------------ DAGMA nonlinear (dagma 1.1.1, same call as the repo script)
class DagmaNL(PDModel):
    def __init__(self, Xz, seed, dce_thr=0.1, thr=0.3):
        from dagma.nonlinear import DagmaMLP, DagmaNonlinear
        import copy
        self._init_pd(Xz); d = Xz.shape[1]; torch.manual_seed(seed)
        eq = DagmaMLP(dims=[d, 10, 1], bias=True, dtype=torch.double)
        alg = DagmaNonlinear(eq, dtype=torch.double)
        W = np.abs(alg.fit(Xz.astype(np.float64), lambda1=0.02, lambda2=0.005, T=4, warm_iter=4000, max_iter=8000, w_threshold=0.0))
        self.net = alg.model; self.d = d; self.net.eval()
        self.W_raw = W; self.dce_thr = dce_thr
        finish(self, W, W >= thr)
        self._masked = {}

    def _net_mode(self, mode):
        if "own" not in self._masked:
            import copy
            n2 = copy.deepcopy(self.net); d = self.d
            with torch.no_grad():
                w = n2.fc1.weight.view(d, -1, d); w.mul_(torch.tensor(self.G_own.T.astype(np.float64))[:, None, :]); n2.fc1.weight.copy_(w.view(d * w.shape[1], d))
            self._masked["own"] = n2
        return self._masked["own"]

    def _raw_net(self):
        if "raw" not in self._masked:
            import copy; n2 = copy.deepcopy(self.net); d = self.d
            with torch.no_grad():
                w = n2.fc1.weight.view(d, -1, d); w.mul_(torch.tensor(1 - np.eye(d))[:, None, :]); n2.fc1.weight.copy_(w.view(d * w.shape[1], d))
            self._masked["raw"] = n2
        return self._masked["raw"]

    @torch.no_grad()
    def predict_all(self, X, mode):      # 'raw' = trained net minus the (never-used) diagonal; 'own' = own-graph pruned net
        net = self._raw_net() if mode == "raw" else self._net_mode("own")
        return net(torch.tensor(np.asarray(X, np.float64))).numpy()

    def dce_scores(self):
        """RMS partial derivative of node e w.r.t. input c over the data ('DCE-like'; NOT the training procedure of Dagma-DCE)."""
        X = torch.tensor(self.Xz.astype(np.float64)); d = self.d; D = np.zeros((d, d)); net = self._raw_net()
        for e in range(d):
            x = X.clone().requires_grad_(True); g, = torch.autograd.grad(net(x)[:, e].sum(), x)
            D[:, e] = np.sqrt((g ** 2).mean(0).numpy())
        np.fill_diagonal(D, 0); return D


class DceLike(DagmaNL):
    """same fitted DAGMA-MLP, re-scored with derivative-based edge strength; own graph = strength >= dce_thr (DAG-ified)."""
    def __init__(self, base, dce_thr=0.1):
        self.__dict__.update({k: v for k, v in base.__dict__.items() if k not in ("S", "G_own", "order", "_masked", "_cache")})
        self._masked = {}; self._cache = {}
        D = self.dce_scores(); finish(self, D, D >= dce_thr)


# ------------------------------------------------------------------ GraN-DAG (gcastle 1.0.4 reference implementation)
class GraNDag(PDModel):
    def __init__(self, Xz, seed, iterations=10000):
        os.environ.setdefault("CASTLE_BACKEND", "pytorch")
        import castle.algorithms as alg
        self._init_pd(Xz); d = Xz.shape[1]
        torch.set_default_dtype(torch.float64)
        m = alg.GraNDAG(input_dim=d, iterations=iterations, device_type="cpu", random_seed=int(seed) + 42)
        m.learn(Xz.astype(np.float64))
        self.m = m; self.d = d
        with torch.no_grad():
            S = m.model.get_w_adj().detach().cpu().numpy(); G = m.model.adjacency.detach().cpu().numpy() > 0
            self.weights, self.biases = [w.detach().clone() for w in m.model.get_parameters(mode="w")[0]], [b.detach().clone() for b in m.model.get_parameters(mode="b")[0]]
            self.adj = torch.tensor(m.model.adjacency.detach().cpu().numpy())
        finish(self, np.abs(S), G)

    @torch.no_grad()
    def predict_all(self, X, mode):      # GraN-DAG hard-clamps removed edges during training: 'raw' == 'own'
        mod = self.m.model; old = mod.adjacency
        try:
            mod.adjacency = self.adj; outs = mod.forward_given_params(torch.tensor(np.asarray(X, np.float64)), self.weights, self.biases)
        finally: mod.adjacency = old
        return np.stack([o[:, 0].numpy() for o in outs], 1)


# ------------------------------------------------------------------ CAM-like (greedy edge addition + spline GAM + F-test pruning)
class CAMLike:
    """NOT the original CAM (R, mgcv): additive models with fixed-df B-spline terms fitted by OLS, forward greedy
    edge addition with a BIC-penalised log-RSS gain, then per-node F-test pruning (p < 0.001, no multiplicity correction)."""
    def __init__(self, Xz, seed, nk=6, alpha=1e-3):
        from sklearn.preprocessing import SplineTransformer
        from scipy import stats
        n, d = Xz.shape; self.d = d; self.Xz = Xz
        self.trs = [SplineTransformer(n_knots=nk, degree=3, knots="quantile", extrapolation="constant").fit(Xz[:, [j]]) for j in range(d)]
        self.Bm = [t.transform(Xz[:, [j]]) for j, t in enumerate(self.trs)]; self.mu = [b.mean(0) for b in self.Bm]
        self.Bm = [b - m for b, m in zip(self.Bm, self.mu)]; nb = self.Bm[0].shape[1]; self.nb = nb
        pen = 0.5 * np.log(n) * nb; cache = {}; y = Xz - Xz.mean(0)

        def fit_rss(e, P):
            key = (e, tuple(sorted(P)))
            if key not in cache:
                if not P: cache[key] = (float((y[:, e] ** 2).sum()), None)
                else:
                    M = np.hstack([self.Bm[p] for p in sorted(P)]); b = np.linalg.lstsq(M, y[:, e], rcond=None)[0]
                    cache[key] = (float(((y[:, e] - M @ b) ** 2).sum()), b)
            return cache[key]
        pa = [set() for _ in range(d)]; reach = np.eye(d, dtype=bool)   # reach[i, j]: path i ~> j
        gain = np.full((d, d), -np.inf)   # [cause, effect]

        def upd(e):
            r0 = fit_rss(e, pa[e])[0]
            for c in range(d):
                gain[c, e] = -np.inf
                if c == e or c in pa[e] or reach[e, c]: continue
                gain[c, e] = 0.5 * n * np.log(r0 / fit_rss(e, pa[e] | {c})[0]) - pen
        for e in range(d): upd(e)
        while True:
            c, e = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[c, e] <= 0: break
            pa[e].add(c)
            anc = np.nonzero(reach[:, c])[0]; desc = np.nonzero(reach[e])[0]; reach[np.ix_(anc, desc)] = True
            for j in range(d): upd(j)   # cycle feasibility changes for several nodes
        self.pa_fwd = [set(p) for p in pa]
        # pruning and p-values
        S = np.zeros((d, d)); pa_own = [set() for _ in range(d)]; self.cache = cache
        for e in range(d):
            P = sorted(pa[e]); full, _ = fit_rss(e, set(P)); dfres = n - 1 - nb * len(P)
            for c in P:
                r0 = fit_rss(e, set(P) - {c})[0]; F = ((r0 - full) / nb) / (full / dfres); p = float(stats.f.sf(F, nb, dfres))
                S[c, e] = min(-np.log10(max(p, 1e-300)), 300.0)
                if p < 1e-3: pa_own[e].add(c)
        self.pa_own = pa_own; G = np.zeros((d, d), bool)
        for e in range(d):
            for c in pa_own[e]: G[c, e] = True
        self.S = S; self.G_own = G; self.order = topo(G); self.G_fwd = np.zeros((d, d), bool)
        for e in range(d):
            for c in pa[e]: self.G_fwd[c, e] = True
        self._coef = {}

    def _coefs(self, e, mode):
        P = self.pa_fwd[e] if mode == "raw" else self.pa_own[e]; key = (e, mode)
        if key not in self._coef:
            Ps = sorted(P); out = {}
            if Ps:
                _, b = self._fit(e, set(Ps))
                for k, c in enumerate(Ps): out[c] = b[k * self.nb:(k + 1) * self.nb]
            self._coef[key] = out
        return self._coef[key]

    def _fit(self, e, P): return self.cache[(e, tuple(sorted(P)))]

    def curve(self, c, e, xs, mode="raw"):
        co = self._coefs(e, mode)
        if c not in co: return np.zeros(len(xs))
        return (self.trs[c].transform(np.asarray(xs, float)[:, None]) - self.mu[c]) @ co[c]

    def node_pred(self, x, e):
        out = np.zeros(len(x))
        for c, b in self._coefs(e, "own").items(): out += (self.trs[c].transform(np.asarray(x[:, c], float)[:, None]) - self.mu[c]) @ b
        return out


# ------------------------------------------------------------------ NAM-parameterised SPADE variant (one small MLP per (effect, cause) edge)
class NAMSpade:
    """x_e = bias_e + sum_c f_ec(x_c), f_ec = a one-hidden-layer tanh MLP (H units) of the single input x_c (a Neural Additive Model
    layer, Agarwal et al. 2021, without ExU units or feature dropout). Same objective as our spline SPADE: MSE + lg * sum of
    group norms + augmented-Lagrangian NOTEARS acyclicity on the group-norm matrix; group norm := sqrt(n_basis) * RMS(f_ec)
    (matches the scale of the B-spline coefficient norm, n_basis = 11). 400 full-batch Adam epochs, ALM update every 50."""
    def __init__(self, Xz, seed, lg, mask=None, ep=400, H=16, lr=1e-2, nbasis=11):
        reset_dtype(); torch.manual_seed(seed); np.random.seed(seed)
        n, d = Xz.shape; self.d = d; self.H = H; Xt = torch.tensor(Xz, dtype=torch.float32); self.Xz = Xz
        self.sc = float(np.sqrt(nbasis))
        self.W1 = torch.nn.Parameter(torch.randn(d, d, H) * 1.0); self.b1 = torch.nn.Parameter((torch.rand(d, d, H) - 0.5) * 2.0)
        self.W2 = torch.nn.Parameter(0.1 * torch.randn(d, d, H) / np.sqrt(H) * 3); self.bias = torch.nn.Parameter(torch.zeros(d))
        sm = (1 - torch.eye(d)) if mask is None else torch.as_tensor(mask, dtype=torch.float32) * (1 - torch.eye(d)); self.sm = sm   # [effect, cause]
        opt = torch.optim.Adam([self.W1, self.b1, self.W2, self.bias], lr=lr); rho, al = 1.0, 0.0
        for e in range(ep):
            opt.zero_grad(); F = self._f(Xt) * sm                                 # [n, effect, cause]
            pred = F.sum(2) + self.bias; mse = ((pred - Xt) ** 2).mean()
            A = self.sc * torch.sqrt((F ** 2).mean(0) + 1e-10) * sm               # [effect, cause]
            h = _h_notears(A); (mse + lg * A.sum() + al * h + 0.5 * rho * h * h).backward()
            torch.nn.utils.clip_grad_norm_([self.W1, self.b1, self.W2, self.bias], 5.0); opt.step()
            if (e + 1) % 50 == 0:
                with torch.no_grad():
                    F2 = self._f(Xt) * sm; hv = _h_notears(self.sc * torch.sqrt((F2 ** 2).mean(0) + 1e-10) * sm).item()
                if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
        with torch.no_grad():
            F = self._f(Xt) * sm; self.S = F.std(0).numpy().T.astype(float)        # [cause, effect]
            self.Wn = (self.sc * torch.sqrt((F ** 2).mean(0) + 1e-10) * sm).numpy().T   # group-norm matrix [cause, effect]
        np.fill_diagonal(self.S, 0)
        self.W1n, self.b1n, self.W2n, self.bn = [t.detach().double().numpy() for t in (self.W1, self.b1, self.W2, self.bias)]
        self.smn = sm.numpy().astype(float)
        self.G_own = None
        from docalc import topo_order               # research/replication/docalc.py (read-only import)
        self.order = topo_order(self.Wn)
        self.pos = np.empty(d, int); self.pos[self.order] = np.arange(d)
        self.par = [[c for c in range(d) if self.smn[e, c] > 0 and self.Wn[c, e] > 0 and self.pos[c] < self.pos[e]] for e in range(d)]

    def _f(self, Xt):   # [n, effect, cause]
        z = torch.tanh(Xt[:, None, :, None] * self.W1[None] + self.b1[None])    # [n, e, c, H]
        return (z * self.W2[None]).sum(3)

    def curve(self, c, e, xs, mode="raw"):
        x = np.asarray(xs, np.float64)[:, None]; z = np.tanh(x * self.W1n[e, c][None] + self.b1n[e, c][None]); return (z * self.W2n[e, c][None]).sum(1) * self.smn[e, c]

    def node_pred(self, x, e):
        out = np.zeros(len(x))
        for c in self.par[e]: out += self.curve(c, e, x[:, c])
        return out


# ------------------------------------------------------------------ NOTEARS-MLP, gcastle port of the original (L-BFGS-B, l1 on fc1, l2 on all weights)
class NotearsMLPgc(PDModel):
    def __init__(self, Xz, seed, thr=0.3):
        os.environ.setdefault("CASTLE_BACKEND", "pytorch")
        from castle.algorithms import NotearsNonlinear
        self._init_pd(Xz); d = Xz.shape[1]; self.d = d; torch.manual_seed(seed); np.random.seed(seed)
        alg = NotearsNonlinear(lambda1=0.01, lambda2=0.01, w_threshold=thr, hidden_layers=(10, 1), device_type="cpu")
        torch.set_default_dtype(torch.float64)
        model = alg.get_model(d); W = np.abs(alg.notears_nonlinear(model, Xz.astype(np.float64)))
        self.net = model.eval(); self.h_final = float(alg.h); self._masked = {}
        finish(self, W, W >= thr)

    def _masked_net(self, mode):
        if mode not in self._masked:
            import copy; n2 = copy.deepcopy(self.net); d = self.d
            m = torch.tensor((1 - np.eye(d)) if mode == "raw" else self.G_own.T.astype(np.float64))[:, None, :]
            with torch.no_grad():
                for fc in (n2.fc1_pos, n2.fc1_neg):
                    w = fc.weight.view(d, -1, d); w.mul_(m); fc.weight.copy_(w.view(d * w.shape[1], d))
            self._masked[mode] = n2
        return self._masked[mode]

    @torch.no_grad()
    def predict_all(self, X, mode):
        return self._masked_net(mode)(torch.tensor(np.asarray(X, np.float64))).numpy()
