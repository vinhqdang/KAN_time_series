"""
G2 pilot: shape-faithful causal discovery benchmark (method-agnostic metrics + generator).

PILOT PROTOCOL (written before running; amendments are listed at the bottom of the file).
Question (RQ-B): do learners rank differently under edge-shape metrics than under edge AUROC?
Generator: d=8 nodes, ER DAG (expected ~8 edges), n=1000. Each edge gets one mechanism uniformly from
  {linear 0.9x, saturating tanh(2x), non-monotone 0.6(x^2-1), threshold sigmoid(6(x-.3))-.5, periodic .9 sin(2x)},
  applied to the STANDARDISED parent; node = sum of edge terms + N(0,.4^2), then standardised.
  Concurvity knob rho in {0, 0.6, 0.9}: root nodes are equicorrelated Gaussians with correlation rho
  (non-root dependence between co-parents arises from shared ancestry and is not controlled).
Learners (all see z-scored data, same Adam/ALM/400-epoch budget where they share a loop, no per-method tuning):
  SPADE      : CausalKANInstant (B-splines, G=8, k=3, group-lasso 0.01, ALM acyclicity), published defaults.
  MLP-edge   : same bottleneck/penalty/budget, 1-3-1 tanh edges (as in scripts/interpretability_fidelity.py).
  GAM-ridge  : classical additive baseline, per-node ridge on spline bases of all other nodes (sklearn
               SplineTransformer 8 knots + RidgeCV), NO DAG constraint and NO sparsity, edge score = std of
               edge contribution. pygam is not installed; this is a plain penalised-GAM stand-in, not CAM.
  NT-MLP     : NOTEARS-MLP style joint model (d->16->1 per node, first-layer column-group-lasso, ALM acyclicity
               on first-layer column norms; edge score = those norms); curves = averaged partial dependence
               (ICE average over 200 background rows), i.e. standard PDP (no derivative-based score).
  Linear     : per-node OLS; curve = line. A floor showing what 'edge found, shape wrong' looks like.
Seeds: VALIDATION 1000-1004 (rho cycled) are used ONLY to pick the oracle-free relative threshold tau
  from {.05,.1,.2,.3,.5} (per method, maximising F1). TEST seeds 2000-2009 x rho {0,.6,.9} are never used
  for any choice. Single-threaded. Bootstrap stability: B=3 half-sample refits per dataset.
Edge selection rules reported: (a) oracle-free tau on score/max(score); (b) oracle top-|E_true| for reference.
Metrics (per TRUE edge, curve compared with the true mechanism on 50 quantiles (5%-95%) of the cause):
  R2_aff  : R2 of best POSITIVE-slope affine map a*phi+b -> f (a<=0 gives 0); clipped to [0,1].
  cls_ok  : class {increasing, decreasing, non-monotone} agreement.
  dsign   : fraction of adjacent-grid steps with |df|>2% of f-range where sign(dphi)==sign(df).
  SF-recall: share of true edges that are found AND have R2_aff>=0.8 ;  SF-F1 = 2*SFTP/(#selected+#true).
  cond-R2 : mean R2_aff over true edges that were found;  uncond-R2: missed edges count as 0.
  stab    : mean pairwise R2_aff between bootstrap curves of the same true edge (found edges only).
  AUROC/AUPRC of edge scores (standard).
"""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch, torch.nn as nn, pandas as pd
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from src.cdkan.causal_kan import CausalKANInstant
from src.cdkan.losses import _h_notears
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV

MECH = {
 "linear": lambda x: 0.9 * x, "saturating": lambda x: np.tanh(2 * x), "nonmonotone": lambda x: 0.6 * (x**2 - 1),
 "threshold": lambda x: 1 / (1 + np.exp(-6 * (x - .3))) - .5, "periodic": lambda x: 0.9 * np.sin(2 * x)}
MNAMES = list(MECH)

def zs(X): return ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)

def generate(seed, d=8, n=1000, rho=0.0):
    r = np.random.RandomState(seed); perm = r.permutation(d); p = 2.0 / (d - 1)
    A = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a):
            if r.rand() < p: A[perm[b], perm[a]] = 1
    if A.sum() == 0: A[perm[0], perm[1]] = 1
    mech = {(c, e): MNAMES[r.randint(5)] for c, e in zip(*np.nonzero(A))}
    roots = [j for j in range(d) if A[:, j].sum() == 0]
    C = np.full((len(roots), len(roots)), rho) + (1 - rho) * np.eye(len(roots))
    Z = r.randn(n, len(roots)) @ np.linalg.cholesky(C).T
    X = np.zeros((n, d))
    for k, j in enumerate(roots): X[:, j] = Z[:, k]
    for j in perm:
        if j in roots: continue
        s = sum(MECH[mech[(c, j)]](X[:, c]) for c in np.nonzero(A[:, j])[0])
        X[:, j] = s + 0.4 * r.randn(n); X[:, j] = (X[:, j] - X[:, j].mean()) / X[:, j].std()
    for j in roots: X[:, j] = (X[:, j] - X[:, j].mean()) / X[:, j].std()
    return X.astype(np.float32), A, mech

# ------------------------------------------------------------------ learners
def _alm(m, Xt, seed, lg=0.01, ep=400):
    torch.manual_seed(seed); opt = torch.optim.Adam(m.parameters(), lr=5e-3); rho, al = 1.0, 0.0
    for e in range(ep):
        opt.zero_grad(); h = m.h()
        (((m(Xt) - Xt) ** 2).mean() + lg * m.group_lasso() + al * h + 0.5 * rho * h * h).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if (e + 1) % 50 == 0:
            hv = m.h().item()
            if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
    return m

class MLPEdge(nn.Module):
    def __init__(s, d, hd=3):
        super().__init__(); s.d = d
        s.w1 = nn.Parameter(0.5 * torch.randn(d, d, hd)); s.b1 = nn.Parameter(torch.zeros(d, d, hd))
        s.w2 = nn.Parameter(0.3 * torch.randn(d, d, hd)); s.bias = nn.Parameter(torch.zeros(d))
        s.register_buffer("mask", 1 - torch.eye(d))
    def edge(s, x): return (torch.tanh(x[:, None, :, None] * s.w1[None] + s.b1[None]) * s.w2[None]).sum(-1) * s.mask[None]
    def forward(s, x): return s.edge(x).sum(2) + s.bias
    def W(s): return torch.linalg.vector_norm(s.w2 * s.mask[..., None], dim=2)
    def group_lasso(s): return s.W().sum()
    def h(s): return _h_notears(s.W())

class NTMLP(nn.Module):
    def __init__(s, d, hd=16):
        super().__init__(); s.d = d
        s.W1 = nn.Parameter(0.3 * torch.randn(d, hd, d)); s.b1 = nn.Parameter(torch.zeros(d, hd))
        s.W2 = nn.Parameter(0.3 * torch.randn(d, hd)); s.bias = nn.Parameter(torch.zeros(d))
        s.register_buffer("mask", 1 - torch.eye(d))
    def forward(s, x):
        z = torch.tanh(torch.einsum("ihj,bj->bih", s.W1 * s.mask[:, None, :], x) + s.b1)
        return (z * s.W2).sum(-1) + s.bias
    def W(s): return torch.linalg.vector_norm(s.W1 * s.mask[:, None, :], dim=1)    # [effect, cause]
    def group_lasso(s): return s.W().sum()
    def h(s): return _h_notears(s.W())

def fit_spade(X, seed):
    torch.manual_seed(seed); Xt = torch.tensor(X); m = CausalKANInstant(X.shape[1], grid_size=8); m = _alm(m, Xt, seed)
    S = m.importance(Xt).numpy().T
    def curve(c, e, xs):
        xt = torch.zeros(len(xs), X.shape[1]); xt[:, c] = torch.tensor(xs, dtype=torch.float32)
        with torch.no_grad(): return (m._bases(xt)[:, c, :] @ (m.coef * m.selfmask)[e, c]).numpy()
    return S, curve

def fit_mlpedge(X, seed):
    torch.manual_seed(seed); Xt = torch.tensor(X); m = _alm(MLPEdge(X.shape[1]), Xt, seed)
    S = m.edge(Xt).std(0).detach().numpy().T
    def curve(c, e, xs):
        xt = torch.zeros(len(xs), X.shape[1]); xt[:, c] = torch.tensor(xs, dtype=torch.float32)
        with torch.no_grad(): return m.edge(xt)[:, e, c].numpy()
    return S, curve

def fit_ntmlp(X, seed):
    torch.manual_seed(seed); Xt = torch.tensor(X); m = _alm(NTMLP(X.shape[1]), Xt, seed)
    S = m.W().detach().numpy().T; bg = Xt[:200].clone()
    def curve(c, e, xs):                                     # averaged partial dependence on cause c
        out = []
        with torch.no_grad():
            for v in xs:
                b = bg.clone(); b[:, c] = float(v); out.append(m(b)[:, e].mean().item())
        return np.array(out)
    return S, curve

def fit_gam(X, seed):
    n, d = X.shape
    trs = [SplineTransformer(n_knots=8, degree=3, knots="quantile", extrapolation="constant").fit(X[:, [j]]) for j in range(d)]
    B = [trs[j].transform(X[:, [j]]) for j in range(d)]; B = [b - b.mean(0) for b in B]; nb = B[0].shape[1]
    coefs = np.zeros((d, d, nb)); S = np.zeros((d, d))
    for e in range(d):
        idx = [j for j in range(d) if j != e]; M = np.hstack([B[j] for j in idx])
        r = RidgeCV(alphas=np.logspace(-1, 3, 9), fit_intercept=True).fit(M, X[:, e])
        for k, j in enumerate(idx):
            coefs[e, j] = r.coef_[k * nb:(k + 1) * nb]; S[j, e] = (B[j] @ coefs[e, j]).std()
    def curve(c, e, xs): return trs[c].transform(np.asarray(xs, float)[:, None]) @ coefs[e, c]
    return S, curve

def fit_linear(X, seed):
    n, d = X.shape; Bm = np.zeros((d, d))
    for e in range(d):
        idx = [j for j in range(d) if j != e]; Bm[e, idx] = np.linalg.lstsq(X[:, idx], X[:, e], rcond=None)[0]
    return np.abs(Bm).T, (lambda c, e, xs: Bm[e, c] * np.asarray(xs))

LEARNERS = {"SPADE": fit_spade, "MLP-edge": fit_mlpedge, "NT-MLP": fit_ntmlp, "GAM-ridge": fit_gam, "Linear": fit_linear}

# ------------------------------------------------------------------ metrics
def r2_aff(phi, f):
    if np.ptp(phi) < 1e-9: return 0.0
    M = np.c_[phi, np.ones_like(phi)]; a, b = np.linalg.lstsq(M, f, rcond=None)[0]
    if a <= 0: return 0.0
    return float(np.clip(1 - np.var(f - (a * phi + b)) / (np.var(f) + 1e-12), 0, 1))

def cls(v):
    rg = np.ptp(v) + 1e-12; dv = np.diff(v); up, dn = dv[dv > 0].sum() / rg, -dv[dv < 0].sum() / rg
    if min(up, dn) > 0.15: return "nonmono"
    return "inc" if up >= dn else "dec"

def dsign(phi, f):
    df, dp = np.diff(f), np.diff(phi); k = np.abs(df) > 0.02 * np.ptp(f)
    return float(np.mean(np.sign(dp[k]) == np.sign(df[k]))) if k.any() else np.nan

def edge_scores(S, A):
    d = A.shape[0]; m = ~np.eye(d, dtype=bool); p = np.nan_to_num(S[m].astype(float)); t = A[m]
    return roc_auc_score(t, p), average_precision_score(t, p)

def selected(S, rule, tau=None, k=None):
    S = S.copy(); np.fill_diagonal(S, 0)
    if rule == "tau": return S / (S.max() + 1e-12) >= tau
    thr = np.sort(S.flatten())[::-1][k - 1]; return S >= thr

def f1_sel(sel, A):
    t = A.astype(bool); tp = (sel & t).sum(); return 2 * tp / (sel.sum() + t.sum() + 1e-12)

def evaluate(X, A, mech, seed, learners, B=3):
    d = X.shape[1]; rng = np.random.RandomState(seed); rows = []; scores = {}
    true_edges = list(zip(*np.nonzero(A)))
    for name in learners:
        t0 = time.time(); S, cur = LEARNERS[name](X, seed); scores[name] = S
        au, ap = edge_scores(S, A); boots = []
        for b in range(B):
            idx = rng.choice(len(X), len(X) // 2, replace=False); boots.append(LEARNERS[name](zs(X[idx]), seed * 10 + b)[1])
        for c, e in true_edges:
            xs = np.quantile(X[:, c], np.linspace(0.05, 0.95, 50)); f = MECH[mech[(c, e)]](xs)
            phi = cur(c, e, xs); bc = [bf(c, e, xs) for bf in boots]
            stab = np.mean([0.5 * (r2_aff(bc[i], bc[j]) + r2_aff(bc[j], bc[i])) for i in range(B) for j in range(i)])
            rows.append(dict(seed=seed, method=name, c=c, e=e, mech=mech[(c, e)], r2=r2_aff(phi, f), cls_ok=cls(phi) == cls(f),
                             dsign=dsign(phi, f), stab=stab, auroc=au, auprc=ap, time=time.time() - t0))
    return pd.DataFrame(rows), scores

def summarise(df, A_by, S_by, tau_by):
    out = []
    for (rho, name), g in df.groupby(["rho", "method"]):
        res = dict(rho=rho, method=name); stabs = []
        for rule in ("tau", "oracleK"):
            tpr, sfr, f1s, sff1, cond, unc = [], [], [], [], [], []
            for seed, gs in g.groupby("seed"):
                A = A_by[seed]; S = S_by[(seed, rho, name)]
                sel = selected(S, "tau", tau_by[name]) if rule == "tau" else selected(S, "k", k=int(A.sum()))
                fnd = np.array([sel[r.c, r.e] for r in gs.itertuples()]); r2 = gs.r2.values; good = fnd & (r2 >= 0.8)
                tpr.append(fnd.mean()); sfr.append(good.mean()); f1s.append(f1_sel(sel, A))
                sff1.append(2 * good.sum() / (sel.sum() + A.sum() + 1e-12))
                cond.append(r2[fnd].mean() if fnd.any() else np.nan); unc.append((r2 * fnd).mean())
                if rule == "tau": stabs.append(gs.stab.values[fnd].mean() if fnd.any() else np.nan)
            sx = "" if rule == "tau" else "_k"
            res.update({f"recall{sx}": np.mean(tpr), f"F1{sx}": np.mean(f1s), f"SFrecall{sx}": np.mean(sfr), f"SF-F1{sx}": np.mean(sff1),
                        f"condR2{sx}": np.nanmean(cond), f"uncondR2{sx}": np.mean(unc)})
        res["stab"] = np.nanmean(stabs)
        res["AUROC"] = g.groupby("seed").auroc.first().mean(); res["AUPRC"] = g.groupby("seed").auprc.first().mean()
        res["R2_allTrue"] = g.r2.mean(); res["cls_ok"] = g.cls_ok.mean(); res["dsign"] = g.dsign.mean()
        out.append(res)
    return pd.DataFrame(out)

if __name__ == "__main__":
    import pickle
    mode = sys.argv[1]; names = list(LEARNERS)
    if mode == "time":
        X, A, mech = generate(1, rho=0.6)
        for n in names:
            t = time.time(); LEARNERS[n](X, 1); print(n, round(time.time() - t, 1), flush=True)
    elif mode == "val":
        res = {n: {tau: [] for tau in (.05, .1, .2, .3, .5)} for n in names}
        for i, seed in enumerate(range(1000, 1005)):
            X, A, mech = generate(seed, rho=(0, .6, .9)[i % 3])
            for n in names:
                S, _ = LEARNERS[n](X, seed)
                for tau in res[n]: res[n][tau].append(f1_sel(selected(S, "tau", tau), A))
        tau_by = {n: max(res[n], key=lambda t: np.mean(res[n][t])) for n in names}
        print({n: {t: round(np.mean(v), 3) for t, v in res[n].items()} for n in names}); print(tau_by)
        json.dump(tau_by, open(os.path.join(HERE, "tau_by_method.json"), "w"))
    elif mode == "test":
        s0, s1 = int(sys.argv[2]), int(sys.argv[3]); dfs = []; A_by = {}; S_by = {}
        for seed in range(s0, s1):
            for rho in (0.0, 0.6, 0.9):
                X, A, mech = generate(seed, rho=rho); A_by[seed] = A
                df, sc = evaluate(X, A, mech, seed, names); df["rho"] = rho; dfs.append(df)
                for n in names: S_by[(seed, rho, n)] = sc[n]
                print("done", seed, rho, flush=True)
        pickle.dump((pd.concat(dfs), A_by, S_by), open(os.path.join(HERE, f"raw_{s0}_{s1}.pkl"), "wb"))
