"""SPIRE benchmark harness (protocol in research/03_formulation.md section 6-7).
Usage: python bench.py OUT.csv d n noise seed_lo seed_hi method1,method2,...   (resume-safe)
Methods: SPIRE_c{c}_m{mu}[_T{TA}] | SPADE | SPADE-adaptive | GAM-ridge | var-sortnregress | R2-sortnregress
"""
import os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), HERE, os.path.join(ROOT, "research/pilots/G1_debiased_selection"),
                os.path.join(ROOT, "research/pilots/G2_shape_benchmark")]
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LassoLarsIC
from g1_pilot import fit as spade_fit, curve_metrics as spade_curve_metrics
from spade_variants import zscore, score
import shape_bench as G2
from spire import fit_spire, spire_curve
from basil import fit_basil


def gen(n, d, seed, noise="unif"):
    """iSCM: each node standardized right after generation (children see standardized parents)."""
    r = np.random.RandomState(seed); dens = 2.0 / (d - 1); perm = r.permutation(d)
    A = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a):
            if r.rand() < dens: A[perm[b], perm[a]] = 1
    fns = [lambda x: 0.8 * np.tanh(1.5 * x), lambda x: 0.7 * np.sin(1.5 * x),
           lambda x: 0.3 * (x ** 2 - 1.0), lambda x: 0.8 * x / (1.0 + np.abs(x))]
    f = {(c, e): fns[r.randint(len(fns))] for c in range(d) for e in range(d) if A[c, e]}
    X = np.zeros((n, d)); sd = np.ones(d); s0 = 0.7 / np.sqrt(3)
    for e in perm:
        if noise == "unif": v = r.uniform(-0.7, 0.7, n)
        elif noise == "gauss": v = r.randn(n) * s0
        elif noise == "laplace": v = r.laplace(0, s0 / np.sqrt(2), n)
        else: raise ValueError(noise)
        for c in range(d):
            if A[c, e]: v = v + f[(c, e)](X[:, c])
        sd[e] = v.std(); X[:, e] = (v - v.mean()) / sd[e]
    return X.astype(np.float32), A, f, sd


def r2_all(X):
    d = X.shape[1]; out = np.zeros(d); Z = (X - X.mean(0)) / X.std(0)
    for j in range(d):
        idx = [k for k in range(d) if k != j]; b = np.linalg.lstsq(Z[:, idx], Z[:, j], rcond=None)[0]
        out[j] = 1 - np.var(Z[:, j] - Z[:, idx] @ b) / np.var(Z[:, j])
    return out


def sortnregress(X, order):
    Xz = (X - X.mean(0)) / X.std(0); d = X.shape[1]; W = np.zeros((d, d))
    for k in range(1, d):
        j = order[k]; pa = order[:k]
        m = LassoLarsIC(criterion="bic").fit(Xz[:, pa], Xz[:, j]); W[pa, j] = np.abs(m.coef_)
    return W


def edge_curve_metrics(curve_fn, X, A, f, sd):
    r2, sl = [], []
    for c, e in zip(*np.nonzero(A)):
        gh = np.asarray(curve_fn(c, e, X[:, c]), float); gh = gh - gh.mean()
        gt = f[(c, e)](X[:, c].astype(float)) / sd[e]; gt = gt - gt.mean()
        r2.append(1 - ((gh - gt) ** 2).sum() / (gt ** 2).sum()); sl.append((gh * gt).sum() / (gt ** 2).sum())
    return float(np.mean(r2)), float(np.median(sl))


EXTRA = {}

def run_method(name, X, A, f, sd, seed):
    EXTRA.clear()
    Xz = zscore(X); Xt = torch.tensor(Xz); t0 = time.time(); cr = (np.nan, np.nan)
    if name.startswith("SPIRE"):
        kw = {}
        for tok in name.split("_")[1:]:
            if tok[0] == "c": kw["c_lam"] = float(tok[1:])
            elif tok[0] == "m": kw["mu"] = float(tok[1:])
            elif tok[0] == "T": kw["TA"] = int(tok[1:])
            elif tok[0] == "g": kw["gam"] = float(tok[1:])
            elif tok[0] == "E": kw["T"] = int(tok[1:])
        S, model = fit_spire(Xz, seed, return_model=True, **kw)
        cr = edge_curve_metrics(lambda c, e, xs: spire_curve(model, c, e, xs), Xz, A, f, sd)
    elif name.startswith("BASIL"):
        gm, pm_, kmax = 1.0, 1.0, 3
        for tok in name.split("_")[1:]:
            if tok[0] == "g": gm = float(tok[1:])
            elif tok[0] == "p": pm_ = float(tok[1:])
            elif tok[0] == "k": kmax = int(tok[1:])
        n_, d_ = Xz.shape
        o = fit_basil(Xz, seed, g=gm * n_, pi=pm_ * 2.0 / (d_ - 1), kmax=kmax)
        S = o["P"]; cr = edge_curve_metrics(lambda c, e, xs: o["mean_curve"](c, e), Xz, A, f, sd)
        cover = []
        for c, e in zip(*np.nonzero(A)):
            lo_, hi_ = o["band"](c, e); gt = f[(c, e)](Xz[:, c].astype(float)) / sd[e]; gt = gt - gt.mean()
            cover.append(np.mean((gt >= lo_ - 1e-9) & (gt <= hi_ + 1e-9)))
        EXTRA.update(cover=float(np.mean(cover)), rhat=o["rhat"], gap=o["chain_gap"])
    elif name == "SPADE":
        m = spade_fit(Xt, seed); S = m.importance(Xt).numpy().T; cr = spade_curve_metrics(m, Xt, A, f, sd)
    elif name == "SPADE-adaptive":
        m1 = spade_fit(Xt, seed)
        with torch.no_grad():
            n1 = torch.linalg.vector_norm(m1.coef * m1.selfmask, dim=2); w = (n1 + 1e-2) ** (-1.0); w = w / w.mean()
        m = spade_fit(Xt, seed, weights=w); S = m.importance(Xt).numpy().T; cr = spade_curve_metrics(m, Xt, A, f, sd)
    elif name == "GAM-ridge":
        S, cur = G2.fit_gam(Xz, seed); cr = edge_curve_metrics(cur, Xz, A, f, sd)
    elif name == "var-sortnregress":
        S = sortnregress(X.astype(np.float64), np.argsort(X.var(0)))
    elif name == "R2-sortnregress":
        S = sortnregress(X.astype(np.float64), np.argsort(r2_all(X.astype(np.float64))))
    else: raise ValueError(name)
    return np.asarray(S, float), cr, time.time() - t0


def ece(P, A, bins=10):
    d = A.shape[0]; msk = ~np.eye(d, dtype=bool); p = P[msk]; t = A[msk]; tot = 0.0
    for b in range(bins):
        sel = (p >= b / bins) & (p < (b + 1) / bins + (b == bins - 1))
        if sel.any(): tot += sel.mean() * abs(p[sel].mean() - t[sel].mean())
    return float(tot)


def metrics(S, A):
    d = A.shape[0]; au, ap, _ = score(S, A)
    iu = np.triu_indices(d, 1); sk = np.maximum(S, S.T)[iu]; tr = (A | A.T)[iu]
    ska = roc_auc_score(tr, sk) if 0 < tr.sum() < len(tr) else np.nan
    o = [1.0 if S[c, e] > S[e, c] else (0.5 if S[c, e] == S[e, c] else 0.0) for c, e in zip(*np.nonzero(A))]
    return au, ap, ska, float(np.mean(o))


if __name__ == "__main__":
    out, d, n, noise, lo, hi, methods = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7].split(",")
    done = set()
    if os.path.exists(out):
        old = pd.read_csv(out); done = set(zip(old.seed, old.method))
    for s in range(lo, hi):
        X, A, f, sd = gen(n, d, s, noise); rows = []
        for m in methods:
            if (s, m) in done: continue
            S, cr, t = run_method(m, X, A, f, sd, 0); au, ap, ska, oa = metrics(S, A)
            rows.append(dict(d=d, n=n, noise=noise, seed=s, method=m, auroc=au, auprc=ap, skel_auroc=ska, orient=oa,
                             curve_r2=cr[0], slope=cr[1], ece=ece(S, A) if m.startswith('BASIL') else np.nan, **EXTRA, nnz=int((S > 1e-6).sum()), n_edges=int(A.sum()), time=t))
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(s, [(r["method"], round(r["auroc"], 3), round(r["orient"], 2)) for r in rows], flush=True)
