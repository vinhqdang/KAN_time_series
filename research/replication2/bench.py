"""SPIRE benchmark harness (protocol in research/03_formulation.md section 6-7).
Usage: python bench.py OUT.csv d n noise seed_lo seed_hi method1,method2,...   (resume-safe)
Methods: SPIRE_c{c}_m{mu}[_T{TA}] | SPADE | SPADE-adaptive | GAM-ridge | var-sortnregress | R2-sortnregress
"""
import os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../.."))  # research/replication2 -> repo root
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
from basil_s import fit_basil_s
from docalc import make_learned, do_error


MECH2 = [lambda x: 0.9 * x, lambda x: np.tanh(2 * x), lambda x: 0.6 * (x ** 2 - 1), lambda x: 1 / (1 + np.exp(-6 * (x - .3))) - .5, lambda x: 0.9 * np.sin(2 * x)]
MECH1 = [lambda x: 0.8 * np.tanh(1.5 * x), lambda x: 0.7 * np.sin(1.5 * x), lambda x: 0.3 * (x ** 2 - 1.0), lambda x: 0.8 * x / (1.0 + np.abs(x))]


def gen_full(n, d, seed, noise="unif", mech="m1"):
    """iSCM: each node standardized right after generation. Returns X, A, f, sd and sim(nsim, c, v) = fresh draws from the true SEM (do(x_c=v) if c given)."""
    r = np.random.RandomState(seed); dens = 2.0 / (d - 1); perm = r.permutation(d)
    A = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a):
            if r.rand() < dens: A[perm[b], perm[a]] = 1
    fns = MECH1 if mech == "m1" else MECH2
    f = {(c, e): fns[r.randint(len(fns))] for c in range(d) for e in range(d) if A[c, e]}
    X = np.zeros((n, d)); sd = np.ones(d); mu = np.zeros(d); s0 = 0.7 / np.sqrt(3)
    def noise_draw(rr, m, e, X_, pa_):
        if noise == "unif": v = rr.uniform(-0.7, 0.7, m)
        elif noise == "gauss": v = rr.randn(m) * s0
        elif noise == "laplace": v = rr.laplace(0, s0 / np.sqrt(2), m)
        elif noise == "hetero": v = rr.randn(m) * s0
        else: raise ValueError(noise)
        if noise == "hetero" and pa_: v = v * (0.4 + 1.2 * np.abs(X_[:, pa_[0]]))
        return v
    for e in perm:
        pa_ = [c for c in range(d) if A[c, e]]
        v = noise_draw(r, n, e, X, pa_)
        for c in pa_: v = v + f[(c, e)](X[:, c])
        sd[e] = v.std(); mu[e] = v.mean(); X[:, e] = (v - mu[e]) / sd[e]
    def sim(nsim, c=None, val=None, seed_=12345):
        rr = np.random.RandomState(seed_); Xs = np.zeros((nsim, d))
        for e in perm:
            if c is not None and e == c: Xs[:, e] = val; continue
            pa_ = [k for k in range(d) if A[k, e]]; v = noise_draw(rr, nsim, e, Xs, pa_)
            for k in pa_: v = v + f[(k, e)](Xs[:, k])
            Xs[:, e] = (v - mu[e]) / sd[e]
        return Xs
    return X.astype(np.float32), A, f, sd, sim


def gen(n, d, seed, noise="unif", mech="m1"):
    return gen_full(n, d, seed, noise, mech)[:4]


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
    r2, sl, nr = [], [], []
    for c, e in zip(*np.nonzero(A)):
        gh = np.asarray(curve_fn(c, e, X[:, c]), float); gh = gh - gh.mean()
        gt = f[(c, e)](X[:, c].astype(float)) / sd[e]; gt = gt - gt.mean()
        r2.append(1 - ((gh - gt) ** 2).sum() / (gt ** 2).sum()); sl.append((gh * gt).sum() / (gt ** 2).sum()); nr.append(np.sqrt(((gh - gt) ** 2).sum() / (gt ** 2).sum()))
    EXTRA['nrmse'] = float(np.mean(nr))
    return float(np.mean(r2)), float(np.median(sl))


EXTRA = {}
EXTRA_T = {}
SS_CACHE = {}
SIM = [None]


def add_do(coef, Xz, A, seed):
    if SIM[0] is None: return
    r = do_error(make_learned(coef, Xz), SIM[0], Xz, A, seed)
    EXTRA.update(do_direct=r['direct'][0], do_indirect=r['indirect'][0], n_do_d=r['direct'][1], n_do_i=r['indirect'][1], sk_do=r['direct'][2] + r['indirect'][2])

def run_method(name, X, A, f, sd, seed):
    EXTRA.clear(); EXTRA_T.clear()
    Xz = zscore(X); Xt = torch.tensor(Xz); t0 = time.time(); cr = (np.nan, np.nan); t_extra = 0.0
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
    elif name.startswith("BASIL") and not name.startswith("BASILS") and not name.startswith("BASILW"):
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
    elif name.startswith("BASILS"):
        kw = {}
        for tok in name.split("_")[1:]:
            if tok[0] == "m": kw["m"] = int(tok[1:])
            elif tok[0] == "i": kw["iters"] = int(float(tok[1:]) * Xz.shape[1])
            elif tok[0] == "c": kw["chains"] = int(tok[1:])
            elif tok[0] == "G": kw["init"] = "greedy"
            elif tok[0] == "S": kw["select"] = True
        o = fit_basil_s(Xz, seed, **kw); S = o["P"]
        cr = edge_curve_metrics(lambda c, e, xs: o["mean_curve"](c, e), Xz, A, f, sd)
        rec = np.mean([c in o["cand"][e] for c, e in zip(*np.nonzero(A))])
        EXTRA.update(recall=float(rec), gap=o["trace_gap"], score_final=o["final_score"])
    elif name in ("SSC", "SSA", "BASILW", "BASILWs") or name.startswith("SSR"):
        import scale_instantaneous as SI
        key = (seed, X.shape, float(X.sum()))
        if key not in SS_CACHE:
            Zs = SI.zscore(X); Xt_ = torch.tensor(Zs); tt = time.time(); mk = SI.screen_mask(Zs, 15, seed)
            mm = SI.fit(Xt_, seed, SI.lg_rule(X.shape[1], 0.2), ep=400, mask=mk); S0 = SI.chunked_importance(mm, Xt_)
            SS_CACHE.clear(); SS_CACHE[key] = (mm, S0, Xt_, time.time() - tt)
        mm, S0, Xt_, t_ss = SS_CACHE[key]; d_ = X.shape[1]
        def sp_curve(model, Xt__):
            Bm = model._bases(Xt__).detach(); cf = (model.coef * model.selfmask).detach()
            return lambda c, e, xs: (Bm[:, c, :] @ cf[e, c]).numpy()
        if name == "SSC":
            S = S0; cr = edge_curve_metrics(sp_curve(mm, Xt_), Xz, A, f, sd); add_do((mm.coef * mm.selfmask).detach().numpy(), Xz, A, seed)
        elif name == "SSA":
            import spade_variants as SV
            t_a0 = time.time()
            mk = mm.selfmask.squeeze(-1).detach().clone()
            with torch.no_grad():
                n1 = torch.linalg.vector_norm(mm.coef * mm.selfmask, dim=2); scr = (mk > 0)
                w = (n1 + 1e-2) ** (-1.0); w = w / w[scr].mean()
            ma = SV.fit(Xt_, seed, SI.lg_rule(d_, 0.2), ep=400, weights=w, mask=mk); S = SI.chunked_importance(ma, Xt_); EXTRA_T['t_refit'] = time.time() - t_a0
            cr = edge_curve_metrics(sp_curve(ma, Xt_), Xz, A, f, sd); add_do((ma.coef * ma.selfmask).detach().numpy(), Xz, A, seed); t_extra = t_ss
        elif name.startswith("SSR"):
            kk = float(name.split("_k")[1]); kn = int(kk * d_); thr = np.sort(S0.flatten())[::-1][kn - 1]
            mk2 = torch.tensor((S0.T >= thr).astype(np.float32)); np.fill_diagonal(mk2.numpy(), 0)
            t_r0 = time.time(); m2 = SI.fit(Xt_, seed, 0.0, ep=400, mask=mk2); S = SI.chunked_importance(m2, Xt_); EXTRA_T['t_refit'] = time.time() - t_r0
            cr = edge_curve_metrics(sp_curve(m2, Xt_), Xz, A, f, sd); add_do((m2.coef * m2.selfmask).detach().numpy(), Xz, A, seed); t_extra = t_ss
        else:  # BASILW
            inw = S0.sum(0).copy(); order0 = []
            for _ in range(d_):
                i = int(np.argmin(inw)); order0.append(i); inw[i] = np.inf; inw -= np.where(np.isinf(inw), 0, S0[i, :])
            t_m0 = time.time(); o = fit_basil_s(Xz, seed, m=20, iters=200 * d_, chains=2, burn_frac=0.3, select=True, cand_score=S0, order0=order0); EXTRA_T['t_mcmc'] = time.time() - t_m0
            S = o["P"]; cr = edge_curve_metrics(lambda c, e, xs: o["mean_curve"](c, e), Xz, A, f, sd)
            cover = []
            for c, e in zip(*np.nonzero(A)):
                lo_, hi_ = o["band"](c, e); gt = f[(c, e)](Xz[:, c].astype(float)) / sd[e]; gt = gt - gt.mean()
                cover.append(np.mean((gt >= lo_ - 1e-9) & (gt <= hi_ + 1e-9)))
            rec = np.mean([c in o["cand"][e] for c, e in zip(*np.nonzero(A))])
            t_e0 = time.time(); cf_ = o["coef"](stable=(name == "BASILWs")); EXTRA_T["t_extract"] = time.time() - t_e0
            add_do(cf_, Xz, A, seed)
            EXTRA.update(cover=float(np.mean(cover)), recall=float(rec), gap=o["trace_gap"]); t_extra = t_ss
        EXTRA['t_ss'] = t_ss
    elif name == "SPADE-scaled-curves":
        import scale_instantaneous as SI
        Zs = SI.zscore(X); Xt_ = torch.tensor(Zs); mk = SI.screen_mask(Zs, 15, seed)
        mm = SI.fit(Xt_, seed, SI.lg_rule(X.shape[1], 0.2), ep=400, mask=mk); S = SI.chunked_importance(mm, Xt_)
        cr = spade_curve_metrics(mm, Xt_, A, f, sd)
    elif name == "SPADE-scaled":
        import scale_instantaneous as SI
        S = SI.spade(X, seed, SI.lg_rule(X.shape[1], 0.2), 15)
    elif name == "SPADE":
        m = spade_fit(Xt, seed); S = m.importance(Xt).numpy().T; cr = spade_curve_metrics(m, Xt, A, f, sd)
    elif name == "SPADE-adaptive":
        m1 = spade_fit(Xt, seed)
        with torch.no_grad():
            n1 = torch.linalg.vector_norm(m1.coef * m1.selfmask, dim=2); w = (n1 + 1e-2) ** (-1.0); w = w / w.mean()
        m = spade_fit(Xt, seed, weights=w); S = m.importance(Xt).numpy().T; cr = spade_curve_metrics(m, Xt, A, f, sd)
    elif name == "GAM-ridge":
        S, cur = G2.fit_gam(Xz, seed); cr = edge_curve_metrics(cur, Xz, A, f, sd)
    elif name in ("SCORE", "NoGAM"):
        import instantaneous_dag_benchmark as IB
        S = IB.METHODS[name](X, 0)
    elif name == "var-sortnregress":
        S = sortnregress(X.astype(np.float64), np.argsort(X.var(0)))
    elif name == "R2-sortnregress":
        S = sortnregress(X.astype(np.float64), np.argsort(r2_all(X.astype(np.float64))))
    else: raise ValueError(name)
    cost = np.nan
    ts = EXTRA.get('t_ss', np.nan)
    if name.startswith('SSR') or name == 'SSA': cost = ts + EXTRA_T.get('t_refit', np.nan)
    elif name.startswith('BASILW'): cost = ts + EXTRA_T.get('t_mcmc', np.nan) + EXTRA_T.get('t_extract', np.nan)
    elif name == 'SSC': cost = ts
    else: cost = np.nan
    EXTRA.update({k: v for k, v in EXTRA_T.items()}); EXTRA['t_cost'] = cost
    return np.asarray(S, float), cr, cost


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
    mech = sys.argv[8] if len(sys.argv) > 8 else "m1"
    done = set()
    if os.path.exists(out):
        old = pd.read_csv(out); done = set(zip(old.seed, old.method))
    for s in range(lo, hi):
        X, A, f, sd, SIM[0] = gen_full(n, d, s, noise, mech); rows = []
        for m in methods:
            if (s, m) in done: continue
            S, cr, t = run_method(m, X, A, f, sd, 0); au, ap, ska, oa = metrics(S, A)
            rows.append(dict(d=d, n=n, noise=noise, mech=mech, seed=s, method=m, auroc=au, auprc=ap, skel_auroc=ska, orient=oa,
                             curve_r2=cr[0], slope=cr[1], ece=ece(S, A) if m.startswith('BASIL') else np.nan, **EXTRA, nnz=int((S > 1e-6).sum()), n_edges=int(A.sum())))
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(s, [(r["method"], round(r["auroc"], 3), round(r["orient"], 2)) for r in rows], flush=True)
