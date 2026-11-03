"""External-vs-ours edge-function benchmark (protocol: research/external/PLAN.md).
Usage: python ext_bench.py CELL d n noise seed_lo seed_hi method1,method2,...      (resume-safe, one CSV per cell in results/)
Our arms (SSC, SSR_k1.5, BASILWs, GAM-ridge) are run through research/replication/bench.py UNCHANGED (read-only import);
SSA is the adaptive arm of research/adaptive/bench.py (branch copied, same code). External arms use extlib.py.
"""
import os, sys, time, warnings, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("NJOBS", "1"); warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, "research/replication"))
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
import bench as B          # research/replication/bench.py (imports its own paths: scripts/, pilots/, ...)
import extlib as EL
from docalc import do_error
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# ------------------------------------------------------------------ per-edge statistics
EDGE = {}


def edge_stats(curve_fn, Xz, A, f, sd):
    r2, sl, nr = [], [], []
    for c, e in zip(*np.nonzero(A)):
        gh = np.asarray(curve_fn(c, e, Xz[:, c]), float); gh = gh - gh.mean()
        gt = f[(c, e)](Xz[:, c].astype(float)) / sd[e]; gt = gt - gt.mean(); den = (gt ** 2).sum()
        r2.append(1 - ((gh - gt) ** 2).sum() / den); sl.append((gh * gt).sum() / den); nr.append(np.sqrt(((gh - gt) ** 2).sum() / den))
    return np.array(r2), np.array(sl), np.array(nr)


def _wrapped(curve_fn, Xz, A, f, sd):          # replaces bench.edge_curve_metrics: same numbers, also keeps the per-edge arrays
    r2, sl, nr = edge_stats(curve_fn, Xz, A, f, sd); EDGE.update(r2=r2, sl=sl, nr=nr); B.EXTRA['nrmse'] = float(nr.mean())
    return float(r2.mean()), float(np.median(sl))


B.edge_curve_metrics = _wrapped


def summ(r2, sl, nr): return dict(nrmse=float(np.mean(nr)), r2=float(np.mean(r2)), slope=float(np.median(sl)))


def select_mask(A, sel):
    """selected[k] for the k-th true edge (np.nonzero order)"""
    return np.array([bool(sel[c, e]) for c, e in zip(*np.nonzero(A))])


def topk_graph(S, k):
    d = S.shape[0]; Sm = S.copy().astype(float); np.fill_diagonal(Sm, -np.inf)
    idx = np.argsort(-Sm.ravel(), kind="stable")[:k]; G = np.zeros((d, d), bool); G.ravel()[idx] = True
    return G & (Sm > 0)          # a zero-score entry is never "selected"


def rules(r2, sl, nr, A, S, own_graph):
    """raw (no selection) / own (method's own graph; missed -> zero curve) / topk (oracle count k=|E|; missed -> zero curve)"""
    out = {}
    for tag, r in (("raw", (r2, sl, nr)),):
        out.update({f"{k}_{tag}": v for k, v in summ(*r).items()})
    for tag, G in (("own", own_graph), ("topk", topk_graph(S, int(A.sum())))):
        if G is None: out.update({f"nrmse_{tag}": np.nan, f"r2_{tag}": np.nan, f"slope_{tag}": np.nan}); continue
        sel = select_mask(A, G); a, b, c_ = np.where(sel, r2, 0.0), np.where(sel, sl, 0.0), np.where(sel, nr, 1.0)
        out.update({f"{k}_{tag}": v for k, v in summ(a, b, c_).items()}); out[f"tp_{tag}"] = int(sel.sum())
        out[f"k_{tag}"] = int(G.sum())
    return out


# ------------------------------------------------------------------ do-effect adapter (same propagation as docalc.make_learned, callable predictor)
def make_do(node_pred, order, Xz):
    d = Xz.shape[1]; pos = np.empty(d, int); pos[order] = np.arange(d)
    resid = np.stack([Xz[:, e] - node_pred(Xz, e) for e in range(d)], 1)

    def do_curve(c, vals, nrows=400):
        rows = np.arange(min(nrows, len(Xz))); out = []
        for v in vals:
            x = Xz[rows].astype(np.float64).copy(); x[:, c] = v
            for e in order[pos[c] + 1:]: x[:, e] = node_pred(x, e) + resid[rows, e]
            out.append(x)
        return np.stack(out)
    return do_curve


# ------------------------------------------------------------------ SSA (copied branch of research/adaptive/bench.py)
def run_ssa(X, A, f, sd, Xz):
    import scale_instantaneous as SI, spade_variants as SV
    mm, S0, Xt_, t_ss = next(iter(B.SS_CACHE.values())); d_ = X.shape[1]; t0 = time.time()
    mk = mm.selfmask.squeeze(-1).detach().clone()
    with torch.no_grad():
        n1 = torch.linalg.vector_norm(mm.coef * mm.selfmask, dim=2); scr = (mk > 0); w = (n1 + 1e-2) ** (-1.0); w = w / w[scr].mean()
    ma = SV.fit(Xt_, 0, SI.lg_rule(d_, 0.2), ep=400, weights=w, mask=mk); S = SI.chunked_importance(ma, Xt_); t_fit = time.time() - t0
    Bm = ma._bases(Xt_).detach(); cf = (ma.coef * ma.selfmask).detach()
    cr = B.edge_curve_metrics(lambda c, e, xs: (Bm[:, c, :] @ cf[e, c]).numpy(), Xz, A, f, sd)
    B.add_do(cf.numpy(), Xz, A, 0)
    return S, cr, t_ss + t_fit


# ------------------------------------------------------------------ driver
OURS = ("SSC", "SSA", "SSR_k1.5", "BASILWs", "GAM-ridge")
OWN_RULE = {"SSR_k1.5": lambda S: S > 1e-6, "BASILWs": lambda S: S > 0.5}
CACHE = {}; CACHE_T = {}


COLS = ["cell", "d", "n", "noise", "mech", "seed", "method", "auroc", "auprc", "skel_auroc", "orient", "n_edges", "nnz",
        "nrmse_raw", "r2_raw", "slope_raw", "nrmse_own", "r2_own", "slope_own", "tp_own", "k_own", "nrmse_ownmodel",
        "nrmse_topk", "r2_topk", "slope_topk", "tp_topk", "k_topk", "own_nedges", "own_tp", "own_fp", "own_rev",
        "do_direct", "do_indirect", "n_do_d", "n_do_i", "sk_do", "do_error", "cover", "recall", "time", "time_def", "error"]


def graph_stats(G, A):
    if G is None: return dict(own_nedges=np.nan, own_tp=np.nan, own_fp=np.nan, own_rev=np.nan)
    tp = int((G & (A > 0)).sum()); rev = int((G & (A.T > 0) & ~(A > 0)).sum())
    return dict(own_nedges=int(G.sum()), own_tp=tp, own_fp=int(G.sum()) - tp, own_rev=rev)


def run(name, X, A, f, sd, sim, key):
    Xz = B.zscore(X).astype(np.float64); row = {}; B.SIM[0] = sim; B.EXTRA.clear(); EDGE.clear()
    if name in OURS:
        if name == "SSA":
            if not B.SS_CACHE or next(iter(B.SS_CACHE)) != (0, X.shape, float(X.sum())): B.run_method("SSC", X, A, f, sd, 0)
            B.EXTRA.clear(); EDGE.clear(); S, cr, cost = run_ssa(X, A, f, sd, B.zscore(X))
        else:
            S, cr, cost = B.run_method(name, X, A, f, sd, 0)
        S = np.asarray(S, float); np.fill_diagonal(S, 0)
        if EDGE:
            og = OWN_RULE[name](S) if name in OWN_RULE else None
            row.update(rules(EDGE['r2'], EDGE['sl'], EDGE['nr'], A, S, og)); row.update(graph_stats(og, A))
        row.update({k: B.EXTRA.get(k, np.nan) for k in ("do_direct", "do_indirect", "n_do_d", "n_do_i", "sk_do", "cover", "recall")})
        row["time"] = cost if np.isfinite(cost) else np.nan; row["time_def"] = "t_cost (screen+fit[+refit/MCMC/extract]); excludes metric and do code" if name.startswith(("SS", "BASIL")) else "n/a"
        return S, row
    t0 = time.time(); t_base_add = 0.0
    try:
        if name == "NOTEARS-MLP": M = EL.NotearsMLP(Xz, 0)
        elif name == "NOTEARS-MLP-gc": M = EL.NotearsMLPgc(Xz, 0)
        elif name == "DAGMA-nonlinear": M = EL.DagmaNL(Xz, 0); CACHE[key] = M
        elif name == "DCE-like":
            if key in CACHE: base = CACHE[key]; t_base = CACHE_T[key]
            else: tb = time.time(); base = EL.DagmaNL(Xz, 0); t_base = time.time() - tb
            t0 = time.time(); M = EL.DceLike(base); t_base_add = t_base
        elif name == "GraN-DAG": M = EL.GraNDag(Xz, 0)
        elif name == "CAM-like": M = EL.CAMLike(Xz, 0)
        elif name in ("NAM-SSC", "NAM-SSR_k1.5"):
            import scale_instantaneous as SI
            if not B.SS_CACHE: B.run_method("SSC", X, A, f, sd, 0)
            mk = next(iter(B.SS_CACHE.values()))[0].selfmask.squeeze(-1).detach().numpy(); d_ = X.shape[1]; t0 = time.time()
            if name == "NAM-SSC": M = EL.NAMSpade(Xz, 0, SI.lg_rule(d_, 0.2), mask=mk)
            else:
                M0 = EL.NAMSpade(Xz, 0, SI.lg_rule(d_, 0.2), mask=mk); thr = np.sort(M0.S.flatten())[::-1][int(1.5 * d_) - 1]
                m2 = (M0.S.T >= thr).astype(np.float32); np.fill_diagonal(m2, 0); M = EL.NAMSpade(Xz, 0, 0.0, mask=m2)
        else: raise ValueError(name)
    finally: EL.reset_dtype()
    t_fit = time.time() - t0 + (t_base_add if name == "DCE-like" else 0.0)
    if name == "DAGMA-nonlinear": CACHE_T[key] = t_fit
    S = M.S; own = M.G_own
    r2, sl, nr = edge_stats(lambda c, e, xs: M.curve(c, e, xs, "raw"), Xz, A, f, sd)
    row.update(rules(r2, sl, nr, A, S, own)); row.update(graph_stats(own, A))
    if own is not None:
        r2o, slo, nro = edge_stats(lambda c, e, xs: M.curve(c, e, xs, "own") if own[c, e] else np.zeros(len(xs)), Xz, A, f, sd)
        row.update({"nrmse_ownmodel": float(nro.mean())})
    # do-effects
    try:
        dc = make_do(M.node_pred, M.order, Xz); r = do_error(dc, sim, Xz, A, 0)
        row.update(do_direct=r['direct'][0], do_indirect=r['indirect'][0], n_do_d=r['direct'][1], n_do_i=r['indirect'][1], sk_do=r['direct'][2] + r['indirect'][2])
    except Exception as ex: row["do_error"] = repr(ex)[:80]
    row["time"] = t_fit; row["time_def"] = "fit + (for DCE-like: DAGMA fit time + derivative extraction)" if name == "DCE-like" else "fit only"
    return np.asarray(S, float), row


if __name__ == "__main__":
    cell, d, n, noise, lo, hi, methods = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7].split(",")
    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True); out = os.path.join(outdir, cell + ".csv")
    done = set()
    if os.path.exists(out): old = pd.read_csv(out); done = set(zip(old.seed, old.method))
    for s in range(lo, hi):
        if all((s, m) in done for m in methods): continue
        Xd, A, f, sd, sim = B.gen_full(n, d, s, noise, "m1")
        for m in methods:
            if (s, m) in done: continue
            t0 = time.time(); key = (s, d, n, noise)
            try:
                S, row = run(m, Xd, A, f, sd, sim, key); au, ap, ska, oa = B.metrics(S, A)
                rec = dict(cell=cell, d=d, n=n, noise=noise, mech="m1", seed=s, method=m, auroc=au, auprc=ap, skel_auroc=ska, orient=oa, n_edges=int(A.sum()), nnz=int((S > 1e-6).sum()), **row, error="")
            except Exception as ex:
                traceback.print_exc(); EL.reset_dtype()
                rec = dict(cell=cell, d=d, n=n, noise=noise, mech="m1", seed=s, method=m, n_edges=int(A.sum()), error=repr(ex)[:200])
            pd.DataFrame([rec]).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(s, m, "auroc=%.3f" % rec.get("auroc", np.nan), "nrmse_raw=%.3f" % rec.get("nrmse_raw", np.nan), "own=%.3f" % rec.get("nrmse_own", np.nan), "topk=%.3f" % rec.get("nrmse_topk", np.nan),
                  "do=%.2f/%.2f" % (rec.get("do_direct", np.nan), rec.get("do_indirect", np.nan)), "%.0fs" % (time.time() - t0), rec["error"], flush=True)
