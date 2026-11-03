"""Semi-synthetic benchmark runner (protocol: PLAN.md). Resume-safe, 1 thread.
Usage: python3.13 bench.py OUTDIR copies n noise seed_lo seed_hi method1,method2,...
Reuses research/adaptive/bench.py (SSC, SSA, SSR_k, BASILWs, GAM-ridge, var-sortnregress branches) read-only; adds SPADE (plain), NOTEARS-linear,
var-sortnregress-raw and per-edge logging."""
import os, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["SPADE_THREADS"] = "1"; os.environ["NJOBS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path[:0] = [os.path.join(ROOT, "research/adaptive"), HERE]
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
import bench as B                      # research/adaptive/bench.py (read-only reuse; resolves via sys.path)
import gen as G
from g1_pilot import fit as spade_fit
from spade_variants import zscore
import instantaneous_dag_benchmark as IB
torch.set_num_threads(1)   # IB sets SPADE_THREADS (default 4) at import

PER_EDGE = []
COLS = ["d", "n", "noise", "seed", "method", "auroc", "auprc", "skel_auroc", "orient", "curve_r2", "slope", "ece", "nrmse", "do_direct", "do_indirect", "n_do_d", "n_do_i", "sk_do",
        "cover", "recall", "gap", "t_ss", "t_cost", "t_refit", "t_mcmc", "t_extract", "nnz", "n_edges", "time"]


def edge_metrics_logged(curve_fn, X, A, f, sd):
    r2, sl, nr = [], [], []; PER_EDGE.clear()
    for c, e in zip(*np.nonzero(A)):
        gh = np.asarray(curve_fn(c, e, X[:, c]), float); gh = gh - gh.mean()
        gt = f[(c, e)](X[:, c].astype(float)) / sd[e]; gt = gt - gt.mean()
        a = np.polyfit(X[:, c].astype(float), gt, 1); gl = np.polyval(a, X[:, c].astype(float)); nl = 1 - ((gt - gl) ** 2).sum() / max((gt ** 2).sum(), 1e-12) if False else ((gt - (gl - gl.mean())) ** 2).sum() / max((gt ** 2).sum(), 1e-12)
        r2.append(1 - ((gh - gt) ** 2).sum() / (gt ** 2).sum()); sl.append((gh * gt).sum() / (gt ** 2).sum()); nr.append(np.sqrt(((gh - gt) ** 2).sum() / (gt ** 2).sum()))
        PER_EDGE.append(dict(c=int(c), e=int(e), gt_amp=float(gt.std()), gt_nonlin=float(nl), nrmse_e=float(nr[-1]), slope_e=float(sl[-1])))
    B.EXTRA['nrmse'] = float(np.mean(nr))
    return float(np.mean(r2)), float(np.median(sl))


B.edge_curve_metrics = edge_metrics_logged


def run(name, X, A, f, sd, raw_sd):
    PER_EDGE.clear(); t0 = time.time()
    if name == "SPADE":
        B.EXTRA.clear(); Xz = zscore(X); Xt = torch.tensor(Xz); m = spade_fit(Xt, 0); S = m.importance(Xt).numpy().T
        Bm = m._bases(Xt).detach(); cf = (m.coef * m.selfmask).detach()
        cr = edge_metrics_logged(lambda c, e, xs: (Bm[:, c, :] @ cf[e, c]).numpy(), Xz, A, f, sd); B.add_do(cf.numpy(), Xz, A, 0)
        return np.asarray(S, float), cr, time.time() - t0
    if name == "NOTEARS-linear":
        B.EXTRA.clear(); return np.asarray(IB.METHODS[name](X, 0), float), (np.nan, np.nan), time.time() - t0
    if name == "var-sortnregress-raw":
        B.EXTRA.clear(); return B.sortnregress(X.astype(np.float64), np.argsort(raw_sd)), (np.nan, np.nan), time.time() - t0
    S, cr, t = B.run_method(name, X, A, f, sd, 0)
    return S, cr, t


if __name__ == "__main__":
    out, copies, n, noise, lo, hi, methods = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7].split(",")
    os.makedirs(out, exist_ok=True); os.makedirs(os.path.join(out, "edges"), exist_ok=True)
    sem = G.fit_sem(); gen = G.make_gen(sem, copies); d = 11 * copies
    for s in range(lo, hi):
        fn = os.path.join(out, f"d{d}_n{n}_{noise}_s{s}.csv"); done = set(pd.read_csv(fn).method) if os.path.exists(fn) else set()
        todo = [m for m in methods if m not in done]
        if not todo: continue
        X, A, f, sd, sim, raw_sd = gen(n, s, noise); B.SIM[0] = sim
        for m in todo:
            S, cr, t = run(m, X, A, f, sd, raw_sd); au, ap, ska, oa = B.metrics(S, A)
            row = dict(d=d, n=n, noise=noise, seed=s, method=m, auroc=au, auprc=ap, skel_auroc=ska, orient=oa, curve_r2=cr[0], slope=cr[1],
                       ece=B.ece(S, A) if m.startswith("BASIL") else np.nan, **B.EXTRA, nnz=int((S > 1e-6).sum()), n_edges=int(A.sum()), time=t)
            pd.DataFrame([row]).reindex(columns=COLS).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)   # fixed columns: EXTRA keys differ by method
            if PER_EDGE:
                pe = pd.DataFrame(PER_EDGE); pe.insert(0, "method", m); pe.insert(0, "seed", s); pe.insert(0, "n", n); pe.insert(0, "noise", noise); pe.insert(0, "d", d)
                efn = os.path.join(out, "edges", os.path.basename(fn)); pe.to_csv(efn, mode="a", header=not os.path.exists(efn), index=False)
            print(s, m, round(au, 3), round(row.get("nrmse", np.nan), 3), f"{t:.0f}s", flush=True)
