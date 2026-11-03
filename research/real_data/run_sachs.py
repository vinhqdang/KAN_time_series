"""Sachs real-data evaluation (protocol: research/real_data/PLAN.md). Resume-safe; 1 CPU thread.
Usage: python3 run_sachs.py [methods comma list]  -> results/structure.csv, results/scores/*.npy, results/curves/*.npz"""
import os, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["NJOBS"] = "1"; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "research/spire")]
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from sklearn.metrics import roc_auc_score, average_precision_score
import bench as B                                  # read-only reuse of research/spire/bench.py (sets paths, imports)
from basil import fit_basil
from basil_s import fit_basil_s
import scale_instantaneous as SI
import sachs_benchmark as SB
import instantaneous_dag_benchmark as IB

RES = os.path.join(HERE, "results"); os.makedirs(os.path.join(RES, "scores"), exist_ok=True); os.makedirs(os.path.join(RES, "curves"), exist_ok=True)
GRID = np.linspace(-1.8, 1.8, 50)
METHODS = ["SSC", "SSR_k1.5", "BASILW", "BASIL", "GAM-ridge", "NOTEARS-linear", "DAGMA-linear", "var-sortnregress", "var-sortnregress-logscale"]


def metrics(S, A):
    d = A.shape[0]; m = ~np.eye(d, dtype=bool); S = np.nan_to_num(np.abs(np.asarray(S, float)))
    p, t = S[m], A[m]
    au = roc_auc_score(t, p); ap = average_precision_score(t, p)
    iu = np.triu_indices(d, 1); ska = roc_auc_score((A | A.T)[iu], np.maximum(S, S.T)[iu])
    o = [1.0 if S[c, e] > S[e, c] else (0.5 if S[c, e] == S[e, c] else 0.0) for c, e in zip(*np.nonzero(A))]
    top = np.argsort(-S.flatten() * (~np.eye(d, dtype=bool).flatten()), kind="stable")[:18]
    hits = int(A.flatten()[top].sum())
    return dict(auroc=au, auprc=ap, skel_auroc=ska, orient=float(np.mean(o)), hits18=hits, nnz=int((S[m] > 1e-6).sum()))


def curves_to_grid(curve_fn, Xz):
    """curve_fn(c,e) -> values on the data rows; returns grid curves [d,d,50] and amplitude (std on data rows) [d,d]."""
    d = Xz.shape[1]; G = np.zeros((d, d, len(GRID))); amp = np.zeros((d, d))
    for c in range(d):
        o = np.argsort(Xz[:, c]); xs = Xz[o, c]
        for e in range(d):
            if c == e: continue
            y = np.asarray(curve_fn(c, e), float); y = y - y.mean(); amp[c, e] = y.std()
            G[c, e] = np.interp(GRID, xs, y[o])
    return G, amp


def run_one(name, Xz, seed, ss):
    d = Xz.shape[1]; Xt = torch.tensor(Xz.astype(np.float32)); C = None
    def sp_curve(model):
        Bm = model._bases(Xt).detach(); cf = (model.coef * model.selfmask).detach()
        return lambda c, e: (Bm[:, c, :] @ cf[e, c]).numpy()
    if name == "SSC":
        S = ss["S0"]; C = curves_to_grid(sp_curve(ss["mm"]), Xz)
    elif name.startswith("SSR"):
        kn = int(float(name.split("_k")[1]) * d); S0 = ss["S0"]; thr = np.sort(S0.flatten())[::-1][kn - 1]
        mk2 = torch.tensor((S0.T >= thr).astype(np.float32)); np.fill_diagonal(mk2.numpy(), 0)
        m2 = SI.fit(Xt, seed, 0.0, ep=400, mask=mk2); S = SI.chunked_importance(m2, Xt); C = curves_to_grid(sp_curve(m2), Xz)
    elif name == "BASILW":
        S0 = ss["S0"]; inw = S0.sum(0).copy(); order0 = []
        for _ in range(d):
            i = int(np.argmin(inw)); order0.append(i); inw[i] = np.inf; inw -= np.where(np.isinf(inw), 0, S0[i, :])
        o = fit_basil_s(Xz, seed, m=20, iters=200 * d, chains=2, burn_frac=0.3, select=True, cand_score=S0, order0=order0)
        S = o["P"]; C = curves_to_grid(lambda c, e: o["mean_curve"](c, e), Xz)
    elif name == "BASIL":
        n_ = len(Xz); o = fit_basil(Xz, seed, g=1.0 * n_, pi=2.0 / (d - 1), kmax=3)
        S = o["P"]; C = curves_to_grid(lambda c, e: o["mean_curve"](c, e), Xz)
    elif name == "GAM-ridge":
        S, cur = B.G2.fit_gam(Xz, seed); C = curves_to_grid(lambda c, e: cur(c, e, Xz[:, c]), Xz)
    elif name in ("NOTEARS-linear", "DAGMA-linear"):
        S = IB.METHODS[name](Xz, seed)
    elif name == "var-sortnregress":
        S = B.sortnregress(Xz.astype(np.float64), np.argsort(Xz.var(0)))              # z-scored: variances all equal, order = float noise
    else:
        raise ValueError(name)
    return np.asarray(S, float), C


def main():
    X, A, names = SB.load(); n_all = len(X)
    methods = sys.argv[1].split(",") if len(sys.argv) > 1 else METHODS
    out = os.path.join(RES, "structure.csv")
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {(int(r["sub"]), r["method"]) for r in rows}
    for r in range(10):
        Xs = X[np.random.RandomState(r).choice(n_all, 2000, replace=False)]; Xz = ((Xs - Xs.mean(0)) / Xs.std(0)).astype(np.float32)
        ss = None
        for m in methods:
            if (r, m) in done: continue
            t0 = time.time()
            try:
                if m == "var-sortnregress-logscale":      # extra, not in PLAN: order by variance of the log data BEFORE z-scoring
                    S = B.sortnregress(Xz.astype(np.float64), np.argsort(Xs.var(0))); C = None
                else:
                    if m in ("SSC", "SSR_k1.5", "BASILW") and ss is None:
                        Zs = SI.zscore(Xz); Xt_ = torch.tensor(Zs); tt = time.time(); mk = SI.screen_mask(Zs, 15, r)
                        mm = SI.fit(Xt_, r, SI.lg_rule(11, 0.2), ep=400, mask=mk); ss = dict(mm=mm, S0=SI.chunked_importance(mm, Xt_), t=time.time() - tt)
                    S, C = run_one(m, Xz, r, ss)
                met = metrics(S, A); np.save(os.path.join(RES, "scores", f"{m}_{r}.npy"), S)
                if C is not None: np.savez_compressed(os.path.join(RES, "curves", f"{m}_{r}.npz"), grid=C[0], amp=C[1])
                rows.append(dict(sub=r, method=m, time=round(time.time() - t0 + (ss["t"] if ss and m in ("SSR_k1.5", "BASILW") else 0), 1), **met))
                print(f"[r{r}] {m:26s} AUROC={met['auroc']:.3f} AUPRC={met['auprc']:.3f} orient={met['orient']:.2f} hits18={met['hits18']} t={time.time()-t0:.0f}s", flush=True)
            except Exception as ex:
                import traceback; traceback.print_exc(); print(f"[r{r}] {m} FAILED {ex}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
