"""
X2: real-data benchmark WITH a ground-truth graph: Sachs et al. (2005) protein-signalling network,
11 phospho-proteins/phospholipids, 7,466 single-cell flow-cytometry measurements (pooled across
experimental conditions; interventions are not used), consensus graph with 18 directed edges
(data/sachs/cyto_full_target.csv, as distributed with the Causal Discovery Toolbox).

PRE-REGISTERED PROTOCOL (fixed before any method was run on these data)
 * preprocessing: natural log of each (strictly positive) measurement, then z-score.
 * PRIMARY: 10 random subsamples of n=2,000 cells (seed 0..9; the subsample, and each method's seed, is the
   resample index), the same matrix for every method. SCALE CHECK: full data (n=7,466), one run,
   SPADE variants and the fast baselines only.
 * metrics on the 110 possible directed edges vs the 18 true edges: AUROC, AUPRC (chance = 0.164),
   best-threshold F1 (same rule for every method), and top-18 hits (true edges among the 18
   highest-scored edges; for binary-output methods, among their output edges, with SHD).
 * methods: the 8 baselines of the instantaneous benchmark + SPADE variants named on the command line.
 * no hyperparameter is tuned on Sachs; every method uses its fixed benchmark settings.
 * results are reported whatever they are.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
CHANCE_NOTE = "chance AUPRC = 18/110 = 0.164"


def load():
    X = pd.read_csv(os.path.join(ROOT, "data", "sachs", "cyto_full_data.csv"))
    T = pd.read_csv(os.path.join(ROOT, "data", "sachs", "cyto_full_target.csv"))
    names = list(X.columns); idx = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(names), len(names)), int)
    for c, e in zip(T.Cause, T.Effect): A[idx[c], idx[e]] = 1
    assert A.sum() == 18, A.sum()
    return np.log(X.to_numpy(float)), A, names


def metrics(S, A):
    d = A.shape[0]; m = ~np.eye(d, dtype=bool)
    p = np.nan_to_num(np.abs(np.asarray(S, float))[m]); t = A[m].astype(int)
    binary = set(np.unique(p)).issubset({0.0, 1.0})
    pn = p / p.max() if p.max() > 0 else p
    au = roc_auc_score(t, pn); ap = average_precision_score(t, pn)
    f1 = max(f1_score(t, (pn >= q).astype(int), zero_division=0) for q in np.unique(np.r_[0, pn, 1]))
    k = 18; order = np.argsort(-p, kind="stable")
    chosen = (p > 0).nonzero()[0] if binary else order[:k]
    hits = int(t[chosen].sum()); shd = int(len(chosen) - hits + (t.sum() - hits))
    return dict(auroc=au, auprc=ap, f1=f1, hits=hits, n_out=len(chosen), shd=shd)


def main():
    from instantaneous_dag_benchmark import METHODS
    import spade_variants as V
    X, A, names = load(); n_all = len(X)
    variants = {k: V.VARIANTS[k] for k in sys.argv[1:] if k in V.VARIANTS}
    methods = {k: v for k, v in METHODS.items() if k != "SPADE"}
    methods.update({("SPADE" if k == "base" else f"SPADE-{k}"): v for k, v in variants.items()})
    fast = ["NOTEARS-linear", "DAGMA-linear", "GOLEM", "NOTEARS-MLP"] + [k for k in methods if k.startswith("SPADE")]
    rows = []; out = os.path.join(ROOT, "experimental_results", "sachs_raw.csv")
    jobs = [(r, "primary", X[np.random.RandomState(r).choice(n_all, 2000, replace=False)], list(methods)) for r in range(10)]
    jobs.append((0, "full", X, fast))
    for r, part, Xs, ms in jobs:
        Xs = (Xs - Xs.mean(0)) / Xs.std(0)
        for m in ms:
            fn = methods[m]; t0 = time.time()
            try:
                S = fn(Xs, r) if m not in variants and not m.startswith("SPADE-") and m != "SPADE" else fn(Xs, r)
                met = metrics(S, A)
                rows.append(dict(part=part, resample=r, method=m, status="ok", time_s=round(time.time() - t0, 1), **met))
                print(f"[{part} r{r}] {m:18s} AUROC={met['auroc']:.3f} AUPRC={met['auprc']:.3f} hits={met['hits']}/{met['n_out']} t={time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                rows.append(dict(part=part, resample=r, method=m, status=f"FAILED: {e}", time_s=0)); print(f"[{part} r{r}] {m} FAILED: {e}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows); ok = df[df.status == "ok"]
    print(ok.groupby(["part", "method"])[["auroc", "auprc", "f1", "hits", "shd"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
