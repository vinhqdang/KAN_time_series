"""
G4 PILOT: benchmark-validity audit of the SPADE instantaneous-ANM benchmark.

PILOT PROTOCOL (written before any run; do not edit after results)
------------------------------------------------------------------
Hardware/env: python3.13, OMP_NUM_THREADS=1, torch.set_num_threads(1) (2 only in the
thread-effect arm). Fresh seeds only: 800-829 (the project's tuning used 100-104,
reporting used 0-4). No manuscript/ edits; project code imported read-only.

Part 1  Sortability + baselines. Generator A = project's generate_instantaneous_anm
  (raw, no internal standardization). For d in {6,10,20,50}: seeds 800-807 (d=50: 800-804).
  Measure varsortability and R2-sortability (Reisach et al. 2021/2023: fraction of
  ancestor-descendant pairs (i anc of j) with var_i<var_j, resp. R2_i<R2_j, ties=.5;
  R2_i = R^2 of OLS of X_i on all other variables) on raw X and on z-scored X.
  Baselines (implemented here): var-sortnregress (order by ascending variance on RAW
  data, then BIC-LassoLars of each node on its predecessors; score=|coef|),
  R2-sortnregress (order by ascending R2), random-order-sortnregress (control).
  Methods: SPADE, NOTEARS-linear, DAGMA-linear, GOLEM, NOTEARS-MLP via the project's own
  METHODS (each z-scores internally, so they never see raw variances). Metric: AUROC over
  off-diagonal pairs (project's `score`), AUPRC. Explained-by-sortability = compare each
  method's AUROC to var-sortnregress AUROC and correlate per-dataset AUROC with
  varsortability / R2-sortability.
Part 2  Generator B = same graph/mechanism/noise as A but each variable is z-scored
  immediately after generation (children see standardized parents). Repeat Part 1.
Part 3  Variance floor, SPADE (d=10, generator A):
  (a) between-dataset sd over seeds 800-829 (SPADE and NOTEARS-linear);
  (b) same data (seeds 800-804), method seeds 0-4 -> within-dataset sd;
  (c) same data+seed, torch threads 1 vs 2 -> |dAUROC|, max |dS| (plus a 1-thread repeat);
  (d) same data+seed, float32 vs float64 -> |dAUROC|.
  Floor = pooled sd of paired differences; a claimed gain must exceed ~2x the sd of the
  mean difference (paired over datasets) and the method-seed sd.
Part 4  Checklist (written after results, in README_results.md).
Pre-declared limits: n=1500 only; one generator family; 5-8 seeds per cell so SEs are wide;
no hyperparameter tuning of any method; SCORE/NoGAM/GraN-DAG/DAGMA-nonlinear excluded for cost.
"""
import os, sys, time, logging
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["SPADE_THREADS"] = "1"
import warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from sklearn.linear_model import LassoLarsIC
import instantaneous_dag_benchmark as B
from src.synthetic import generate_instantaneous_anm
OUT = os.path.dirname(os.path.abspath(__file__))


def gen_std(n, d, seed):
    """Generator B: identical RNG stream/graph to generator A but z-score each variable at creation."""
    r = np.random.RandomState(seed)
    density = 2.0 / (d - 1)
    perm = r.permutation(d); A = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a):
            if r.rand() < density: A[perm[b], perm[a]] = 1
    fns = [lambda x: 0.8*np.tanh(1.5*x), lambda x: 0.7*np.sin(1.5*x),
           lambda x: 0.3*(x**2-1.0), lambda x: 0.8*x/(1.0+np.abs(x))]
    f = {}
    for c in range(d):
        for e in range(d):
            if A[c, e]: f[(c, e)] = fns[r.randint(len(fns))]
    X = np.zeros((n, d))
    for e in perm:
        v = r.uniform(-0.7, 0.7, n)
        for c in range(d):
            if A[c, e]: v = v + f[(c, e)](X[:, c])
        X[:, e] = (v - v.mean()) / v.std()
    return X.astype(np.float32), A


def ancestor_pairs(A):
    d = A.shape[0]; R = A.astype(bool).copy()
    for k in range(d): R = R | (R[:, [k]] & R[[k], :])
    return np.argwhere(R)          # (i,j): i is ancestor of j


def r2_all(X):
    X = X - X.mean(0); d = X.shape[1]; C = X.T @ X / len(X)
    P = np.linalg.inv(C + 1e-10*np.eye(d))
    return 1 - 1.0/(np.diag(P)*np.diag(C))


def sortability(vals, A):
    pr = ancestor_pairs(A)
    if len(pr) == 0: return float("nan")
    a, b = vals[pr[:, 0]], vals[pr[:, 1]]
    return float(np.mean((a < b) + 0.5*(a == b)))


def sortnregress(X, order):
    Xz = (X - X.mean(0)) / X.std(0); d = X.shape[1]; W = np.zeros((d, d))
    for k in range(1, d):
        j = order[k]; pa = order[:k]
        m = LassoLarsIC(criterion="bic").fit(Xz[:, pa], Xz[:, j]); W[pa, j] = np.abs(m.coef_)
    return W   # [cause,effect]


def baselines(X, seed):
    X = X.astype(np.float64)
    return {"var-sortnregress": sortnregress(X, np.argsort(X.var(0))),
            "R2-sortnregress": sortnregress(X, np.argsort(r2_all(X))),
            "random-sortnregress": sortnregress(X, np.random.RandomState(seed + 10**6).permutation(X.shape[1]))}


METH = ["SPADE", "NOTEARS-linear", "DAGMA-linear", "GOLEM", "NOTEARS-MLP"]


def part12(gen, d, seeds, csv):
    G = generate_instantaneous_anm if gen == "A_raw" else gen_std
    done = set()
    if os.path.exists(csv):
        old = pd.read_csv(csv); done = set(zip(old.gen, old.d, old.seed, old.method))
    for s in seeds:
        X, A = G(1500, d, seed=s); Xd = X.astype(np.float64); Z = (Xd - Xd.mean(0)) / Xd.std(0)
        meta = dict(gen=gen, d=d, seed=s, vs_raw=sortability(Xd.var(0), A), vs_std=sortability(Z.var(0), A),
                    r2s_raw=sortability(r2_all(Xd), A), r2s_std=sortability(r2_all(Z), A), n_edges=int(A.sum()))
        res = {}
        if (gen, d, s, "var-sortnregress") not in done: res.update(baselines(X, s))
        for m in METH:
            if (gen, d, s, m) in done: continue
            res[m] = B.METHODS[m](X, 0)
        rows = []
        for m, S in res.items():
            au, ap, f1 = B.score(S, A); rows.append(dict(meta, method=m, auroc=au, auprc=ap, f1=f1))
        if rows: pd.DataFrame(rows).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
        print(gen, d, s, {r["method"]: round(r["auroc"], 3) for r in rows}, flush=True)


def cdkan64(X, seed, ep=400, lg=0.01):
    from src.cdkan.causal_kan import CausalKANInstant
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(seed); np.random.seed(seed)
        Xt = torch.tensor(B.zscore(X), dtype=torch.float64); m = CausalKANInstant(X.shape[1], grid_size=8).double()
        opt = torch.optim.Adam(m.parameters(), lr=5e-3); rho, al = 1.0, 0.0
        for e in range(ep):
            opt.zero_grad(); pred = m(Xt); mse = ((pred-Xt)**2).mean(); h = m.h()
            (mse + lg*m.group_lasso() + al*h + 0.5*rho*h*h).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            if (e+1) % 50 == 0:
                with torch.no_grad(): hv = m.h().item()
                if hv > 1e-8: al += rho*hv; rho = min(rho*2, 1e10)
        return m.importance(Xt).detach().numpy().T
    finally:
        torch.set_default_dtype(torch.float32)


def part3(which):
    d = 10; rows = []
    path = os.path.join(OUT, f"variance_floor_{which}.csv")
    def add(**k):
        rows.append(k); pd.DataFrame(rows).to_csv(path, index=False)
    a = None
    if which == "between":
        for s in range(800, 830):
            X, A = generate_instantaneous_anm(1500, d, seed=s)
            for m in ["SPADE", "NOTEARS-linear"]:
                add(arm="between", seed=s, mseed=0, method=m, auroc=B.score(B.METHODS[m](X, 0), A)[0])
            print("between", s, rows[-2]["auroc"], rows[-1]["auroc"], flush=True)
    if which == "within":
        for s in range(800, 805):
            X, A = generate_instantaneous_anm(1500, d, seed=s)
            for ms in range(5):
                add(arm="mseed", seed=s, mseed=ms, method="SPADE", auroc=B.score(B.cdkan(X, ms), A)[0])
            S1 = B.cdkan(X, 0); torch.set_num_threads(2); S2 = B.cdkan(X, 0); torch.set_num_threads(1)
            S64 = cdkan64(X, 0); S1b = B.cdkan(X, 0)
            au = lambda S: B.score(S, A)[0]
            nrm = lambda S: np.abs(S)/np.abs(S).max()
            for arm, S in (("repeat_1thr", S1b), ("thr2", S2), ("float64", S64)):
                add(arm=arm, seed=s, mseed=0, method="SPADE", auroc=au(S), dAUROC=au(S)-au(S1),
                    maxdS=float(np.abs(nrm(S)-nrm(S1)).max()))
            print("within", s, rows[-3:], flush=True)


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "p12":
        gen, d, lo, hi = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        part12(gen, d, range(lo, hi), os.path.join(OUT, f"results_{gen}.csv"))
    else:
        part3(a)
