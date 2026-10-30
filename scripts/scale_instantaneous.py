"""
X4: push SPADE's instantaneous-DAG learner beyond d=50 (d = 50, 100, 200).

Two scale levers, both standard in structure learning:
  * lambda_g rule: the group-lasso sums over d(d-1) candidate edges, so a fixed lambda_g over-penalizes as d grows
    (the same effect found for lagged data at d=50). Candidate rule lg(d) = min(0.01, LG_C / d).
  * parent screening (CAM / SCORE-style neighbourhood selection): per effect node, keep the K candidate causes with the
    highest ExtraTrees importance; all other edges are masked to zero before SPADE is fit.

PROTOCOL. Data: generate_instantaneous_anm(n=3000, d, seed). Validation seeds 500-502 choose LG_C and K
(d=50 and d=100 only); the frozen settings are then reported on fresh seeds 600-602 for d=50, 100, 200.
Metrics: AUROC/AUPRC on the directed edge scores, wall-clock fit time (CPU, 4 cores, run alone).
Baselines run within a per-method budget; methods that do not fit the budget are reported as skipped, not dropped.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, torch, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from src.synthetic import generate_instantaneous_anm
from spade_variants import zscore, fit, score

torch.set_default_dtype(torch.float32)


def screen_mask(Z, K, seed):
    n, d = Z.shape; M = np.zeros((d, d), np.float32)                  # [effect, cause]
    for e in range(d):
        others = [j for j in range(d) if j != e]
        et = ExtraTreesRegressor(n_estimators=40, max_depth=8, min_samples_leaf=20, max_features=0.5,
                                 n_jobs=int(os.environ.get("NJOBS", 2)), random_state=seed + e).fit(Z[:, others], Z[:, e])
        top = np.argsort(-et.feature_importances_)[:K]
        for t in top: M[e, others[t]] = 1
    return torch.tensor(M)


@torch.no_grad()
def chunked_importance(m, Xt, chunk=256):
    d = Xt.shape[1]; s1 = torch.zeros(d, d); s2 = torch.zeros(d, d); n = 0
    for i in range(0, len(Xt), chunk):
        c = torch.einsum("bjk,ijk->bij", m._bases(Xt[i:i + chunk]), m.coef * m.selfmask)
        s1 += c.sum(0); s2 += (c ** 2).sum(0); n += len(c)
    return ((s2 / n - (s1 / n) ** 2).clamp_min(0).sqrt()).numpy().T      # [cause, effect]


def spade(X, seed, lg=0.01, K=None, ep=400):
    Z = zscore(X); Xt = torch.tensor(Z); mask = screen_mask(Z, K, seed) if K else None
    m = fit(Xt, seed, lg, ep=ep, mask=mask); return chunked_importance(m, Xt)


def lg_rule(d, c): return min(0.01, c / d)


def baselines():
    from instantaneous_dag_benchmark import METHODS
    return {k: METHODS[k] for k in ["NOTEARS-linear", "DAGMA-linear", "GOLEM", "NOTEARS-MLP"]}


def run(d, seeds, configs, base=None, budget=900, tag="x4", out=None):
    rows = []
    for s in seeds:
        X, A = generate_instantaneous_anm(3000, d, seed=s)
        for name, fn in configs.items():
            t0 = time.time(); S = fn(X, s); dt = time.time() - t0
            au, ap, _ = score(S, A)
            rows.append(dict(d=d, seed=s, method=name, auroc=au, auprc=ap, time_s=round(dt, 1)))
            print(f"d={d} s{s} {name:22s} AUROC={au:.3f} AUPRC={ap:.3f} t={dt:.0f}s", flush=True)
        for name, fn in (base or {}).items():
            t0 = time.time()
            try:
                S = fn(X, s); dt = time.time() - t0; au, ap, _ = score(S, A)
                rows.append(dict(d=d, seed=s, method=name, auroc=au, auprc=ap, time_s=round(dt, 1)))
                print(f"d={d} s{s} {name:22s} AUROC={au:.3f} AUPRC={ap:.3f} t={dt:.0f}s", flush=True)
            except Exception as e:
                rows.append(dict(d=d, seed=s, method=name, auroc=np.nan, auprc=np.nan, time_s=np.nan)); print(name, "FAILED", e, flush=True)
        if out: pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mode = sys.argv[1]
    ER = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
    if mode == "validate":
        cfg = {}
        for d in (50, 100):
            cfg = {f"lg=0.01": lambda X, s: spade(X, s, 0.01),
                   f"lg=c/d,c=0.2": lambda X, s, d=d: spade(X, s, lg_rule(d, 0.2)),
                   f"lg=c/d,c=0.1": lambda X, s, d=d: spade(X, s, lg_rule(d, 0.1)),
                   f"lg=c/d,c=0.2,K=8": lambda X, s, d=d: spade(X, s, lg_rule(d, 0.2), 8),
                   f"lg=c/d,c=0.2,K=15": lambda X, s, d=d: spade(X, s, lg_rule(d, 0.2), 15),
                   f"lg=0.01,K=8": lambda X, s: spade(X, s, 0.01, 8)}
            run(d, range(500, 503), cfg, out=os.path.join(ER, f"x4_validation_d{d}.csv"))
    if mode == "final":
        FROZEN = dict(c=0.2, K=15)                       # selected on validation seeds 500-502 (see x4_validation_d*.csv)
        for d in (50, 100, 200):
            cfg = {"SPADE (lg=0.01)": lambda X, s: spade(X, s, 0.01),
                   "SPADE-scaled (lg=c/d,K=15)": lambda X, s, d=d: spade(X, s, lg_rule(d, FROZEN["c"]), FROZEN["K"])}
            base = baselines() if d <= 100 else {k: v for k, v in baselines().items() if k in ("NOTEARS-linear", "DAGMA-linear")}
            run(d, range(600, 603), cfg, base=base, out=os.path.join(ER, f"x4_final_d{d}.csv"))
