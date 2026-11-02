"""
G3 / RQ-C PILOT PROTOCOL (written before the validation and test runs; compute budget about 75 min total on ONE thread,
OMP_NUM_THREADS=1 and torch.set_num_threads(1), machine shared with four other workers).

Question: can an order-restricted, screened, spline-additive learner (SPOT, see spot.py) keep per-edge spline functions
at d = 50, 100, 200 with n = 3000 and beat the linear learners that actually run at that size, and what does it lose to
its order estimate?

Data. n = 3000 always. Two generators, both ER with expected d edges and random node permutation:
  std : sortability-controlled, internally standardized SCM (gen_metrics.gen_std; unit-variance signal at every node,
        Gaussian noise sd 0.5). varsortability of the truth is printed for both generators.
  old : the existing generator src.synthetic.generate_instantaneous_anm (uniform noise; tends to be varsortable).
Seeds. Choices (K, lambda, adaptive) are made on VALIDATION seeds 700-701 (d=50) and 700 (d=100) for BOTH generators,
by mean AUPRC (AUROC tie-break), then FROZEN. Debugging of the order estimator used validation seeds 700-702 at d=15-20
only. TEST seeds are 800 and 801 (d=50, 100) and 800 (d=200); disjoint from validation and from every earlier SPADE
seed (0-4, 100-119, 200-219, 300-319, 500-502, 600-602). No tuning on test seeds. Two test seeds = a pilot; no
significance claims.
Metrics (on the directed [cause, effect] score matrix): AUROC and AUPRC over all d(d-1) ordered pairs; best-threshold F1
(oracle threshold, optimistic); F1 at a label-free graph (support of the group-lasso for SPOT, |w|>0.3 for the linear
baselines on z-scored data, the returned graph for SCORE); recall at precision >= 0.5 and >= 0.8; wall-clock seconds for
the whole fit, 1 thread; for SPOT also the fraction of true edges whose direction agrees with the estimated order
(order_agree = recall ceiling) and a DIAGNOSTIC oracle-order run (true topological order, uses labels, NOT a method).
Methods:
  SPOT                  : the new pilot algorithm (order -> screen -> group-lasso splines).
  SPOT-oracle-order     : diagnostic only.
  NOTEARS-linear, DAGMA-linear : from scripts/instantaneous_dag_benchmark.py (same code as earlier experiments).
  SCORE (dodiscover)    : Stein ordering + CAM pruning, d=50 only on one seed and only if it finishes within 8 min;
                          larger sizes are expected not to finish at 1 thread and are reported as unavailable, not dropped.
  SPADE-screened (earlier ExtraTrees-screened SPADE with lambda_g=0.2/d, K=15): d=50 only, one seed (its d=100/200 cost
                          was 300-540 s at 4 threads in the earlier study; at 1 thread it does not fit this budget).
Non-linear baselines (NOTEARS-MLP, GOLEM, GraN-DAG, DAGMA-MLP, NoGAM) are NOT run beyond d=50: earlier logs already show
more than 10 min per fit at d=50 on 4 threads. This is a budget cut, not a result.
"""
import os, sys, time, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("SPADE_THREADS", "1")
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spot import spot, zscore
from gen_metrics import gen_std, gen_old, metrics, topo_order, order_agree, varsortability

GENS = {"std": gen_std, "old": gen_old}
GRID = [(K, lam, ad) for K in (8, 15) for lam in (0.01, 0.03) for ad in (False, True)]


def validate(out):
    rows = []
    for gname in ("std", "old"):
        for d, seeds in ((50, (700, 701)), (100, (700,))):
            for s in seeds:
                X, A = GENS[gname](3000, d, s); cache = {}
                for K, lam, ad in GRID:
                    S, sup, info = spot(X, s, K=K, lam=lam, adaptive=ad, cache=cache)
                    m = metrics(S, A, sup); m.update(gen=gname, d=d, seed=s, K=K, lam=lam, adaptive=ad, order_agree=order_agree(info["order"], A))
                    rows.append(m); print(gname, d, s, K, lam, ad, round(m["auroc"], 3), round(m["auprc"], 3), flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows); g = df.groupby(["K", "lam", "adaptive"])[["auprc", "auroc", "f1_fixed"]].mean().sort_values(["auprc", "auroc"], ascending=False)
    print(g.round(3)); return g


def run_test(gname, d, seed, methods, out):
    X, A = GENS[gname](3000, d, seed); vs = varsortability(X, A); rows = []
    FROZEN = json.load(open(os.path.join(HERE, "frozen.json")))
    for mth in methods:
        t0 = time.time(); extra = {}; binary = None
        try:
            if mth == "SPOT":
                S, binary, info = spot(X, seed, **FROZEN); extra = dict(order_agree=order_agree(info["order"], A), t_order=info["t_order"], t_screen=info["t_screen"], t_fit=info["t_fit"])
            elif mth == "SPOT-oracle-order":
                S, binary, info = spot(X, seed, order=topo_order(A), **FROZEN)
            elif mth in ("NOTEARS-linear", "DAGMA-linear", "SCORE"):
                import instantaneous_dag_benchmark as B
                S = {"NOTEARS-linear": B.notears_linear, "DAGMA-linear": lambda X, s: B.dagma_linear(X), "SCORE": B.score_alg}[mth](X, seed)
                binary = (np.abs(S) > (0.3 if mth != "SCORE" else 0.5))
            elif mth == "SPADE-screened":
                import scale_instantaneous as SI
                S = SI.spade(X, seed, SI.lg_rule(d, 0.2), 15); binary = None
            m = metrics(S, A, binary); m.update(extra)
        except Exception as e:
            m = dict(error=str(e)[:200])
        m.update(gen=gname, d=d, seed=seed, method=mth, time_s=round(time.time() - t0, 1), varsort=round(vs, 3))
        rows.append(m); print(m, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--gen", default="std"); ap.add_argument("--d", type=int, default=50)
    ap.add_argument("--seed", type=int, default=800); ap.add_argument("--methods", default="SPOT"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); res = os.path.join(HERE, "results"); os.makedirs(res, exist_ok=True)
    if a.mode == "validate": validate(os.path.join(res, "validation.csv"))
    else: run_test(a.gen, a.d, a.seed, a.methods.split(","), a.out or os.path.join(res, f"test_{a.gen}_d{a.d}_s{a.seed}_{a.methods.replace(',', '+')}.csv"))
