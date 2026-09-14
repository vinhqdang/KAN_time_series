"""
Held-out sweep of the group-lasso weight lambda_g for the headline instantaneous-DAG
task (Section "Headline" of the manuscript, scripts/instantaneous_dag_benchmark.py).

Motivation: the ablation study (scripts/ablation_scale.py, on a different 5-node
lagged benchmark) found that removing the group-lasso raises AUROC substantially
(0.902 -> 0.984), i.e. the sparsity penalty currently used (lambda_g=0.02 in
instantaneous_dag_benchmark.py's cdkan()) may be trading away ranking quality
(AUROC/AUPRC) for sparsity on this style of benchmark too. This script checks
that directly on the ACTUAL headline benchmark (not the ablation's benchmark),
at d=6, across several lambda_g values, using seeds disjoint from the official
reporting seeds (0-4 for d=6/10, 0-2 for d=20 in instantaneous_dag_benchmark.py's
CONFIGS) so the choice is not tuned on the evaluation data.

CORRECTION (found on independent review of this revision): an earlier version of
this script hardcoded seeds=[0,1,2,3,4] -- identical to the official reporting
seeds -- which contradicted the "held-out validation seeds 100-104" claim made in
the manuscript text and in instantaneous_dag_benchmark.py's comment. That was a
real bug, not a documentation lag: the lambda_g=0.005 value actually applied had
been chosen on the leaked (reporting-seed) sweep, not a genuine held-out one. This
script's default seeds are now 100-104 as the text always claimed, and re-running
the sweep on those seeds finds lambda_g=0.01 (not 0.005) is the best-performing
value -- see the printed sweep below. instantaneous_dag_benchmark.py's cdkan() now
uses lg=0.01 accordingly.
"""
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
torch.set_num_threads(4)
from src.synthetic import generate_instantaneous_anm
from src.cdkan.causal_kan import CausalKANInstant
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def zscore(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def score(S, A):
    d = A.shape[0]; m = ~np.eye(d, dtype=bool)
    p = np.nan_to_num(np.abs(S)[m].astype(float)); t = A[m].astype(int)
    if p.max() > 0: p = p / p.max()
    au = roc_auc_score(t, p) if t.sum() and (1 - t).sum() else float("nan")
    ap = average_precision_score(t, p) if t.sum() else float("nan")
    f1 = max(f1_score(t, (p >= q).astype(int), zero_division=0) for q in np.unique(np.r_[0, p, 1]))
    return au, ap, f1


def cdkan(X, seed, lg, ep=400):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(zscore(X), dtype=torch.float32); m = CausalKANInstant(X.shape[1], grid_size=8)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3); rho, al = 1.0, 0.0
    for e in range(ep):
        opt.zero_grad(); pred = m(Xt); mse = ((pred - Xt) ** 2).mean(); h = m.h()
        (mse + lg * m.group_lasso() + al * h + 0.5 * rho * h * h).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if (e + 1) % 50 == 0:
            with torch.no_grad(): hv = m.h().item()
            if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
    return m.importance(Xt).numpy().T


if __name__ == "__main__":
    d = 6
    lambdas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04]
    seeds = [100, 101, 102, 103, 104]  # held out; disjoint from the official reporting seeds 0-4/0-2
    print(f"Sweeping lambda_g at d={d}, held-out seeds={seeds}\n")
    for lg in lambdas:
        aus, aps, f1s, ts = [], [], [], []
        for s in seeds:
            X, A = generate_instantaneous_anm(n_samples=1500, n_nodes=d, seed=s)
            t0 = time.time()
            S = cdkan(X, s, lg)
            ts.append(time.time() - t0)
            au, ap, f1 = score(S, A)
            aus.append(au); aps.append(ap); f1s.append(f1)
        print(f"lambda_g={lg:<6} AUROC={np.mean(aus):.3f}+/-{np.std(aus):.3f}  "
              f"AUPRC={np.mean(aps):.3f}+/-{np.std(aps):.3f}  F1={np.mean(f1s):.3f}  "
              f"t={np.mean(ts):.1f}s", flush=True)
