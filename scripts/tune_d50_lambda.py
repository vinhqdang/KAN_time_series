"""
Held-out sweep of the group-lasso weight lambda_g for the lagged LINEAR d=50
configuration of scripts/honest_causal_benchmark.py.

Motivation: SPADE's linear d=50 AUROC was 0.829 under the benchmark's default
lambda_g=0.01 (which works well at d<=20), a large gap versus GOLEM's 1.000 --
the paper's most prominent remaining weak point at the time. group_lasso()
sums a per-edge norm over all d*(d-1)*max_lag candidate edges, so a fixed
lambda_g becomes a progressively stronger penalty relative to the MSE term as
d grows; this script checks whether a smaller lambda_g recovers the gap,
using validation seeds 142-144 -- disjoint from the official reporting seeds
42-44 used everywhere else for this benchmark -- so the choice is not tuned
on the evaluation data. Run this before, not after, checking the official
seeds; the chosen value is then applied once, on 42-44, in
honest_causal_benchmark.py's fit_cdkan call site (lg=0.002 for d>=50).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, torch
torch.set_num_threads(4)
import honest_causal_benchmark as hcb

D, N = 50, 3000
VAL_SEEDS = [142, 143, 144]  # disjoint from the official reporting seeds 42-44


def fit_cdkan_lg(X, seed, max_lag=3, epochs=150, lr=5e-3, lg=0.01, grid_size=8):
    from src.cdkan.causal_kan import CausalKAN
    torch.manual_seed(seed); np.random.seed(seed)
    d = X.shape[1]
    Xw, yw = hcb.make_windows(X, window=max_lag + 1)
    Xw = torch.tensor(Xw, dtype=torch.float32)
    yw = torch.tensor(yw[:, 0, :], dtype=torch.float32)
    model = CausalKAN(d, max_lag=max_lag, grid_size=grid_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xw)
        loss = ((pred - yw) ** 2).mean() + lg * model.group_lasso()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model.importance(Xw).cpu().numpy()


if __name__ == "__main__":
    print(f"Sweeping lambda_g at d={D} (linear), held-out seeds={VAL_SEEDS}\n")
    for lg in [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02]:
        aus, aps, f1s = [], [], []
        for s in VAL_SEEDS:
            X, true_adj, _ = hcb.make_dataset("linear", N, D, s)
            imp = fit_cdkan_lg(X, s, lg=lg)
            m = hcb.score_adj(imp, true_adj)
            aus.append(m["auroc"]); aps.append(m["auprc"]); f1s.append(m["f1"])
        print(f"lambda_g={lg:<7} AUROC={np.mean(aus):.4f}  "
              f"AUPRC={np.mean(aps):.4f}  F1={np.mean(f1s):.4f}", flush=True)
