"""
HEADLINE experiment: non-linear INSTANTANEOUS DAG discovery.

CD-KAN's distinctive strength is recovering non-linear additive-noise DAGs, where
its learnable B-spline edges + acyclicity constraint beat both linear DAG learners
and fixed-architecture non-linear learners. Baselines:
  - linear NOTEARS (ours), DAGMA-linear      (linear score-based)
  - NOTEARS-MLP (ours), DAGMA-nonlinear-MLP  (SOTA non-linear score-based)
Metrics: threshold-free AUROC/AUPRC (+ best-F1, SHD), multiple seeds, d in {6,10,20}.
Writes experimental_results/instdag_*.csv and figures/instdag.png + tab_instdag.tex.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_num_threads(4)
from src.synthetic import generate_instantaneous_anm
from src.cdkan.causal_kan import CausalKANInstant
from src.cdkan.losses import _h_notears
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

RES = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
FIG = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")


def zscore(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def score(S, A):                         # S, A indexed [cause, effect]
    d = A.shape[0]; m = ~np.eye(d, dtype=bool)
    p = np.nan_to_num(np.abs(S)[m].astype(float)); t = A[m].astype(int)
    if p.max() > 0: p = p / p.max()
    au = roc_auc_score(t, p) if t.sum() and (1 - t).sum() else float("nan")
    ap = average_precision_score(t, p) if t.sum() else float("nan")
    f1 = max(f1_score(t, (p >= q).astype(int), zero_division=0) for q in np.unique(np.r_[0, p, 1]))
    return au, ap, f1


# ---- CD-KAN (ours) : returns [cause, effect] ----
def cdkan(X, seed, ep=400):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(zscore(X), dtype=torch.float32); m = CausalKANInstant(X.shape[1], grid_size=8)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3); rho, al = 1.0, 0.0
    for e in range(ep):
        opt.zero_grad(); pred = m(Xt); mse = ((pred - Xt) ** 2).mean(); h = m.h()
        (mse + 0.02 * m.group_lasso() + al * h + 0.5 * rho * h * h).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if (e + 1) % 50 == 0:
            with torch.no_grad(): hv = m.h().item()
            if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
    return m.importance(Xt).numpy().T          # [effect,cause]->[cause,effect]


# ---- NOTEARS-MLP (ours) : per-node MLP, acyclicity on 1st-layer weight norms ----
def notears_mlp(X, seed, ep=600, hidden=10):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(zscore(X), dtype=torch.float32); d = X.shape[1]
    f32 = torch.float32
    W1 = torch.nn.Parameter(0.1 * torch.randn(d, hidden, d, dtype=f32))   # [effect, hidden, cause]
    b1 = torch.nn.Parameter(torch.zeros(d, hidden, dtype=f32))
    W2 = torch.nn.Parameter(0.1 * torch.randn(d, hidden, dtype=f32))
    b2 = torch.nn.Parameter(torch.zeros(d, dtype=f32))
    mask = (1 - torch.eye(d)).to(f32)                          # [effect, cause]
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-2); rho, al = 1.0, 0.0
    def Amat():
        return torch.linalg.vector_norm(W1 * mask.unsqueeze(1), dim=1).t()  # [cause,effect]
    for e in range(ep):
        opt.zero_grad()
        Wm = W1 * mask.unsqueeze(1)
        hlay = torch.tanh(torch.einsum("bc,ehc->beh", Xt, Wm) + b1)          # [B,effect,hidden]
        pred = torch.einsum("beh,eh->be", hlay, W2) + b2
        mse = ((pred - Xt) ** 2).mean()
        h = _h_notears(Amat())
        (mse + 0.01 * Wm.abs().mean() + al * h + 0.5 * rho * h * h).backward()
        opt.step()
        if (e + 1) % 60 == 0:
            with torch.no_grad(): hv = _h_notears(Amat()).item()
            if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
    return Amat().detach().numpy()                             # [cause,effect]


def dagma_linear(X):
    from dagma.linear import DagmaLinear
    return np.abs(DagmaLinear(loss_type="l2").fit(zscore(X), lambda1=0.02, w_threshold=0.0))
def dagma_nonlinear(X):
    from dagma.nonlinear import DagmaMLP, DagmaNonlinear
    d = X.shape[1]; eq = DagmaMLP(dims=[d, 10, 1], bias=True, dtype=torch.double)
    return np.abs(DagmaNonlinear(eq, dtype=torch.double).fit(
        zscore(X).astype(np.float64), lambda1=0.02, lambda2=0.005, T=4,
        warm_iter=4000, max_iter=8000, w_threshold=0.0))
def notears_linear(X, seed=0, ep=200):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed); Xt = torch.tensor(zscore(X), dtype=torch.float32); d = X.shape[1]
    W = torch.zeros(d, d, dtype=torch.float32, requires_grad=True); opt = torch.optim.Adam([W], lr=1e-2)
    mask = (1 - torch.eye(d)).to(torch.float32); rho, al = 1.0, 0.0
    for it in range(ep):
        opt.zero_grad(); Wm = W * mask; pred = Xt @ Wm
        mse = ((pred - Xt) ** 2).mean(); h = _h_notears(Wm)
        (mse + 0.02 * Wm.abs().sum() + al * h + 0.5 * rho * h * h).backward(); opt.step()
        if (it + 1) % 20 == 0:
            with torch.no_grad(): hv = _h_notears(W * mask).item()
            if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
    return (W * mask).abs().detach().numpy()                   # [cause,effect]


METHODS = {
    "CD-KAN": lambda X, s: cdkan(X, s),
    "DAGMA-nonlinear": lambda X, s: dagma_nonlinear(X),
    "NOTEARS-MLP": lambda X, s: notears_mlp(X, s),
    "DAGMA-linear": lambda X, s: dagma_linear(X),
    "NOTEARS-linear": lambda X, s: notears_linear(X, s),
}


def main():
    import pandas as pd
    configs = [(6, 5), (10, 5), (20, 3)]         # (d, n_seeds)
    rows = []
    for d, nseed in configs:
        for s in range(nseed):
            X, A = generate_instantaneous_anm(n_samples=1500, n_nodes=d, seed=s)
            for name, fn in METHODS.items():
                try:
                    t0 = time.time(); S = fn(X, s); dt = time.time() - t0
                    au, ap, f1 = score(S, A)
                    rows.append(dict(d=d, seed=s, method=name, auroc=au, auprc=ap,
                                     f1=f1, time_s=round(dt, 1)))
                    print(f"d={d} s{s} {name:16s} AUROC={au:.3f} AUPRC={ap:.3f} F1={f1:.3f} t={dt:.0f}s", flush=True)
                except Exception as ex:
                    print(f"d={d} s{s} {name} FAILED: {ex}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, "instdag_raw.csv"), index=False)
    df = pd.DataFrame(rows)
    agg = df.groupby(["method"]).agg(auroc=("auroc", "mean"), auprc=("auprc", "mean"),
                                     f1=("f1", "mean"), t=("time_s", "mean"))
    byd = df.pivot_table(index="method", columns="d", values="auroc")
    order = ["CD-KAN", "DAGMA-nonlinear", "NOTEARS-MLP", "DAGMA-linear", "NOTEARS-linear"]
    agg = agg.reindex([m for m in order if m in agg.index]); byd = byd.reindex(agg.index)
    print("\n", agg.round(3).to_string()); print("\nby d:\n", byd.round(3).to_string())
    agg.to_csv(os.path.join(RES, "instdag_agg.csv"))

    # LaTeX table: AUROC by d + overall F1
    ds = sorted(df.d.unique())
    lines = ["% auto-generated", "\\begin{tabular}{l" + "c" * len(ds) + "c}", "\\toprule",
             "\\textbf{Method} & " + " & ".join(f"$d{{=}}{dd}$" for dd in ds) + " & F1 \\\\",
             "\\midrule"]
    for m in agg.index:
        cells = " & ".join(f"{byd.loc[m, dd]:.3f}" for dd in ds)
        lab = "\\textbf{CD-KAN (ours)}" if m == "CD-KAN" else m
        lines.append(f"{lab} & {cells} & {agg.loc[m,'f1']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(FIG, "tab_instdag.tex"), "w").write("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for m in agg.index:
        ax.plot(ds, [byd.loc[m, dd] for dd in ds], "-o",
                lw=2.4 if m == "CD-KAN" else 1.3, label=m)
    ax.set_xlabel("number of variables $d$"); ax.set_ylabel("AUROC"); ax.set_xticks(ds)
    ax.axhline(0.5, color="gray", ls=":"); ax.set_ylim(0.45, 1.02)
    ax.set_title("Non-linear instantaneous-DAG recovery", fontweight="bold"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "instdag.png"), dpi=200); plt.close()
    print("saved tab_instdag.tex + instdag.png")


if __name__ == "__main__":
    main()
