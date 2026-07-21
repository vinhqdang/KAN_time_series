"""
Instantaneous (contemporaneous) DAG recovery experiment.

Validates that CD-KAN's acyclicity constraint is genuinely exercised: on SVAR
data with a KNOWN instantaneous DAG + lagged edges, we recover both blocks and
verify the contemporaneous residual h(W0)->0. Compares the contemporaneous block
against a linear instantaneous-NOTEARS baseline.
Writes experimental_results/contemp_results.csv and a LaTeX table.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, pandas as pd
torch.set_num_threads(4)
from src.synthetic import generate_svar_contemp
from src.cdkan.causal_kan import CausalKANContemp
from src.cdkan.losses import _h_notears
from sklearn.metrics import roc_auc_score, f1_score

RES = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
FIG = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")


def zscore(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def score(imp, true):
    d = imp.shape[0]; m = ~np.eye(d, dtype=bool)
    p = np.nan_to_num(np.asarray(imp)[m].astype(float)); t = true[m].astype(int)
    if p.max() > 0: p = p / p.max()
    au = roc_auc_score(t, p) if t.sum() and (1 - t).sum() else float("nan")
    f1 = max(f1_score(t, (p >= q).astype(int), zero_division=0) for q in np.unique(np.r_[0, p, 1]))
    return au, f1


def linear_notears_contemp(X, max_iter=120, l1=0.02):
    """Linear instantaneous NOTEARS: X ~ X W (no self), DAG via ALM."""
    T, d = X.shape
    Xt = torch.tensor(X, dtype=torch.float32)
    W = torch.zeros(d, d, requires_grad=True)
    opt = torch.optim.Adam([W], lr=1e-2)
    rho, alpha = 1.0, 0.0
    mask = 1 - torch.eye(d)
    for it in range(max_iter):
        opt.zero_grad()
        Wm = W * mask
        pred = Xt @ Wm.t()                    # x_i ~ sum_j W_ij x_j
        mse = ((pred - Xt) ** 2).mean()
        h = _h_notears(Wm)
        loss = mse + l1 * Wm.abs().sum() + alpha * h + 0.5 * rho * h * h
        loss.backward(); opt.step()
        if (it + 1) % 20 == 0:
            with torch.no_grad(): hv = _h_notears(W * mask).item()
            if hv > 1e-8: alpha += rho * hv; rho = min(rho * 2, 1e10)
    return (W * mask).abs().detach().numpy()


def fit_cdkan_contemp(X, seed, max_lag=2, epochs=300, lg=0.01):
    torch.manual_seed(seed); np.random.seed(seed)
    d = X.shape[1]; L = max_lag
    Xc = torch.tensor(X[L:], dtype=torch.float32)
    Xl = torch.tensor(np.stack([X[L - 1 - h:len(X) - 1 - h] for h in range(L)], axis=1),
                      dtype=torch.float32)
    m = CausalKANContemp(d, max_lag=L, grid_size=8)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    rho, alpha = 1.0, 0.0
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        pred = m(Xc, Xl); mse = ((pred - Xc) ** 2).mean()
        h = m.h0()
        (mse + lg * m.group_lasso() + alpha * h + 0.5 * rho * h * h).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad(): hv = m.h0().item()
            if hv > 1e-8: alpha += rho * hv; rho = min(rho * 2, 1e10)
    dt = time.time() - t0
    return (m.contemp_importance(Xc).numpy(), m.lagged_importance(Xl).numpy(),
            float(m.h0().item()), dt)


def main():
    rows = []
    for d in [6, 10]:
        for seed in range(5):
            X, B0, BL = generate_svar_contemp(n_samples=2000, n_nodes=d,
                                              contemp_density=min(0.35, 2.5 / (d - 1)),
                                              lag_density=min(0.2, 1.5 / (d - 1)),
                                              max_lag=2, seed=seed)
            if not np.isfinite(X).all():
                continue
            Xz = zscore(X)
            i0, iL, hval, dt = fit_cdkan_contemp(Xz, seed)
            au0, f0 = score(i0, B0); auL, fL = score(iL, BL)
            # baseline: linear NOTEARS on contemporaneous
            bl = linear_notears_contemp(Xz)
            bau0, bf0 = score(bl, B0)
            rows.append(dict(d=d, seed=seed, ck_contemp_auroc=au0, ck_contemp_f1=f0,
                             ck_lag_auroc=auL, h_W0=hval,
                             notears_contemp_auroc=bau0, time_s=round(dt, 1)))
            print(f"d={d} s{seed}: CD-KAN contemp AUROC={au0:.3f} F1={f0:.3f} "
                  f"lag AUROC={auL:.3f} h(W0)={hval:.1e} | NOTEARS contemp={bau0:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "contemp_results.csv"), index=False)
    agg = df.groupby("d").agg(
        ck_contemp=("ck_contemp_auroc", "mean"), ck_contemp_f1=("ck_contemp_f1", "mean"),
        ck_lag=("ck_lag_auroc", "mean"), h=("h_W0", "mean"),
        notears=("notears_contemp_auroc", "mean"), t=("time_s", "mean")).reset_index()
    print("\n", agg.round(3).to_string())
    # LaTeX table
    lines = ["% auto-generated", "\\begin{tabular}{c ccc c c}", "\\toprule",
             "$d$ & \\multicolumn{3}{c}{\\textbf{CD-KAN}} & NOTEARS & \\\\",
             "\\cmidrule(lr){2-4}",
             " & contemp.\\ AUROC & lagged AUROC & $h(W_0)$ & contemp.\\ AUROC & time (s) \\\\",
             "\\midrule"]
    for _, r in agg.iterrows():
        lines.append(f"{int(r['d'])} & {r['ck_contemp']:.3f} & {r['ck_lag']:.3f} & "
                     f"{r['h']:.1e} & {r['notears']:.3f} & {r['t']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(FIG, "tab_contemp.tex"), "w") as f:
        f.write("\n".join(lines))
    print("saved tab_contemp.tex")


if __name__ == "__main__":
    main()
