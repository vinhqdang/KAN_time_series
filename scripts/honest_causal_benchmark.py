"""
Honest, reproducible causal-discovery benchmark for CD-KAN.

Design goals (addressing reviewer concerns):
  * REAL baselines only: PCMCI (tigramite), VarLiNGAM (lingam), VAR-Lasso,
    linear NOTEARS, GOLEM-EV (linear), GC-KAN (genuine KAN edge functions),
    and the GC-KAN+ALM ablation requested by Reviewer 3.
  * Threshold-free primary metrics (AUROC / AUPRC) so structural-ranking quality
    is decoupled from threshold selection; F1/SHD reported at each method's own
    best-F1 threshold, applied identically to every method (equal treatment).
  * Multiple seeds -> mean +/- std, plus paired Wilcoxon signed-rank tests.
  * Wall-clock timing recorded per (method, dataset, seed) on identical hardware.
  * Larger graphs (d = 5, 10) with per-series standardisation for numerical
    stability, plus a scalability sweep (params + fit time) up to d = 20.

Everything writes to experimental_results/honest_*.csv
"""
import os, sys, time, json, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

from src.synthetic import generate_nonlinear_scm
from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer, CDKANTrainerConfig
from src.cdkan.layers import BSplineFunction
from src.evaluation import compute_graph_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def zscore(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd


def _sparse_density(d):
    """Keep the graph learnably sparse as d grows (~2 expected parents/node),
    as is standard in causal-discovery benchmarks. Dense graphs at high d are
    unrecoverable by ANY method and make the comparison uninformative."""
    return min(0.3, 2.0 / (d - 1))


def make_dataset(kind, n, d, seed):
    """Return (X_standardised [T,d], true_adj [d,d] binary effect<-cause)."""
    dens = _sparse_density(d)
    if kind == "linear":
        np.random.seed(seed)
        adj = np.tril((np.random.rand(d, d) < dens).astype(int), k=-1)
        coeffs = np.random.uniform(0.4, 0.8, size=(d, d)) * adj
        X = np.zeros((n, d))
        for t in range(1, n):
            X[t] = coeffs @ X[t - 1] + np.random.normal(0, 0.1, size=d)
        true_adj = adj
    elif kind == "nonlinear":
        X, true_adj, _ = generate_nonlinear_scm(n_samples=n, n_nodes=d,
                                                density=dens, max_lag=3, seed=seed)
    else:
        raise ValueError(kind)
    return zscore(X.astype(np.float64)), true_adj.astype(int)


def make_windows(data, window, horizon=1):
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i:i + window])
        y.append(data[i + window:i + window + horizon])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


# ---------------------------------------------------------------------------
# Real baselines
# ---------------------------------------------------------------------------

def bl_var_lasso(X, max_lag=3, alpha=1e-3):
    from sklearn.linear_model import Lasso
    T, D = X.shape
    inp, tgt = [], []
    for t in range(max_lag, T):
        inp.append(X[t - max_lag:t][::-1].reshape(-1)); tgt.append(X[t])
    inp, tgt = np.asarray(inp), np.asarray(tgt)
    adj = np.zeros((D, D))
    for i in range(D):
        m = Lasso(alpha=alpha, max_iter=5000).fit(inp, tgt[:, i])
        adj[i] = np.abs(m.coef_.reshape(max_lag, D)).sum(0)
    return adj


def bl_correlation(X, max_lag=3):
    import pandas as pd
    T, D = X.shape
    adj = np.zeros((D, D))
    df = pd.DataFrame(X)
    for lag in range(1, max_lag + 1):
        c = np.abs(pd.concat([df.shift(lag), df], axis=1).corr().values[:D, D:])
        adj = np.maximum(adj, np.nan_to_num(c.T))
    np.fill_diagonal(adj, 0.0)
    return adj


def bl_notears_linear(X, max_lag=3, lambda1=0.05, max_iter=60):
    """Linear NOTEARS on lagged design: x_t = sum_l X_{t-l} W_l, edge = max_l|W_l|.
    Augmented-Lagrangian on the contemporaneous-summary acyclicity."""
    T, D = X.shape
    inp, tgt = [], []
    for t in range(max_lag, T):
        inp.append(X[t - max_lag:t][::-1].reshape(-1)); tgt.append(X[t])
    Xin = torch.tensor(np.asarray(inp), dtype=torch.float32)
    Yt = torch.tensor(np.asarray(tgt), dtype=torch.float32)
    W = torch.zeros(max_lag * D, D, requires_grad=True)
    opt = torch.optim.Adam([W], lr=1e-2)
    rho, alpha = 1.0, 0.0
    for it in range(max_iter):
        opt.zero_grad()
        pred = Xin @ W
        mse = ((pred - Yt) ** 2).mean()
        Wsum = W.view(max_lag, D, D).abs().sum(0)          # [D,D] cause->effect
        h = torch.trace(torch.matrix_exp(Wsum * Wsum)) - D
        loss = mse + lambda1 * W.abs().sum() + alpha * h + 0.5 * rho * h * h
        loss.backward(); opt.step()
        if (it + 1) % 15 == 0:
            with torch.no_grad():
                hv = (torch.trace(torch.matrix_exp(Wsum * Wsum)) - D).item()
            alpha += rho * hv; rho = min(rho * 2, 1e10)
    adj = W.detach().view(max_lag, D, D).abs().max(0).values.numpy()
    return adj.T   # to effect<-cause convention


def bl_golem_linear(X, max_lag=3, l1=0.02, epochs=150):
    """GOLEM-EV style linear AR structure learner with soft DAG + L1."""
    T, D = X.shape
    inp, tgt = [], []
    for t in range(max_lag, T):
        inp.append(X[t - max_lag:t][::-1].reshape(-1)); tgt.append(X[t])
    Xin = torch.tensor(np.asarray(inp), dtype=torch.float32)
    Yt = torch.tensor(np.asarray(tgt), dtype=torch.float32)
    W = torch.zeros(max_lag * D, D, requires_grad=True)
    opt = torch.optim.Adam([W], lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        pred = Xin @ W
        resid = pred - Yt
        # GOLEM equal-variance likelihood term
        like = 0.5 * D * torch.log(((resid ** 2).sum()) + 1e-8)
        Wsum = W.view(max_lag, D, D).abs().sum(0)
        h = torch.trace(torch.matrix_exp(Wsum * Wsum)) - D
        loss = like + l1 * W.abs().sum() + 5.0 * h
        loss.backward(); opt.step()
    return W.detach().view(max_lag, D, D).abs().max(0).values.numpy().T


def bl_pcmci(X, max_lag=3):
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    import tigramite.data_processing as pp
    T, D = X.shape
    df = pp.DataFrame(X)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=0.05)
    val = np.abs(res["val_matrix"])              # [D(cause), D(effect), lag+1]
    adj = val[:, :, 1:].max(axis=2)              # drop contemporaneous lag 0
    return adj.T                                 # effect<-cause


def bl_varlingam(X, max_lag=3):
    import lingam
    m = lingam.VARLiNGAM(lags=max_lag)
    m.fit(X)
    mats = np.abs(np.asarray(m.adjacency_matrices_))   # [lag+1, D, D] effect<-cause
    # exclude contemporaneous (index 0) to match lagged ground truth
    return mats[1:].max(0) if mats.shape[0] > 1 else mats.max(0)


def bl_neural_gc(X, max_lag=3, hidden=10, lam=0.02, lr=1e-2, epochs=1500, seed=0):
    """Neural Granger Causality (cMLP), Tank et al., IEEE TPAMI 2021.
    Uses the authors' cMLP architecture and GC() read-out (vendored under
    external/neural_gc); trained with Adam + group-lasso on the first-layer
    input weights, the method's prescribed group-sparse objective. Returns a
    [effect, cause] importance matrix."""
    _EXT = os.path.join(os.path.dirname(__file__), "..", "external", "neural_gc")
    if _EXT not in sys.path:
        sys.path.insert(0, _EXT)
    from models.cmlp import cMLP
    torch.manual_seed(seed)
    D = X.shape[1]
    Xt = torch.tensor(X[np.newaxis], dtype=torch.float32)          # (1,T,D)
    cmlp = cMLP(D, lag=max_lag, hidden=[hidden])
    opt = torch.optim.Adam(cmlp.parameters(), lr=lr); lf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = cmlp(Xt[:, :-1])
        loss = sum(lf(pred[:, :, i], Xt[:, max_lag:, i]) for i in range(D))
        gl = sum(torch.norm(net.layers[0].weight, dim=(0, 2)).sum() for net in cmlp.networks)
        (loss + lam * gl).backward(); opt.step()
    return cmlp.GC(threshold=False).detach().cpu().numpy()          # [effect, cause]


def bl_nts_notears(X, max_lag=3, hidden=10, lambda1=0.02, lambda2=0.01, seed=0):
    """NTS-NOTEARS, Sun et al., AISTATS 2023 (vendored, unmodified, under
    external/nts_notears): 1D-CNN structural functions for nonlinear lagged and
    instantaneous edges with a NOTEARS acyclicity constraint. Returns a
    [effect, cause] summary importance matrix (max over lagged + instantaneous)."""
    _EXT = os.path.join(os.path.dirname(__file__), "..", "external", "nts_notears")
    for p in (os.path.join(_EXT, "notears"), _EXT):
        if p not in sys.path:
            sys.path.insert(0, p)
    from notears.nts_notears import NTS_NOTEARS, train_NTS_NOTEARS
    prev_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.double)
    try:
        D = X.shape[1]
        m = NTS_NOTEARS(dims=[D, hidden, 1], bias=True, number_of_lags=max_lag,
                        variable_names_no_time=[str(i) for i in range(D)], prior_knowledge=[])
        train_NTS_NOTEARS(m, X.astype(np.float64), device="cpu", lambda1=lambda1,
                          lambda2=lambda2, w_threshold=0.0, h_tol=1e-8, verbose=0, rho_max=1e16)
        Wsim, Wnar = m.fc1_to_adj()                                 # row->col = [cause, effect]
        Wsim = np.abs(np.asarray(Wsim)); Wnar = np.abs(np.asarray(Wnar)).reshape(max_lag, D, D)
        summ = np.maximum(Wsim, Wnar.max(0))                        # [cause, effect]
        return summ.T                                              # -> [effect, cause]
    finally:
        torch.set_default_dtype(prev_dtype)


class _EdgeKAN(nn.Module):
    """Per-target KAN Granger predictor: genuine B-spline edge functions."""
    def __init__(self, d_in, grid=8):
        super().__init__()
        self.funcs = nn.ModuleList([BSplineFunction(grid_size=grid) for _ in range(d_in)])

    def forward(self, x):                        # x:[B,d_in]
        out = 0.0
        for j, f in enumerate(self.funcs):
            out = out + f(x[:, j])
        return out

    def edge_norms(self):
        return torch.stack([f.coef.abs().mean() for f in self.funcs])


def bl_gc_kan(X, max_lag=3, epochs=120, lr=5e-3, l1=1e-2, use_alm=False):
    """Genuine KAN-based Granger causality (edge functions are B-splines).
    If use_alm: add a NOTEARS acyclicity penalty on the contemporaneous-summary
    adjacency via a light Augmented-Lagrangian loop (the GC-KAN+ALM ablation)."""
    T, D = X.shape
    inp, tgt = [], []
    for t in range(max_lag, T):
        inp.append(X[t - max_lag:t][::-1].reshape(-1)); tgt.append(X[t])
    Xin = torch.tensor(np.asarray(inp), dtype=torch.float32)
    Yt = torch.tensor(np.asarray(tgt), dtype=torch.float32)
    adj = np.zeros((D, D))
    # a shared learnable adjacency logit for ALM ablation
    S = torch.zeros(D, D, requires_grad=True) if use_alm else None
    opt_S = torch.optim.Adam([S], lr=1e-2) if use_alm else None
    rho, alpha = 1.0, 0.0
    models = [_EdgeKAN(max_lag * D) for _ in range(D)]
    params = [p for m in models for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        if use_alm:
            opt_S.zero_grad()
        loss = 0.0
        for i in range(D):
            pred = models[i](Xin)
            loss = loss + ((pred - Yt[:, i]) ** 2).mean()
            loss = loss + l1 * models[i].edge_norms().sum()
        if use_alm:
            Wsum = torch.sigmoid(S)
            h = torch.trace(torch.matrix_exp(Wsum * Wsum)) - D
            loss = loss + alpha * h + 0.5 * rho * h * h + 1e-3 * Wsum.sum()
        loss.backward(); opt.step()
        if use_alm:
            opt_S.step()
            if (ep + 1) % 20 == 0:
                with torch.no_grad():
                    Wsum = torch.sigmoid(S)
                    hv = (torch.trace(torch.matrix_exp(Wsum * Wsum)) - D).item()
                alpha += rho * hv; rho = min(rho * 2, 1e8)
    for i in range(D):
        with torch.no_grad():
            en = models[i].edge_norms().numpy().reshape(max_lag, D).sum(0)   # [D]
        adj[i] = en
    if use_alm:
        adj = adj * torch.sigmoid(S).detach().numpy()
    return adj


def fit_cdkan(X, seed, max_lag=3, epochs=150, lr=5e-3, lg=0.01, grid_size=8):
    """Train the improved (component-wise) CausalKAN; return (importance, time, n_params).

    This is the corrected CD-KAN: each target is predicted only through per-(cause,
    lag) B-spline edges (information bottleneck), whole edges are driven to zero by
    a group-lasso, and structure is read from each edge's contribution to the
    forecast. See src/cdkan/causal_kan.py for the rationale."""
    from src.cdkan.causal_kan import CausalKAN
    torch.manual_seed(seed); np.random.seed(seed)
    D = X.shape[1]
    Xw, yw = make_windows(X, window=max_lag + 1)
    Xw = torch.tensor(Xw, dtype=torch.float32)
    yw = torch.tensor(yw[:, 0, :], dtype=torch.float32)
    model = CausalKAN(D, max_lag=max_lag, grid_size=grid_size)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xw)
        loss = ((pred - yw) ** 2).mean() + lg * model.group_lasso()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    dt = time.time() - t0
    imp = model.importance(Xw).cpu().numpy()
    return imp, dt, n_params


# ---------------------------------------------------------------------------
# Metric wrapper: threshold-free AUROC/AUPRC + best-F1 threshold (equal to all)
# ---------------------------------------------------------------------------

def score_adj(pred, true):
    d = pred.shape[0]
    mask = ~np.eye(d, dtype=bool)
    p = np.nan_to_num(pred[mask].astype(float)); t = true[mask].astype(int)
    if p.max() > 0:
        p = p / p.max()
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    auroc = roc_auc_score(t, p) if t.sum() and (1 - t).sum() else float("nan")
    auprc = average_precision_score(t, p) if t.sum() else float("nan")
    # best-F1 over thresholds (identical procedure for every method)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.unique(np.concatenate([[0.0], p, [1.0]])):
        f = f1_score(t, (p >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr
    shd = int(np.abs((p >= best_thr).astype(int) - t).sum())
    return dict(auroc=auroc, auprc=auprc, f1=best_f1, shd=shd)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    configs = [
        ("linear",    "linear_d5_n1500",   1500, 5),
        ("nonlinear", "nonlin_d5_n1500",   1500, 5),
        ("linear",    "linear_d10_n2000",  2000, 10),
        ("nonlinear", "nonlin_d10_n2000",  2000, 10),
        ("linear",    "linear_d20_n2000",  2000, 20),
        ("nonlinear", "nonlin_d20_n2000",  2000, 20),
        ("linear",    "linear_d50_n3000",  3000, 50),
        ("nonlinear", "nonlin_d50_n3000",  3000, 50),
    ]
    if args.quick:
        configs = configs[:1]
        args.seeds = args.seeds[:1]

    # some classical baselines are O(d^2)-O(d^3) and slow at high d
    SLOW_MAXD = {"PCMCI": 20, "VarLiNGAM": 20, "NOTEARS": 50, "GOLEM": 50,
                 "GC-KAN": 10, "GC-KAN+ALM": 10, "Neural-GC": 20, "NTS-NOTEARS": 10}

    raw_path = os.path.join(RESULTS_DIR, "honest_causal_raw.csv")
    import pandas as pd
    _cols = ["dataset", "kind", "d", "n", "seed", "method",
             "auroc", "auprc", "f1", "shd", "time_s", "n_params"]

    def _flush(rows):
        pd.DataFrame(rows)[_cols].to_csv(raw_path, index=False)

    methods = {
        "VAR-Lasso":     lambda X: (bl_var_lasso(X), None, None),
        "Correlation":   lambda X: (bl_correlation(X), None, None),
        "NOTEARS":       lambda X: (bl_notears_linear(X), None, None),
        "GOLEM":         lambda X: (bl_golem_linear(X), None, None),
        "PCMCI":         lambda X: (bl_pcmci(X), None, None),
        "VarLiNGAM":     lambda X: (bl_varlingam(X), None, None),
        "GC-KAN":        lambda X: (bl_gc_kan(X, use_alm=False), None, None),
        "GC-KAN+ALM":    lambda X: (bl_gc_kan(X, use_alm=True), None, None),
        "Neural-GC":     lambda X: (bl_neural_gc(X), None, None),
        "NTS-NOTEARS":   lambda X: (bl_nts_notears(X), None, None),
    }

    rows = []
    for kind, name, n, d in configs:
        for seed in args.seeds:
            X, true_adj = make_dataset(kind, n, d, seed)
            # baselines
            for mname, fn in methods.items():
                if d > SLOW_MAXD.get(mname, 999):
                    continue   # skip baselines that don't scale to this width
                try:
                    t0 = time.time()
                    adj, _, _ = fn(X)
                    dt = time.time() - t0
                    m = score_adj(adj, true_adj)
                    m.update(dataset=name, kind=kind, d=d, n=n, seed=seed,
                             method=mname, time_s=round(dt, 3), n_params="")
                    rows.append(m)
                    print(f"[{name} s{seed}] {mname:12s} "
                          f"AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} "
                          f"F1={m['f1']:.3f} SHD={m['shd']} t={dt:.1f}s", flush=True)
                except Exception as e:
                    print(f"[{name} s{seed}] {mname} FAILED: {e}", flush=True)
            # CD-KAN (improved component-wise CausalKAN)
            try:
                imp, dt, npar = fit_cdkan(X, seed)
                m = score_adj(imp, true_adj)
                m.update(dataset=name, kind=kind, d=d, n=n, seed=seed,
                         method="CD-KAN", time_s=round(dt, 3), n_params=npar)
                rows.append(m)
                print(f"[{name} s{seed}] {'CD-KAN':12s} "
                      f"AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} "
                      f"F1={m['f1']:.3f} SHD={m['shd']} t={dt:.1f}s", flush=True)
            except Exception as e:
                print(f"[{name} s{seed}] CD-KAN FAILED: {e}", flush=True)
            _flush(rows)   # incremental checkpoint after each dataset*seed

    df = pd.DataFrame(rows)
    out = raw_path
    df.to_csv(out, index=False)
    print("\nSaved", out)

    # aggregate mean +/- std across dataset*seed
    agg = df.groupby("method").agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
        shd_mean=("shd", "mean"), shd_std=("shd", "std"),
        time_mean=("time_s", "mean"),
    ).sort_values("f1_mean", ascending=False)
    agg.to_csv(os.path.join(RESULTS_DIR, "honest_causal_agg.csv"))
    print(agg.round(3).to_string())

    # paired Wilcoxon: CD-KAN vs each baseline on F1 (matched dataset,seed)
    from scipy.stats import wilcoxon
    piv = df.pivot_table(index=["dataset", "seed"], columns="method", values="f1")
    stat_rows = []
    if "CD-KAN" in piv.columns:
        for mth in piv.columns:
            if mth == "CD-KAN":
                continue
            a, b = piv["CD-KAN"], piv[mth]
            ok = a.notna() & b.notna()
            if ok.sum() >= 5 and (a[ok] - b[ok]).abs().sum() > 0:
                try:
                    w, pval = wilcoxon(a[ok], b[ok])
                except Exception:
                    w, pval = float("nan"), float("nan")
            else:
                w, pval = float("nan"), float("nan")
            stat_rows.append(dict(baseline=mth, n_pairs=int(ok.sum()),
                                  cdkan_mean=round(a[ok].mean(), 3),
                                  base_mean=round(b[ok].mean(), 3),
                                  wilcoxon_W=w, p_value=pval))
    pd.DataFrame(stat_rows).to_csv(
        os.path.join(RESULTS_DIR, "honest_causal_stats.csv"), index=False)
    print("\nWilcoxon (CD-KAN vs baselines, F1):")
    print(pd.DataFrame(stat_rows).to_string(index=False))


if __name__ == "__main__":
    main()
