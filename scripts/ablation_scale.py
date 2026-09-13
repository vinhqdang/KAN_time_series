"""
CD-KAN ablation (why the design works) + scalability sweep.

Ablation on non-linear synthetic (structure AUROC/F1, 3 seeds):
  * CD-KAN (component-wise, ours)  -- the information-bottleneck model
  * w/ dense backbone (coupled)    -- feeds the causal layer into a residual KAN
                                      backbone; adjacency becomes non-identifiable
  * w/o group-lasso                -- no edge-level sparsity
This isolates the two ingredients responsible for identifiable structure.

Scalability: parameter count + fit time for the component-wise model at d=10,15,20.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, pandas as pd
torch.set_num_threads(4)
from src.synthetic import generate_nonlinear_scm
from src.cdkan.causal_kan import CausalKAN
from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer, CDKANTrainerConfig
from sklearn.metrics import roc_auc_score, f1_score

RES = os.path.join(os.path.dirname(__file__), "..", "experimental_results")


def zscore(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def windows(d, w):
    X, Y = [], []
    for i in range(len(d) - w - 1):
        X.append(d[i:i + w]); Y.append(d[i + w])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


def score(pred, true):
    D = pred.shape[0]; m = ~np.eye(D, dtype=bool)
    p = np.nan_to_num(pred[m].astype(float)); t = true[m].astype(int)
    if p.max() > 0: p = p / p.max()
    au = roc_auc_score(t, p) if t.sum() and (1 - t).sum() else float("nan")
    f1 = max(f1_score(t, (p >= thr).astype(int), zero_division=0)
             for thr in np.unique(np.r_[0, p, 1]))
    return au, f1


def fit_component(X, seed, lg=0.01, epochs=150, edge_type="spline", mlp_hidden=3):
    """The real SPADE training loop (component-wise, B-spline edges). Also
    used for the matched-capacity MLP-edge ablation (Reviewer 2): passing
    edge_type='mlp' swaps only the per-edge function, keeping the optimizer,
    lr, epoch count, seed, group-lasso weight, and data identical, so the
    comparison isolates the edge parameterization."""
    torch.manual_seed(seed); np.random.seed(seed)
    D = X.shape[1]; Xw, yw = windows(X, 4)
    m = CausalKAN(D, max_lag=3, grid_size=8, edge_type=edge_type, mlp_hidden=mlp_hidden)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(Xw) - yw) ** 2).mean() + lg * m.group_lasso()
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m.importance(Xw).numpy()


def fit_dense(X, seed, epochs=60):
    """The coupled design: causal layer + dense residual KAN backbone."""
    torch.manual_seed(seed); np.random.seed(seed)
    D = X.shape[1]; Xw, yw = windows(X, 4)
    m = CDKANForecaster(D, hidden_dim=32, out_features=D, max_lag=3,
                        n_layers=2, grid_size=8, causal_bottleneck=False)
    tr = CDKANTrainer(m, device="cpu",
                      config=CDKANTrainerConfig(seed=seed, lr=5e-4, lr_adj=1e-2,
                                                lambda_sparse=0.02))
    tr.train({"X": Xw, "y": yw}, {"X": Xw, "y": yw}, epochs=epochs, patience=epochs)
    return m.get_summary_adjacency().detach().numpy()


def main():
    # Matched-capacity MLP edge (Reviewer 2): 1->3->1 tanh MLP, 10 params/edge,
    # vs. the B-spline's 11 (grid_size=8, spline_order=3) -- the closest match
    # of the two capacity points considered (a 1->5->1 MLP would be 16, farther
    # off). Everything else (optimizer, lr, epochs, seeds, lambda_g, data) is
    # identical to "CD-KAN (component-wise)", via the shared fit_component().
    variants = {
        "CD-KAN (component-wise)": lambda X, s: fit_component(X, s, lg=0.01),
        "w/ dense backbone (coupled)": lambda X, s: fit_dense(X, s),
        "w/o group-lasso": lambda X, s: fit_component(X, s, lg=0.0),
        "CD-KAN w/ MLP edges (matched capacity)":
            lambda X, s: fit_component(X, s, lg=0.01, edge_type="mlp", mlp_hidden=3),
    }
    rows = []
    for name, fn in variants.items():
        aus, f1s, times = [], [], []
        for seed in [42, 43, 44]:
            X, adj, _ = generate_nonlinear_scm(n_samples=1500, n_nodes=5,
                                               density=0.2, seed=seed)
            X = zscore(X)
            t0 = time.time()
            imp = fn(X, seed)
            times.append(time.time() - t0)
            au, f1 = score(imp, adj)
            aus.append(au); f1s.append(f1)
        rows.append(dict(config=name, auroc=round(np.mean(aus), 3),
                         f1=round(np.mean(f1s), 3),
                         fit_time_s=round(np.mean(times), 2)))
        print(f"{name:40s} AUROC={np.mean(aus):.3f} F1={np.mean(f1s):.3f} "
              f"fit_time={np.mean(times):.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "ablation_results.csv"), index=False)

    # scalability
    srows = []
    for d in [10, 15, 20]:
        Xd, _, _ = generate_nonlinear_scm(n_samples=800, n_nodes=d, density=0.15, seed=42)
        Xd = zscore(Xd); Xw, yw = windows(Xd, 4)
        torch.manual_seed(0)
        m = CausalKAN(d, max_lag=3, grid_size=8)
        npar = sum(p.numel() for p in m.parameters())
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        t0 = time.time()
        for _ in range(10):
            opt.zero_grad()
            loss = ((m(Xw) - yw) ** 2).mean() + 0.01 * m.group_lasso()
            loss.backward(); opt.step()
        per = (time.time() - t0) / 10
        srows.append(dict(d=d, params=npar, fit_time_s=round(per * 150, 1)))
        print(f"d={d} params={npar} est-150ep={per*150:.1f}s", flush=True)
    pd.DataFrame(srows).to_csv(os.path.join(RES, "scalability_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
