"""
Two validation experiments that go beyond stability (Reviewers 3.4, 6.2.3):

  (A) Time-reversal placebo. A genuine temporal causal method should recover the
      true (forward) graph far better from forward data than from time-reversed
      data. If performance is invariant to reversal, the "edges" are spurious
      correlations. We report forward vs reversed AUROC on synthetic data.

  (B) Real-data regime/event analysis. We learn the financial-market graph
      separately on a pre-event and post-event window (split at the 2022 rate-hike
      onset) and report which causal edges appear/disappear---an interpretable,
      event-anchored check that the structure tracks known macro regime shifts,
      rather than only measuring self-consistency.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, pandas as pd
torch.set_num_threads(4)
from src.synthetic import generate_nonlinear_scm
from src.cdkan.causal_kan import CausalKAN
from sklearn.metrics import roc_auc_score

RES = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "financial_2020_2025.csv")


def zscore(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def windows(d, w):
    X, Y = [], []
    for i in range(len(d) - w - 1):
        X.append(d[i:i + w]); Y.append(d[i + w])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


def fit_importance(X, seed=42, max_lag=3, epochs=150, lg=0.01):
    torch.manual_seed(seed); np.random.seed(seed)
    d = X.shape[1]
    Xw, yw = windows(X, max_lag + 1)
    m = CausalKAN(d, max_lag=max_lag, grid_size=8)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(Xw) - yw) ** 2).mean() + lg * m.group_lasso()
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m.importance(Xw).numpy(), m


def auroc(imp, true):
    d = imp.shape[0]; mask = ~np.eye(d, dtype=bool)
    p = np.nan_to_num(imp[mask].astype(float)); t = true[mask].astype(int)
    if p.max() > 0: p = p / p.max()
    return roc_auc_score(t, p) if t.sum() and (1 - t).sum() else float("nan")


def test_time_reversal():
    print("== (A) time-reversal placebo (nonlinear d=10, 3 seeds) ==")
    fwd, rev = [], []
    for seed in [42, 43, 44]:
        X, adj, _ = generate_nonlinear_scm(n_samples=2000, n_nodes=10, density=0.15, seed=seed)
        X = zscore(X)
        imp_f, _ = fit_importance(X, seed)
        imp_r, _ = fit_importance(X[::-1].copy(), seed)
        af, ar = auroc(imp_f, adj), auroc(imp_r, adj)
        fwd.append(af); rev.append(ar)
        print(f"  seed{seed}: forward AUROC={af:.3f}  reversed AUROC={ar:.3f}", flush=True)
    print(f"  MEAN forward={np.mean(fwd):.3f}  reversed={np.mean(rev):.3f}  "
          f"drop={np.mean(fwd)-np.mean(rev):+.3f}")
    pd.DataFrame([dict(forward_auroc=round(np.mean(fwd), 3),
                       reversed_auroc=round(np.mean(rev), 3),
                       drop=round(np.mean(fwd) - np.mean(rev), 3))]
                 ).to_csv(os.path.join(RES, "timereversal.csv"), index=False)


def load_financial():
    df = pd.read_csv(DATA, index_col=0, skiprows=[1, 2])
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").ffill().dropna()
    return df


def test_event_regime():
    print("\n== (B) real-data pre/post 2022 rate-hike regime analysis ==")
    df = load_financial()
    names = list(df.columns)
    pre = df[df.index < "2022-01-01"].values.astype(np.float32)
    post = df[df.index >= "2022-01-01"].values.astype(np.float32)
    print(f"  pre-2022: {len(pre)} days,  post-2022: {len(post)} days")
    imp_pre, _ = fit_importance(zscore(pre), seed=42, max_lag=5)
    imp_post, _ = fit_importance(zscore(post), seed=42, max_lag=5)
    d = len(names)

    def topedges(imp, k=8):
        mask = ~np.eye(d, dtype=bool)
        vals = [(imp[i, j], names[j], names[i]) for i in range(d) for j in range(d) if mask[i, j]]
        vals.sort(reverse=True)
        return vals[:k]

    def biner(imp, thr_q=0.75):
        mask = ~np.eye(d, dtype=bool)
        thr = np.quantile(imp[mask], thr_q)
        return (imp >= thr) & mask

    bpre, bpost = biner(imp_pre), biner(imp_post)
    appeared = [(names[j], names[i]) for i in range(d) for j in range(d)
                if bpost[i, j] and not bpre[i, j]]
    disappeared = [(names[j], names[i]) for i in range(d) for j in range(d)
                   if bpre[i, j] and not bpost[i, j]]
    inter = (bpre & bpost).sum(); union = (bpre | bpost).sum()
    jac = inter / union if union else float("nan")
    print(f"  pre->post Jaccard of top-quartile edges: {jac:.3f}")
    print(f"  edges that APPEARED post-2022: {appeared[:8]}")
    print(f"  edges that DISAPPEARED post-2022: {disappeared[:8]}")
    pd.DataFrame([dict(pre_post_jaccard=round(float(jac), 3),
                       n_appeared=len(appeared), n_disappeared=len(disappeared),
                       appeared=str(appeared[:8]), disappeared=str(disappeared[:8]))]
                 ).to_csv(os.path.join(RES, "event_regime.csv"), index=False)


if __name__ == "__main__":
    test_time_reversal()
    test_event_regime()
