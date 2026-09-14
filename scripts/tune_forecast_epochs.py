"""
Held-out epoch-count sweep for SPADE's forecasting head, used by
scripts/honest_forecast_benchmark.py.

Motivation: SPADE's CausalKANForecast originally trained for only 30 epochs
against the deep baselines' 80, despite training roughly 15-20x faster per
epoch -- an unused compute budget, not a matched comparison. This script
finds a better epoch count \emph{without touching the real test folds}
reported in Table~\ref{tab:forecast}: it carves an inner train/validation
split out of the first training block only (the same 900-sample block
honest_forecast_benchmark.py uses as n_train), trains on the inner 80% and
validates on the inner 20%, with train-only z-scoring matching
src/evaluation.walk_forward_split's leakage-free protocol. The chosen epoch
count is then applied once to the real benchmark
(scripts/honest_forecast_benchmark.py --cdkan_epochs), evaluated on the
actual rolling-origin test folds and official seeds (42/43/44) for the first
time.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, pandas as pd
torch.set_num_threads(4)
from src.cdkan.causal_kan import CausalKAN
from src.cdkan.model import RevIN
from src.evaluation import _make_windows

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "financial_2020_2025.csv")


def load_financial():
    df = pd.read_csv(DATA, skiprows=[1, 2])
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("price")], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce").ffill().dropna()
    return df.values.astype(np.float32)


class CausalKANForecast(torch.nn.Module):
    def __init__(self, d, max_lag=5, grid_size=8):
        super().__init__()
        self.revin = RevIN(d)
        self.net = CausalKAN(d, max_lag=max_lag, grid_size=grid_size)

    def forward(self, x):
        xn = self.revin(x, 'norm')
        out = self.net(xn)
        return self.revin(out.unsqueeze(1), 'denorm').squeeze(1)


if __name__ == "__main__":
    data = load_financial()
    T, d = data.shape
    window = 16
    n_train = 900                      # matches honest_forecast_benchmark.py's n_train
    train_block = data[:n_train]
    inner_cut = int(n_train * 0.8)      # 720 inner-train / 180 inner-val; test folds untouched

    mu = train_block[:inner_cut].mean(0, keepdims=True)
    sd = train_block[:inner_cut].std(0, keepdims=True) + 1e-8
    scaled = (train_block - mu) / sd

    Xtr, ytr = _make_windows(scaled[:inner_cut], window, 1)
    Xval, yval = _make_windows(scaled[inner_cut - window:], window, 1)
    ytr = ytr.squeeze(1); yval = yval.squeeze(1)

    MAX_EP = 800
    results = {}
    for seed in [42, 43, 44]:
        torch.manual_seed(seed); np.random.seed(seed)
        model = CausalKANForecast(d, max_lag=5, grid_size=8)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
        Xval_t, yval_t = torch.tensor(Xval), torch.tensor(yval)
        curve = []
        for ep in range(1, MAX_EP + 1):
            opt.zero_grad()
            out = model(Xtr_t)
            loss = ((out - ytr_t) ** 2).mean()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            if ep % 20 == 0 or ep == 1:
                model.eval()
                with torch.no_grad():
                    vmse = ((model(Xval_t) - yval_t) ** 2).mean().item()
                model.train()
                curve.append((ep, loss.item(), vmse))
        for ep, tr, va in curve:
            if ep % 100 == 0 or ep in (1, MAX_EP):
                print(f"seed={seed} ep={ep:3d} train_mse={tr:.5f} val_mse={va:.5f}", flush=True)
        best = min(curve, key=lambda r: r[2])
        print(f"seed={seed} BEST val_mse={best[2]:.5f} at ep={best[0]}\n", flush=True)
        results[seed] = curve

    eps = [c[0] for c in results[42]]
    avg_val = [(ep, np.mean([results[s][i][2] for s in [42, 43, 44]]))
               for i, ep in enumerate(eps)]
    best_avg = min(avg_val, key=lambda r: r[1])
    print("Average-over-seeds best epoch:", best_avg, flush=True)
