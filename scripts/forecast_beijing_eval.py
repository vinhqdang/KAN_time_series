"""
X5b: does SPADE-fast add value where one-step-ahead signal EXISTS? Beijing hourly weather + PM2.5 (5 series:
DEWP, TEMP, PRES, Iws, log1p PM2.5; forward-filled), PRE-REGISTERED: 2 rolling-origin folds (train 2,000 h, test 1,000 h,
stride = (T-3000)), window 16, seeds 42-43 for learned models. SPADE-fast uses the configuration chosen on the FINANCIAL
inner-validation split (residual + group-lasso 1e-3, 300 epochs); nothing is tuned on Beijing. Baselines as in the
financial benchmark (80 epochs, lr 5e-4, +RevIN). Naive last-value included. Test MSE is in raw units of each series
(averaged over the 5 series after z-scoring each series with train-fold statistics, so no series dominates).
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from honest_forecast_benchmark import build, train_torch, eval_torch
from src.evaluation import _make_windows
from fast_forecaster import FastSPADEForecaster

ER = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "beijing_pm25", "PRSA_data_2010.1.1-2014.12.31.csv"))
df["pm"] = np.log1p(df["pm2.5"].ffill().bfill())
X = df[["DEWP", "TEMP", "PRES", "Iws", "pm"]].to_numpy(np.float32); T, d = X.shape; W = 16
folds = []
for k, start in enumerate([0, T - 3000]):
    tr, te = X[start:start + 2000], X[start + 2000 - W:start + 3000]
    mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True) + 1e-8
    Xtr, ytr = _make_windows((tr - mu) / sd, W, 1); Xte, yte = _make_windows((te - mu) / sd, W, 1)
    folds.append((Xtr, ytr.squeeze(1), Xte, yte.squeeze(1)))
rows = []


def add(model, seed, mses, dt):
    rows.append(dict(model=model, seed=seed, mse=np.mean(mses), time_s=round(dt, 2)))
    print(f"{model:22s} seed{seed} MSE={np.mean(mses):.4f} t={dt:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ER, "x5b_beijing_raw.csv"), index=False)


for seed in (42, 43):
    add("Naive (last value)", seed, [float(((Xte[:, -1, :] - yte) ** 2).mean()) for _, _, Xte, yte in folds], 0)
    for name, kw in {"SPADE-fast (res+gl)": dict(residual=True, lam=1e-3), "SPADE-fast (no res)": dict()}.items():
        t0 = time.time(); ms = []
        for Xtr, ytr, Xte, yte in folds:
            f = FastSPADEForecaster(d, seed=seed, **kw).fit(Xtr, ytr); ms.append(float(((f.predict(Xte).numpy() - yte) ** 2).mean()))
        add(name, seed, ms, time.time() - t0)
    for name in ["TSMixer+RevIN", "PatchTST+RevIN", "N-BEATS+RevIN", "LSTM+RevIN"]:
        torch.manual_seed(seed); np.random.seed(seed); t0 = time.time(); ms = []
        for Xtr, ytr, Xte, yte in folds:
            m = build(name, d, W); train_torch(m, Xtr, ytr, 80, lr=5e-4); ms.append(eval_torch(m, Xte, yte)[0])
        add(name, seed, ms, time.time() - t0)
print(pd.DataFrame(rows).groupby("model").agg(mse=("mse", "mean"), t=("time_s", "mean")).sort_values("mse").round(4))
