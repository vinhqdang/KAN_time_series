"""
X5 final evaluation on the real rolling-origin test folds (same folds, seeds 42-44, window 16 as
honest_forecast_benchmark.py), with the SPADE-fast variants chosen on the INNER validation split
(scripts/tune_forecast_v2.py: selected = residual + group-lasso 1e-3, 300 epochs, cached bases).
All models are re-run here in the current environment so that timings and MSEs are comparable.
Adds the naive last-value predictor, which the original comparison lacked.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from honest_forecast_benchmark import load_financial, build, train_torch, eval_torch
from src.evaluation import walk_forward_split
from fast_forecaster import FastSPADEForecaster

ER = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
SEEDS = [42, 43, 44]; WINDOW = 16
FAST = {"SPADE-fast (res+gl)": dict(residual=True, lam=1e-3),
        "SPADE-fast (res+ES)": dict(residual=True, early_stop=True),
        "SPADE-fast (no res)": dict()}
DEEP = ["CD-KAN", "TSMixer+RevIN", "PatchTST+RevIN", "N-BEATS+RevIN", "LSTM+RevIN"]

data = load_financial(); T, d = data.shape
stride = max(1, (T - 900 - 300) // 1)
folds = walk_forward_split(data, n_train=900, n_test=300, stride=stride, window=WINDOW, horizon=1)[:2]
rows = []


def add(model, seed, mses, maes, dt):
    rows.append(dict(model=model, seed=seed, mse=np.mean(mses), mae=np.mean(maes), time_s=round(dt, 2)))
    print(f"{model:22s} seed{seed} MSE={np.mean(mses):.4f} MAE={np.mean(maes):.4f} t={dt:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ER, "x5_final_raw.csv"), index=False)


for seed in SEEDS:
    mses, maes = [], []
    for (Xtr, ytr, Xte, yte) in folds:
        pred = Xte[:, -1, :]; y = yte.squeeze(1)
        mses.append(float(((pred - y) ** 2).mean())); maes.append(float(np.abs(pred - y).mean()))
    add("Naive (last value)", seed, mses, maes, 0.0)
    for name, kw in FAST.items():
        t0 = time.time(); mses, maes = [], []
        for (Xtr, ytr, Xte, yte) in folds:
            f = FastSPADEForecaster(d, seed=seed, **kw).fit(Xtr, ytr.squeeze(1)); p = f.predict(Xte).numpy(); y = yte.squeeze(1)
            mses.append(float(((p - y) ** 2).mean())); maes.append(float(np.abs(p - y).mean()))
        add(name, seed, mses, maes, time.time() - t0)
    for name in DEEP:
        torch.manual_seed(seed); np.random.seed(seed); t0 = time.time(); mses, maes = [], []
        for (Xtr, ytr, Xte, yte) in folds:
            m = build(name, d, WINDOW); ep = 300 if name == "CD-KAN" else 80; lr = 5e-3 if name == "CD-KAN" else 5e-4
            train_torch(m, Xtr, ytr.squeeze(1), ep, lr=lr); mse, mae = eval_torch(m, Xte, yte.squeeze(1)); mses.append(mse); maes.append(mae)
        add(name, seed, mses, maes, time.time() - t0)
df = pd.DataFrame(rows)
print(df.groupby("model").agg(mse=("mse", "mean"), sd=("mse", "std"), mae=("mae", "mean"), time=("time_s", "mean")).sort_values("mse").round(4))
