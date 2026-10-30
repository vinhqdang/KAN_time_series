"""
X5: make SPADE's forecaster faster and more accurate, tuned ONLY on an inner train/validation split of the
first 900-sample training block (never the rolling-origin test folds), as in tune_forecast_epochs.py.

Variants (all RevIN + component-wise CausalKAN, window 16, max_lag 5, grid 8):
  A published     : prediction = spline sum;                 300 epochs, lr 5e-3, full batch
  B residual      : prediction = last value + spline sum;   300 epochs, lr 5e-3
  C residual+fast : residual, lr 1e-2, cosine decay,         150 epochs
  D residual+ES   : residual, lr 1e-2, cosine, early stopping on the inner-val tail of the training windows
                    (patience 15 evaluations of every 5 epochs), max 300 epochs
Selection rule (fixed before running): lowest mean inner-val MSE over seeds 42-44 among variants whose
training time is below variant A's; ties within 2% go to the faster variant.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, pandas as pd
from src.cdkan.causal_kan import CausalKAN
from src.cdkan.model import RevIN
from src.evaluation import _make_windows
sys.path.append(os.path.dirname(__file__))
from tune_forecast_epochs import load_financial


class Forecaster(torch.nn.Module):
    def __init__(self, d, residual=False, max_lag=5, grid_size=8):
        super().__init__(); self.revin = RevIN(d); self.net = CausalKAN(d, max_lag=max_lag, grid_size=grid_size); self.residual = residual
    def forward(self, x):
        xn = self.revin(x, "norm"); out = self.net(xn)
        if self.residual: out = out + xn[:, -1, :]
        return self.revin(out.unsqueeze(1), "denorm").squeeze(1)


def train(model, X, y, epochs, lr, cosine=False, early_stop=False, val_frac=0.15, patience=15, seed=0):
    X, y = torch.tensor(X), torch.tensor(y); Xv = yv = None
    if early_stop:
        nv = int(len(X) * val_frac); Xv, yv = X[-nv:], y[-nv:]; X, y = X[:-nv], y[:-nv]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if cosine else None
    best, best_state, bad, ep_used = 1e18, None, 0, epochs
    for ep in range(1, epochs + 1):
        opt.zero_grad(); loss = ((model(X) - y) ** 2).mean()
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if sched: sched.step()
        if early_stop and ep % 5 == 0:
            with torch.no_grad(): v = ((model(Xv) - yv) ** 2).mean().item()
            if v < best - 1e-6: best, bad = v, 0; best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: ep_used = ep; break
    if best_state is not None: model.load_state_dict(best_state)
    return model, ep_used


VARIANTS = {
    "A_published": dict(residual=False, epochs=300, lr=5e-3),
    "B_residual": dict(residual=True, epochs=300, lr=5e-3),
    "C_residual_fast": dict(residual=True, epochs=150, lr=1e-2, cosine=True),
    "D_residual_ES": dict(residual=True, epochs=300, lr=1e-2, cosine=True, early_stop=True),
}

FAST = {
    "E_cached": dict(),                                              # same as A, bases cached, no RevIN affine
    "F_cached_res": dict(residual=True),
    "G_cached_ES": dict(early_stop=True),
    "H_cached_res_ES": dict(residual=True, early_stop=True),
    "I_cached_res_gl": dict(residual=True, lam=1e-3),
    "J_cached_gl": dict(lam=1e-3),
}

if __name__ == "__main__":
    data = load_financial(); T, d = data.shape; window = 16; n_train = 900; cut = int(n_train * 0.8)
    blk = data[:n_train]; mu, sd = blk[:cut].mean(0, keepdims=True), blk[:cut].std(0, keepdims=True) + 1e-8
    sc = (blk - mu) / sd
    Xtr, ytr = _make_windows(sc[:cut], window, 1); Xv, yv = _make_windows(sc[cut - window:], window, 1)
    ytr, yv = ytr.squeeze(1), yv.squeeze(1); rows = []
    # naive reference: predict the last observed value
    naive = float(((Xv[:, -1, :] - yv) ** 2).mean()); print(f"naive last-value inner-val MSE (scaled units) = {naive:.5f}")
    for name, cfg in ({} if os.environ.get('ONLY_FAST') else VARIANTS).items():
        for seed in (42, 43, 44):
            torch.manual_seed(seed); np.random.seed(seed)
            m = Forecaster(d, cfg["residual"]); t0 = time.time()
            m, ep = train(m, Xtr, ytr, cfg["epochs"], cfg["lr"], cfg.get("cosine", False), cfg.get("early_stop", False))
            dt = time.time() - t0
            with torch.no_grad(): v = ((m(torch.tensor(Xv)) - torch.tensor(yv)) ** 2).mean().item()
            rows.append(dict(variant=name, seed=seed, val_mse=v, time_s=round(dt, 1), epochs_used=ep))
            print(f"{name:16s} seed{seed} val_mse={v:.5f} t={dt:.1f}s ep={ep}", flush=True)
    from fast_forecaster import FastSPADEForecaster
    for name, kw in FAST.items():
        for seed in (42, 43, 44):
            t0 = time.time(); f = FastSPADEForecaster(d, seed=seed, **kw).fit(Xtr, ytr); dt = time.time() - t0
            v = ((f.predict(Xv) - torch.tensor(yv)) ** 2).mean().item()
            rows.append(dict(variant=name, seed=seed, val_mse=v, time_s=round(dt, 1), epochs_used=f.epochs_used))
            print(f"{name:16s} seed{seed} val_mse={v:.5f} t={dt:.1f}s ep={f.epochs_used}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(__file__), "..", "experimental_results", "x5_tuning_raw.csv"), index=False)
    print(df.groupby("variant")[["val_mse", "time_s", "epochs_used"]].mean().round(5))
