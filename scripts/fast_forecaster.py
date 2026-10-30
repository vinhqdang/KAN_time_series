"""
SPADE-fast forecaster (X5): identical component-wise B-spline model, but with the RevIN affine
parameters removed (affine=False), so each window's normalized inputs and spline bases are FIXED and can be
computed once; training then only touches the spline coefficients through a single einsum per epoch.
Prediction = bias + sum_{j,h} phi_{ijh}(x_norm[t-h, j]) (optionally + last normalized value), de-normalized with
the window's own mean/std (train-window statistics only; no leakage).
"""
import numpy as np, torch
from src.cdkan.causal_kan import CausalKAN


class FastSPADEForecaster:
    def __init__(self, d, max_lag=5, grid_size=8, residual=False, lam=0.0, lr=5e-3, epochs=300,
                 early_stop=False, patience=15, val_frac=0.15, seed=0):
        torch.manual_seed(seed); np.random.seed(seed)
        self.net = CausalKAN(d, max_lag=max_lag, grid_size=grid_size); self.d = d
        self.residual, self.lam, self.lr, self.epochs = residual, lam, lr, epochs
        self.early_stop, self.patience, self.val_frac = early_stop, patience, val_frac
        self.epochs_used = epochs

    @torch.no_grad()
    def _prep(self, X):
        x = torch.tensor(X); mean = x.mean(1, keepdim=True); std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        xn = (x - mean) / std
        bases = self.net._bases(self.net._lagged(xn))              # [B, d, L, K]
        return bases, xn[:, -1, :], mean.squeeze(1), std.squeeze(1)

    def _pred(self, bases, last, mean, std):
        out = torch.einsum("bjhk,ijhk->bi", bases, self.net.coef) + self.net.bias
        if self.residual: out = out + last
        return out * std + mean

    def fit(self, X, y):
        bases, last, mean, std = self._prep(X); y = torch.tensor(y)
        if self.early_stop:
            nv = int(len(y) * self.val_frac); tr = slice(0, len(y) - nv); va = slice(len(y) - nv, None)
        else: tr, va = slice(None), None
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state, bad = 1e18, None, 0
        for ep in range(1, self.epochs + 1):
            opt.zero_grad()
            loss = ((self._pred(bases[tr], last[tr], mean[tr], std[tr]) - y[tr]) ** 2).mean()
            if self.lam > 0: loss = loss + self.lam * self.net.group_lasso()
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0); opt.step()
            if self.early_stop and ep % 5 == 0:
                with torch.no_grad(): v = ((self._pred(bases[va], last[va], mean[va], std[va]) - y[va]) ** 2).mean().item()
                if v < best - 1e-9: best, bad = v, 0; best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience: self.epochs_used = ep; break
        if best_state is not None: self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, X): return self._pred(*self._prep(X))
