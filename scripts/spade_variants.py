"""
X1 (post-acceptance extension): candidate fixes for SPADE's sparsity/shrinkage weaknesses.

Candidates, all built on CausalKANInstant (same model, same ALM acyclicity):
  base      : lambda_g=0.01, 400 epochs (the published method)
  adaptive  : adaptive group lasso. Stage 1 = base. Stage 2 = refit from scratch with the
              penalty sum_ij w_ij ||theta_ij||, w_ij = (||theta_ij|| + eps)^(-gamma) from stage 1
              (Zou 2006 adaptive lasso, group version): removes the shrinkage bias on strong
              edges while pruning weak ones harder.
  refit     : stage 2 = refit with NO penalty on edges kept by stage 1 (top-k by importance, k=2d),
              others masked; importance from refit (debiasing).
  boot      : mean importance over B=5 half-sample fits (stability / variance reduction).
  adaptive_boot : adaptive stage 2 averaged over B=5 half-samples.

SELECTION PROTOCOL: candidates are compared ONLY on validation seeds 200-219 (d=6,10) and
200-204 (d=20), disjoint from every seed used to report published results (0-4/0-2) or to tune
lambda_g earlier (100-119). The winner is then frozen and evaluated on fresh seeds 300-319.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from src.synthetic import generate_instantaneous_anm
from src.cdkan.causal_kan import CausalKANInstant
from src.cdkan.losses import _h_notears
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

torch.set_default_dtype(torch.float32)


def zscore(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def score(S, A):
    d = A.shape[0]; m = ~np.eye(d, dtype=bool)
    p = np.nan_to_num(np.abs(S)[m].astype(float)); t = A[m].astype(int)
    if p.max() > 0: p = p / p.max()
    au = roc_auc_score(t, p) if t.sum() and (1 - t).sum() else float("nan")
    ap = average_precision_score(t, p) if t.sum() else float("nan")
    f1 = max(f1_score(t, (p >= q).astype(int), zero_division=0) for q in np.unique(np.r_[0, p, 1]))
    return au, ap, f1


def fit(Xt, seed, lg=0.01, ep=400, weights=None, mask=None, smooth=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xt.shape[1]; m = CausalKANInstant(d, grid_size=8)
    if mask is not None: m.selfmask.mul_(mask.unsqueeze(-1))
    opt = torch.optim.Adam(m.parameters(), lr=5e-3); rho, al = 1.0, 0.0
    for e in range(ep):
        opt.zero_grad(); pred = m(Xt); mse = ((pred - Xt) ** 2).mean(); h = m.h()
        norms = torch.linalg.vector_norm(m.coef * m.selfmask, dim=2)
        pen = (norms * weights).sum() if weights is not None else norms.sum()
        if smooth > 0:
            c = m.coef * m.selfmask; d2 = c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]; pen = pen + (smooth / lg) * (d2 ** 2).sum()
        (mse + lg * pen + al * h + 0.5 * rho * h * h).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if (e + 1) % 50 == 0:
            with torch.no_grad(): hv = m.h().item()
            if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
    return m


def importance(m, Xt): return m.importance(Xt).numpy().T          # [cause, effect]


def base(X, seed, lg=0.01):
    Xt = torch.tensor(zscore(X)); return importance(fit(Xt, seed, lg), Xt)


def adaptive(X, seed, lg=0.01, gamma=1.0, lg2=None, eps=1e-2):
    Xt = torch.tensor(zscore(X)); m1 = fit(Xt, seed, lg)
    with torch.no_grad():
        n1 = torch.linalg.vector_norm(m1.coef * m1.selfmask, dim=2)
        w = (n1 + eps) ** (-gamma); w = w / w.mean()
    m2 = fit(Xt, seed, lg2 if lg2 is not None else lg, weights=w)
    return importance(m2, Xt)


def refit(X, seed, lg=0.01):
    Xt = torch.tensor(zscore(X)); d = Xt.shape[1]; m1 = fit(Xt, seed, lg)
    imp = m1.importance(Xt)                                          # [effect, cause]
    k = 2 * d; thr = torch.sort(imp.flatten(), descending=True)[0][min(k, imp.numel()) - 1]
    mask = (imp >= thr).float(); mask.fill_diagonal_(0)
    m2 = fit(Xt, seed, 0.0, mask=mask)
    S = importance(m2, Xt); return S


def boot(X, seed, B=5, inner=base):
    rng = np.random.RandomState(seed); n = len(X); S = 0
    for b in range(B):
        idx = rng.choice(n, n // 2, replace=False)
        s = inner(X[idx], seed * 100 + b); S = S + s / (s.max() + 1e-12)
    return S / B


def adaptive_boot(X, seed, B=5): return boot(X, seed, B, adaptive)


VARIANTS = {"base": base, "adaptive": adaptive, "refit": refit, "boot": boot, "adaptive_boot": adaptive_boot}

if __name__ == "__main__":
    import pandas as pd
    seeds = {6: range(200, 220), 10: range(200, 220), 20: range(200, 205)}
    only = sys.argv[1:] or list(VARIANTS)
    rows = []; out = os.path.join(os.path.dirname(__file__), "..", "experimental_results", "x1_validation_raw.csv")
    for d, ss in seeds.items():
        for s in ss:
            X, A = generate_instantaneous_anm(n_samples=1500, n_nodes=d, seed=s)
            for name in only:
                t0 = time.time(); au, ap, f1 = score(VARIANTS[name](X, s), A)
                rows.append(dict(d=d, seed=s, variant=name, auroc=au, auprc=ap, f1=f1, time_s=round(time.time() - t0, 1)))
                print(f"d={d} s{s} {name:14s} AUROC={au:.3f} AUPRC={ap:.3f} t={time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows); print(df.groupby(["d", "variant"])[["auroc", "auprc", "f1", "time_s"]].mean().round(3))
