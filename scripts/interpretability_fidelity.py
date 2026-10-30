"""
X3: put a NUMBER on the interpretability claim. The manuscript asserts that SPADE's spline edge
functions expose the *shape* of each dependency; here we measure how well the learned edge function
recovers the TRUE mechanism f_ce on synthetic non-linear ANM data (instantaneous generator,
true mechanisms in {tanh, sin, quadratic, rational}).

Protocol (fixed before running). Data: generate_instantaneous_anm(n=1500, d=6 and 10), seeds 400-419.
For each method, for every TRUE edge (c->e) that the method ranks within its top-|E_true| scored
edges (so only edges it actually 'finds' are judged), extract the learned curve phi(x) on
x in [-2, 2] (standardized cause; others held at 0) and compare it with the centred true curve f_ce:
  * shape R^2: R^2 of the best affine map a*phi+b onto f (1 = same shape up to scale/offset)
  * qualitative class agreement: class in {increasing, decreasing, non-monotone} from the sign pattern
    of the slope (threshold 5% of range), learned vs true.
Methods: SPADE (B-spline edges), MLP-edge (same bottleneck/penalty/budget, 1-3-1 tanh MLP edges),
linear (a NOTEARS-style linear SEM: edge curve = straight line; the 'scalar coefficient' view).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, torch, torch.nn as nn, pandas as pd
from src.synthetic import generate_instantaneous_anm
from src.cdkan.causal_kan import CausalKANInstant
from spade_variants import zscore, fit, importance
torch.set_default_dtype(torch.float32)


class MLPEdgeInstant(nn.Module):
    """Same bottleneck as SPADE, edge = 1->3->1 tanh MLP (10 parameters/edge incl. biases)."""
    def __init__(self, d, hidden=3):
        super().__init__(); self.d = d; self.hidden = hidden
        self.w1 = nn.Parameter(0.5 * torch.randn(d, d, hidden)); self.b1 = nn.Parameter(torch.zeros(d, d, hidden))
        self.w2 = nn.Parameter(0.3 * torch.randn(d, d, hidden)); self.bias = nn.Parameter(torch.zeros(d))
        self.register_buffer("selfmask", (1 - torch.eye(d)))
    def edge(self, x):                                   # x [n, d(cause)] -> [n, d(effect), d(cause)]
        z = torch.tanh(x[:, None, :, None] * self.w1[None] + self.b1[None])
        return (z * self.w2[None]).sum(-1) * self.selfmask[None]
    def forward(self, x): return self.edge(x).sum(2) + self.bias
    def group_lasso(self): return torch.linalg.vector_norm(self.w2 * self.selfmask[..., None], dim=2).sum()
    def h(self):
        from src.cdkan.losses import _h_notears
        return _h_notears(torch.linalg.vector_norm(self.w2 * self.selfmask[..., None], dim=2))
    @torch.no_grad()
    def importance(self, x): return self.edge(x).std(0)


def train(m, Xt, seed, lg=0.01, ep=400):
    torch.manual_seed(seed); opt = torch.optim.Adam(m.parameters(), lr=5e-3); rho, al = 1.0, 0.0
    for e in range(ep):
        opt.zero_grad(); mse = ((m(Xt) - Xt) ** 2).mean(); h = m.h()
        (mse + lg * m.group_lasso() + al * h + 0.5 * rho * h * h).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if (e + 1) % 50 == 0:
            hv = m.h().item()
            if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
    return m


def true_curve(seed, c, e, xs, d):
    """Re-create the generator's mechanism for edge c->e (same RNG consumption order)."""
    r = np.random.RandomState(seed); density = 2.0 / (d - 1); perm = r.permutation(d)
    A = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a):
            if r.rand() < density: A[perm[b], perm[a]] = 1
    fns = [lambda x: 0.8 * np.tanh(1.5 * x), lambda x: 0.7 * np.sin(1.5 * x),
           lambda x: 0.3 * (x ** 2 - 1.0), lambda x: 0.8 * x / (1.0 + np.abs(x))]
    f = {(a, b): fns[r.randint(len(fns))] for a in range(d) for b in range(d) if A[a, b]}
    return f[(c, e)](xs)


def r2_affine(phi, f):
    A = np.c_[phi, np.ones_like(phi)]; coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    res = f - A @ coef; return 1 - res.var() / (f.var() + 1e-12)


def cls(v):
    """qualitative class from total upward vs downward movement (a wiggle below 15% of the range is ignored)."""
    rng = v.max() - v.min() + 1e-12; dv = np.diff(v); up, dn = dv[dv > 0].sum() / rng, -dv[dv < 0].sum() / rng
    if min(up, dn) > 0.15: return "non-monotone"
    return "increasing" if up >= dn else "decreasing"


SMOOTH = float(os.environ.get("SMOOTH", 0)); SEEDS = range(int(os.environ.get("S0", 400)), int(os.environ.get("S1", 420))); OUT = os.environ.get("OUTCSV", "experimental_results/x3_fidelity_raw.csv")


def main():
    rows = []; xs = np.linspace(-2, 2, 81)
    for d in (6, 10):
        for seed in SEEDS:
            X, A = generate_instantaneous_anm(1500, d, seed=seed); Z = zscore(X); Xt = torch.tensor(Z)
            # standardization changes the x-axis scale of each cause; map xs (standardized) back to raw units
            mu, sd = X.mean(0), X.std(0) + 1e-8
            models = {}
            models["SPADE"] = fit(Xt, seed, 0.01)
            if SMOOTH:
                models["SPADE-smooth"] = fit(Xt, seed, 0.01, smooth=SMOOTH)
                imp1 = models["SPADE"].importance(Xt); thr = torch.sort(imp1.flatten(), descending=True)[0][2 * d - 1]
                mask = (imp1 >= thr).float(); mask.fill_diagonal_(0)
                models["SPADE-2stage"] = fit(Xt, seed, 0.001, mask=mask, smooth=SMOOTH)   # structure from stage 1, smooth display refit
            models["MLP-edge"] = train(MLPEdgeInstant(d), Xt, seed)
            Atrue = np.argwhere(A == 1)
            for name, m in models.items():
                imp = importance(models['SPADE'] if name == 'SPADE-2stage' else m, Xt)   # [cause, effect]; 2-stage ranks with stage 1
                k = len(Atrue); flat = np.argsort(-imp.flatten())[:k]
                found = {(i // d, i % d) for i in flat}
                for c, e in Atrue:
                    xt = torch.zeros(len(xs), d); xt[:, c] = torch.tensor(xs, dtype=torch.float32)
                    with torch.no_grad():
                        if name.startswith("SPADE"):
                            b = m._bases(xt); phi = (b[:, c, :] @ (m.coef * m.selfmask)[e, c]).numpy()
                        else:
                            phi = m.edge(xt)[:, e, c].numpy()
                    f = true_curve(seed, c, e, mu[c] + sd[c] * xs, d)
                    lin = np.polyfit(xs, f, 1); linear_phi = np.polyval(lin, xs)   # best possible linear (oracle slope)
                    rows.append(dict(d=d, seed=seed, method=name, c=c, e=e, found=(c, e) in found, r2=r2_affine(phi, f),
                                     cls_ok=cls(phi) == cls(f), linear_oracle_r2=r2_affine(linear_phi, f)))
            print(f"d={d} seed={seed} done", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    df["nonlinear"] = df.linear_oracle_r2 < 0.5       # edges a straight line cannot describe (sin, quadratic)
    print("all true edges"); print(df.groupby(["d", "method"]).agg(r2=("r2", "mean"), cls_ok=("cls_ok", "mean"), lin_oracle=("linear_oracle_r2", "mean"), found=("found", "mean"), n=("r2", "size")).round(3))
    print("strongly non-linear true edges only"); print(df[df.nonlinear].groupby(["d", "method"]).agg(r2=("r2", "mean"), cls_ok=("cls_ok", "mean"), lin_oracle=("linear_oracle_r2", "mean"), found=("found", "mean"), n=("r2", "size")).round(3))


if __name__ == "__main__":
    main()
