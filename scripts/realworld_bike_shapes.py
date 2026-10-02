"""
SPADE spline-shape checks on the bike-sharing data (companion to
realworld_bike_sharing.py; criteria below were fixed in the same pre-registration).

Fit SPADE (same settings as the benchmark) on resample 0 (original data) and on the
5 block-bootstrap resamples, using each resample's seed. For each learned edge
phi_{effect,cause}(x) evaluated on the data range of the standardized cause:
  * workingday -> registered : expected POSITIVE   (phi(workingday=1) > phi(workingday=0))
  * workingday -> casual     : expected NEGATIVE
  * temp -> casual           : expected INCREASING then FLATTENING
        increasing:  phi(q90) - phi(q10) > 0
        flattening:  slope over the upper half of the temp range < slope over the lower half
  * temp -> registered       : reported with the same two criteria (not pre-registered as a
                               shape expectation, only as a positive edge)
Reports the number of the 6 fits satisfying each criterion and the edge's importance rank.
Writes experimental_results/realworld_bike_shapes.csv and figures/bike_shapes.png.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from realworld_bike_sharing import load_design, block_bootstrap, NAMES, IDX, B_RESAMPLES, OUT_DIR
from src.cdkan.causal_kan import CausalKANInstant

FIG = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures", "bike_shapes.png")


def fit(X, seed, ep=400, lg=0.01):
    Z = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    torch.set_default_dtype(torch.float32); torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(Z, dtype=torch.float32); m = CausalKANInstant(X.shape[1], grid_size=8)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3); rho, al = 1.0, 0.0
    for e in range(ep):
        opt.zero_grad(); pred = m(Xt); mse = ((pred - Xt) ** 2).mean(); h = m.h()
        (mse + lg * m.group_lasso() + al * h + 0.5 * rho * h * h).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if (e + 1) % 50 == 0:
            with torch.no_grad(): hv = m.h().item()
            if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
    return m, Xt, Z


def phi(m, effect, cause, xs):
    xs_t = torch.tensor(xs, dtype=torch.float32)
    b = m._bases(xs_t)                                   # [n, k]
    c = (m.coef * m.selfmask)[effect, cause]             # [k]
    return (b @ c).detach().numpy()


def main():
    X0 = load_design(); rows = []; curves = {}
    for r in range(B_RESAMPLES + 1):
        X = block_bootstrap(X0, r); m, Xt, Z = fit(X, r)
        imp = m.importance(Xt).numpy()                   # [effect, cause]
        flat = sorted([(imp[i, j], i, j) for i in range(12) for j in range(12) if i != j], reverse=True)
        rank = {(i, j): k + 1 for k, (_, i, j) in enumerate(flat)}
        wd = np.unique(Z[:, IDX["workingday"]])
        for cause, effect in [("workingday", "registered"), ("workingday", "casual"),
                              ("temp", "casual"), ("temp", "registered")]:
            c, e = IDX[cause], IDX[effect]
            rec = dict(resample=r, cause=cause, effect=effect, importance=float(imp[e, c]), rank=rank[(e, c)])
            if cause == "workingday":
                v = phi(m, e, c, wd); rec["delta"] = float(v[-1] - v[0])
                rec["sign_ok"] = bool((rec["delta"] > 0) if effect == "registered" else (rec["delta"] < 0))
            else:
                q = np.quantile(Z[:, c], [0.1, 0.5, 0.9]); g = np.linspace(q[0], q[2], 41); v = phi(m, e, c, g)
                lo = (phi(m, e, c, [q[1]])[0] - v[0]) / (q[1] - q[0]); hi = (v[-1] - phi(m, e, c, [q[1]])[0]) / (q[2] - q[1])
                rec.update(delta=float(v[-1] - v[0]), slope_lower=float(lo), slope_upper=float(hi),
                           increasing=bool(v[-1] - v[0] > 0), flattening=bool(hi < lo))
                curves[(r, cause, effect)] = (g, v - v.mean())
            rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT_DIR, "realworld_bike_shapes.csv"), index=False)
    print(df.round(3).to_string())
    for (cause, effect), g in df.groupby(["cause", "effect"]):
        if cause == "workingday":
            print(f"{cause}->{effect}: sign ok in {int(g.sign_ok.sum())}/6; median rank {g['rank'].median():.0f}")
        else:
            print(f"{cause}->{effect}: increasing {int(g.increasing.sum())}/6, flattening {int(g.flattening.sum())}/6, "
                  f"both {int((g.increasing & g.flattening).sum())}/6; median rank {g['rank'].median():.0f}")
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for k, (cause, effect, title) in enumerate([("temp", "casual", "temperature $\\to$ casual riders"),
                                                ("temp", "registered", "temperature $\\to$ registered riders")]):
        for r in range(B_RESAMPLES + 1):
            g, v = curves[(r, cause, effect)]
            ax[k].plot(g, v, color="C0" if r == 0 else "0.6", lw=2.2 if r == 0 else 1.0, alpha=1 if r == 0 else .8,
                       label="original data" if r == 0 else ("bootstrap resamples" if r == 1 else None))
        ax[k].set_title(title, fontsize=10); ax[k].set_xlabel("standardized temperature"); ax[k].set_ylabel("learned effect (centered)")
        ax[k].grid(alpha=.3)
    ax[0].legend(fontsize=8, frameon=False)
    sub = df[df.cause == "workingday"]
    for i, (effect, col) in enumerate([("casual", "C3"), ("registered", "C2")]):
        d = sub[sub.effect == effect].sort_values("resample")
        ax[2].bar(np.arange(6) + (i - .5) * 0.38, d.delta.values, 0.38, color=col, label=f"working day $\\to$ {effect}")
    ax[2].axhline(0, color="k", lw=.8); ax[2].set_xticks(range(6)); ax[2].set_xticklabels(["orig"] + [f"b{i}" for i in range(1, 6)])
    ax[2].set_title("working day: effect of 0 $\\to$ 1", fontsize=10); ax[2].legend(fontsize=8, frameon=False); ax[2].grid(alpha=.3, axis="y")
    ax[2].set_ylabel("change in learned effect")
    plt.tight_layout(); os.makedirs(os.path.dirname(FIG), exist_ok=True); plt.savefig(FIG, dpi=200); print("saved", FIG)


if __name__ == "__main__":
    main()
