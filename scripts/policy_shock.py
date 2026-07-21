"""
Real-data interventional case study: do documented policy shocks coincide with
larger structural change in CD-KAN's causal graph than placebo (non-event) dates?

Design (pre-registered, directionally agnostic, placebo-controlled):
  * Work on log-RETURNS (stationary; price levels give degenerate causal structure
    -- a limitation we verify and report, addressing the non-stationarity concern).
  * For each candidate split date T, learn the graph on the W days before and the
    W days after, and measure graph dissimilarity (1 - Jaccard).
  * Documented policy shocks: COVID / Fed emergency cut (2020-03) and the Fed
    rate-hike liftoff (2022-03).
  * Placebo null: all other candidate split dates. Report an empirical p-value =
    fraction of placebo splits with dissimilarity >= the documented-shock split.

Writes experimental_results/policy_shock.csv + figures/policy_shock.png.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_num_threads(4)
from src.cdkan.causal_kan import CausalKAN

RES = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
FIG = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "financial_2020_2025.csv")
W = 300           # trading days each side
STEP = 20
MAXLAG = 5


def zscore(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def wins(dd, w):
    X, Y = [], []
    for i in range(len(dd) - w - 1):
        X.append(dd[i:i + w]); Y.append(dd[i + w])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


def fit_imp(X, seed=0, epochs=120, lg=0.02):
    torch.manual_seed(seed); np.random.seed(seed)
    Xw, yw = wins(zscore(X), MAXLAG + 1)
    m = CausalKAN(X.shape[1], max_lag=MAXLAG, grid_size=8)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(epochs):
        opt.zero_grad(); (((m(Xw) - yw) ** 2).mean() + lg * m.group_lasso()).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m.importance(Xw).numpy()


def biner(imp, q=0.8):
    d = imp.shape[0]; off = ~np.eye(d, dtype=bool)
    return (imp >= np.quantile(imp[off], q)) & off


def dissim(a, b):
    a = a.flatten().astype(bool); b = b.flatten().astype(bool); u = (a | b).sum()
    return 1 - ((a & b).sum() / u if u else 1.0)


def main():
    df = pd.read_csv(DATA, index_col=0, skiprows=[1, 2])
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").ffill().dropna()
    ret = np.log(df / df.shift(1)).dropna()
    R = ret.values.astype(np.float32); dates = ret.index

    splits = list(range(W, len(R) - W, STEP))
    rows = []
    for s in splits:
        ip = fit_imp(R[s - W:s]); iq = fit_imp(R[s:s + W])
        rows.append(dict(idx=s, date=dates[s], dissim=dissim(biner(ip), biner(iq))))
        print(f"  split {dates[s].date()}: dissim={rows[-1]['dissim']:.3f}", flush=True)
    sd = pd.DataFrame(rows)
    sd.to_csv(os.path.join(RES, "policy_shock.csv"), index=False)

    def nearest(target):
        t = pd.Timestamp(target)
        k = (sd.date - t).abs().idxmin()
        return sd.loc[k]

    shocks = {"COVID / Fed cut (2020-03)": "2020-03-16",
              "Fed rate-hike liftoff (2022-03)": "2022-03-16"}
    results = []
    for label, dt in shocks.items():
        row = nearest(dt)
        placebo = sd[np.abs((sd.date - pd.Timestamp(dt)).dt.days) > 120]["dissim"].values
        pval = (1 + (placebo >= row["dissim"]).sum()) / (len(placebo) + 1)
        results.append(dict(shock=label, date=str(row["date"].date()),
                            dissim=round(float(row["dissim"]), 3),
                            placebo_mean=round(float(placebo.mean()), 3),
                            p_value=round(float(pval), 3)))
        print(f"{label}: dissim={row['dissim']:.3f} vs placebo mean {placebo.mean():.3f}  p={pval:.3f}")
    pd.DataFrame(results).to_csv(os.path.join(RES, "policy_shock_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sd.date, sd.dissim, "-o", ms=3, color="#2c7bb6")
    for label, dt in shocks.items():
        ax.axvline(pd.Timestamp(dt), color="#d73027", ls="--", lw=1.5)
        ax.text(pd.Timestamp(dt), sd.dissim.max() * 0.97, label.split(" (")[0],
                rotation=90, fontsize=8, va="top")
    ax.axhline(sd.dissim.mean(), color="gray", ls=":", label="mean")
    ax.set_xlabel("split date"); ax.set_ylabel("pre/post graph dissimilarity")
    ax.set_title("CD-KAN structural change vs documented policy shocks (log-returns)",
                 fontweight="bold"); ax.legend(fontsize=8)
    fig.autofmt_xdate(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "policy_shock.png"), dpi=200); plt.close()
    print("saved figures/policy_shock.png")


if __name__ == "__main__":
    main()
