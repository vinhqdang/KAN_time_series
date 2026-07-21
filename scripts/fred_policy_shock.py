"""
Real macro interventional study on FRED data (includes the Fed Funds rate).

Documented policy shock: the March 2022 Fed rate-hike liftoff. We test, with a
symmetric-window, placebo-controlled design (directionally agnostic):
  does CD-KAN's causal graph change MORE across the documented shock than across
  placebo (non-event) split dates?
We also report, candidly, the DIRECTION of change in the interest-rate block's
causal out-influence (a pre-registered economic prior we do NOT confirm).

Data: FRED daily series 2016-2024 (Fed Funds DFF, 2Y/3M/10Y yields, broad USD
index, S&P 500, WTI, VIX). Stationary representation: yields/rates/VIX -> first
differences; price indices -> log-returns.
Writes experimental_results/fred_policy_*.csv and figures/fred_policy.png.
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
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "macro_fred.csv")
DIFF = {"FEDFUNDS_D", "T10Y", "T2Y", "T3M", "VIX"}
W = 400; STEP = 20; MAXLAG = 5


def stationary(df):
    X = pd.DataFrame(index=df.index)
    for c in df.columns:
        X[c] = df[c].diff() if c in DIFF else np.log(df[c]).diff()
    return X.replace([np.inf, -np.inf], np.nan).dropna()


def zscore(A): return (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-8)
def wins(dd, w):
    Xs, Ys = [], []
    for i in range(len(dd) - w - 1): Xs.append(dd[i:i + w]); Ys.append(dd[i + w])
    return torch.tensor(np.array(Xs), dtype=torch.float32), torch.tensor(np.array(Ys), dtype=torch.float32)
def fit_imp(A, seed=0, ep=150):
    torch.manual_seed(seed); np.random.seed(seed)
    Xw, yw = wins(zscore(A), MAXLAG + 1); m = CausalKAN(A.shape[1], max_lag=MAXLAG, grid_size=8)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(ep):
        opt.zero_grad(); (((m(Xw) - yw) ** 2).mean() + 0.02 * m.group_lasso()).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    imp = m.importance(Xw)
    return imp.cpu().numpy() if torch.is_tensor(imp) else np.asarray(imp)


def biner(imp, q=0.8):
    d = imp.shape[0]; off = ~np.eye(d, dtype=bool)
    return (imp >= np.quantile(imp[off], q)) & off
def dissim(a, b):
    a = a.flatten().astype(bool); b = b.flatten().astype(bool); u = (a | b).sum()
    return 1 - ((a & b).sum() / u if u else 1.0)


def main():
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    X = stationary(df); names = list(X.columns); R = X.values.astype(np.float32); dates = X.index
    d = len(names); off = ~np.eye(d, dtype=bool)

    # symmetric-window dissimilarity across candidate split dates
    splits = list(range(W, len(R) - W, STEP))
    recs = []
    for s in splits:
        ip = fit_imp(R[s - W:s]); iq = fit_imp(R[s:s + W])
        recs.append(dict(date=dates[s], dissim=dissim(biner(ip), biner(iq))))
        print(f"  {dates[s].date()}: dissim={recs[-1]['dissim']:.3f}", flush=True)
    sd = pd.DataFrame(recs); sd.to_csv(os.path.join(RES, "fred_policy_curve.csv"), index=False)

    shock = pd.Timestamp("2022-03-16")
    row = sd.iloc[(sd.date - shock).abs().argmin()]
    placebo = sd[np.abs((sd.date - shock).dt.days) > 150]["dissim"].values
    pval = (1 + (placebo >= row["dissim"]).sum()) / (len(placebo) + 1)

    # direction: rate-block out-influence pre vs post (reported candidly)
    rates = [names.index(c) for c in ("FEDFUNDS_D", "T2Y", "T3M")]
    def outshare(imp, idxs):
        tot = imp[off].sum()
        return float(sum(imp[:, j][np.arange(d) != j].sum() for j in idxs) / (tot + 1e-8))
    pre = X[X.index < shock].values.astype(np.float32); post = X[X.index >= shock].values.astype(np.float32)
    pre_r = np.mean([outshare(fit_imp(pre, s), rates) for s in range(5)])
    post_r = np.mean([outshare(fit_imp(post, s), rates) for s in range(5)])

    summary = dict(shock="2022-03-16", shock_dissim=round(float(row["dissim"]), 3),
                   placebo_mean=round(float(placebo.mean()), 3), p_value=round(float(pval), 3),
                   rate_outshare_pre=round(float(pre_r), 3), rate_outshare_post=round(float(post_r), 3))
    pd.DataFrame([summary]).to_csv(os.path.join(RES, "fred_policy_summary.csv"), index=False)
    print("\nSUMMARY:", summary)

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(sd.date, sd.dissim, "-o", ms=3, color="#2c7bb6")
    ax.axvline(shock, color="#d73027", ls="--", lw=1.5)
    ax.text(shock, sd.dissim.max() * 0.97, "Fed liftoff 2022-03", rotation=90, va="top", fontsize=8)
    ax.axhline(sd.dissim.mean(), color="gray", ls=":", label="mean")
    ax.set_xlabel("split date"); ax.set_ylabel("pre/post graph dissimilarity")
    ax.set_title("CD-KAN structural change on FRED macro panel (incl. Fed Funds rate)",
                 fontweight="bold"); ax.legend(fontsize=8)
    fig.autofmt_xdate(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fred_policy.png"), dpi=200); plt.close()
    print("saved figures/fred_policy.png")


if __name__ == "__main__":
    main()
