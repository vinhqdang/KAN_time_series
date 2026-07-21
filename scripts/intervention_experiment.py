"""
Intervention / structural-change validation (Reviewer 3.4).

(A) GROUND-TRUTH change-point: concatenate two SVAR regimes with DIFFERENT causal
    graphs at a known break. A sliding-window CD-KAN should (i) show a spike in
    graph dissimilarity when the window straddles the break, and (ii) recover the
    correct regime graph on each side. This is a true-ground-truth version of the
    "does CD-KAN recover the expected change in causal structure?" test.

(B) REAL-DATA change-point: the same sliding-window graph-dissimilarity signal on
    the financial panel, checked against documented events (COVID-2020, 2022
    rate-hike onset).

Writes experimental_results/intervention_*.csv and figures/intervention.png.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_num_threads(4)
from src.synthetic import generate_nonlinear_scm, generate_svar_contemp
from src.cdkan.causal_kan import CausalKAN
from sklearn.metrics import roc_auc_score

RES = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
FIG = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "financial_2020_2025.csv")


def zscore(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def windows(dd, w):
    X, Y = [], []
    for i in range(len(dd) - w - 1):
        X.append(dd[i:i + w]); Y.append(dd[i + w])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


def fit_imp(X, seed=0, max_lag=3, epochs=120, lg=0.02):
    torch.manual_seed(seed); np.random.seed(seed)
    d = X.shape[1]
    Xw, yw = windows(zscore(X), max_lag + 1)
    m = CausalKAN(d, max_lag=max_lag, grid_size=8)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss = ((m(Xw) - yw) ** 2).mean() + lg * m.group_lasso()
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m.importance(Xw).numpy()


def biner(imp, q=0.8):
    d = imp.shape[0]; off = ~np.eye(d, dtype=bool)
    thr = np.quantile(imp[off], q)
    return (imp >= thr) & off


def dissim(a, b):
    a = a.flatten().astype(bool); b = b.flatten().astype(bool)
    u = (a | b).sum()
    return 1 - ((a & b).sum() / u if u else 1.0)


def auroc(imp, true):
    d = imp.shape[0]; m = ~np.eye(d, dtype=bool)
    p = np.nan_to_num(imp[m].astype(float)); t = true[m].astype(int)
    if p.max() > 0: p = p / p.max()
    return roc_auc_score(t, p) if t.sum() and (1 - t).sum() else float("nan")


def part_A():
    print("== (A) ground-truth change-point ==")
    d = 6; n = 1500
    # stable lagged SVAR regimes (contractive functions) with DIFFERENT graphs
    X1, _, G1 = generate_svar_contemp(n_samples=n, n_nodes=d, contemp_density=0.0,
                                      lag_density=0.22, max_lag=3, seed=2)
    X2, _, G2 = generate_svar_contemp(n_samples=n, n_nodes=d, contemp_density=0.0,
                                      lag_density=0.22, max_lag=3, seed=7)
    X = np.vstack([X1, X2]); brk = n
    w, stride = 600, 150
    centers, imps = [], []
    starts = list(range(0, len(X) - w, stride))
    for s in starts:
        imps.append(fit_imp(X[s:s + w], seed=0)); centers.append(s + w // 2)
    diss = [dissim(biner(imps[k]), biner(imps[k + 1])) for k in range(len(imps) - 1)]
    dcent = [(centers[k] + centers[k + 1]) / 2 for k in range(len(imps) - 1)]
    # recovery: windows fully in regime 1 vs regime 2
    early = [auroc(imps[k], G1) for k, s in enumerate(starts) if s + w <= brk]
    late = [auroc(imps[k], G2) for k, s in enumerate(starts) if s >= brk]
    peak = dcent[int(np.argmax(diss))]
    print(f"  break at t={brk}; dissimilarity peak at t={peak:.0f}")
    print(f"  regime-1 recovery AUROC (early windows)={np.nanmean(early):.3f}")
    print(f"  regime-2 recovery AUROC (late windows) ={np.nanmean(late):.3f}")
    pd.DataFrame(dict(center=dcent, dissimilarity=diss)).to_csv(
        os.path.join(RES, "intervention_synth.csv"), index=False)
    pd.DataFrame([dict(break_t=brk, peak_t=round(peak), regime1_auroc=round(np.nanmean(early), 3),
                       regime2_auroc=round(np.nanmean(late), 3))]).to_csv(
        os.path.join(RES, "intervention_synth_summary.csv"), index=False)
    return dcent, diss, brk, np.nanmean(early), np.nanmean(late)


def part_B():
    print("== (B) real-data change-point ==")
    df = pd.read_csv(DATA, index_col=0, skiprows=[1, 2])
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").ffill().dropna()
    X = df.values.astype(np.float32); dates = df.index
    w, stride = 300, 60
    centers, imps = [], []
    for s in range(0, len(X) - w, stride):
        imps.append(fit_imp(X[s:s + w], seed=0, max_lag=5)); centers.append(s + w // 2)
    diss = [dissim(biner(imps[k]), biner(imps[k + 1])) for k in range(len(imps) - 1)]
    dcent_dates = [dates[int((centers[k] + centers[k + 1]) / 2)] for k in range(len(imps) - 1)]
    pd.DataFrame(dict(date=dcent_dates, dissimilarity=diss)).to_csv(
        os.path.join(RES, "intervention_real.csv"), index=False)
    top = sorted(zip(diss, dcent_dates), reverse=True)[:3]
    print("  largest structural shifts near:", [str(d.date()) for _, d in top])
    return dcent_dates, diss, top


def main():
    dc, ds, brk, r1, r2 = part_A()
    rd, rds, top = part_B()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(dc, ds, "-o", color="#2c7bb6")
    ax[0].axvline(brk, color="#d73027", ls="--", lw=2, label=f"true break (t={brk})")
    ax[0].set_xlabel("window centre (t)"); ax[0].set_ylabel("graph dissimilarity (1-Jaccard)")
    ax[0].set_title(f"(A) Ground-truth change-point\nregime recovery AUROC: {r1:.2f} / {r2:.2f}",
                    fontweight="bold"); ax[0].legend()
    ax[1].plot(rd, rds, "-o", color="#2c7bb6")
    for ev, lab in [("2020-03-01", "COVID"), ("2022-03-01", "rate hikes")]:
        ax[1].axvline(pd.Timestamp(ev), color="#d73027", ls="--", lw=1.5)
        ax[1].text(pd.Timestamp(ev), max(rds) * 0.95, lab, rotation=90, fontsize=8, va="top")
    ax[1].set_xlabel("date"); ax[1].set_ylabel("graph dissimilarity (1-Jaccard)")
    ax[1].set_title("(B) Real financial data\nstructural shifts vs documented events", fontweight="bold")
    fig.autofmt_xdate()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "intervention.png"), dpi=200); plt.close()
    print("saved figures/intervention.png")


if __name__ == "__main__":
    main()
